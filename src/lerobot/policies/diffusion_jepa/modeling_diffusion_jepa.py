#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionRgbEncoder,
    _make_noise_scheduler,
)
from lerobot.policies.diffusion_jepa.configuration_diffusion_jepa import DiffusionJEPAConfig
from lerobot.policies.diffusion_jepa.modeling_lewm_predictor import (
    LeWorldModelPredictor,
    _stepwise_batch_norm,
)
from lerobot.policies.diffusion_jepa.sigreg import SIGReg
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class SharedObservationEncoder(nn.Module):
    """Encode each multimodal observation into the shared LeWorldModel latent."""

    def __init__(self, config: DiffusionJEPAConfig):
        super().__init__()
        self.config = config

        raw_feature_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                raw_feature_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                raw_feature_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            raw_feature_dim += config.env_state_feature.shape[0]

        self.raw_feature_dim = raw_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(raw_feature_dim, config.jepa_latent_dim),
            nn.SiLU(),
            nn.LayerNorm(config.jepa_latent_dim),
        )
        # LeWorldModel places a one-layer MLP + BatchNorm projector after the
        # backbone normalization so SIGReg can shape the output distribution.
        self.projector = nn.Linear(config.jepa_latent_dim, config.jepa_latent_dim, bias=False)
        self.projector_norm = nn.BatchNorm1d(config.jepa_latent_dim)

    def _encode_images(self, images: Tensor, batch_size: int, sequence_length: int) -> Tensor:
        if self.config.use_separate_rgb_encoder_per_camera:
            images_per_camera = einops.rearrange(images, "b s n ... -> n (b s) ...")
            image_features = torch.cat(
                [
                    encoder(camera_images)
                    for encoder, camera_images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
            )
            return einops.rearrange(
                image_features,
                "(n b s) ... -> b s (n ...)",
                b=batch_size,
                s=sequence_length,
            )

        image_features = self.rgb_encoder(einops.rearrange(images, "b s n ... -> (b s n) ..."))
        return einops.rearrange(
            image_features,
            "(b s n) ... -> b s (n ...)",
            b=batch_size,
            s=sequence_length,
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        if state.ndim != 3:
            raise ValueError(f"Expected `{OBS_STATE}` shaped (B, T, D). Got {state.shape}.")
        batch_size, sequence_length = state.shape[:2]
        features = [state]

        if self.config.image_features:
            images = batch[OBS_IMAGES]
            if images.shape[:2] != (batch_size, sequence_length):
                raise ValueError("Image and state sequences must have matching batch/time axes.")
            features.append(self._encode_images(images, batch_size, sequence_length))
        if self.config.env_state_feature:
            features.append(batch[OBS_ENV_STATE])

        fused = self.fusion(torch.cat(features, dim=-1))
        projected = self.projector(fused)
        return _stepwise_batch_norm(self.projector_norm, projected)


class DiffusionJEPAModel(nn.Module):
    """Diffusion action generator and LeWorldModel sharing one observation encoder."""

    def __init__(self, config: DiffusionJEPAConfig):
        super().__init__()
        self.config = config
        self.observation_encoder = SharedObservationEncoder(config)
        self.diffusion_condition_projection = nn.Linear(
            config.jepa_latent_dim,
            self.observation_encoder.raw_feature_dim,
        )
        self.unet = DiffusionConditionalUnet1d(
            config,
            global_cond_dim=self.observation_encoder.raw_feature_dim * config.n_obs_steps,
        )
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            self.noise_scheduler.config.num_train_timesteps
            if config.num_inference_steps is None
            else config.num_inference_steps
        )

        self.world_model_predictor = LeWorldModelPredictor(config)
        self.sigreg = SIGReg(
            num_projections=config.jepa_sigreg_num_projections,
            num_frequencies=config.jepa_sigreg_num_frequencies,
        )
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def _prepare_global_conditioning(self, latents: Tensor) -> Tensor:
        if latents.shape[1] < self.config.n_obs_steps:
            raise ValueError(
                f"Diffusion-JEPA needs {self.config.n_obs_steps} observation steps. Got {latents.shape[1]}."
            )
        history = latents[:, : self.config.n_obs_steps]
        return self.diffusion_condition_projection(history).flatten(start_dim=1)

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        sample = (
            noise
            if noise is not None
            else torch.randn(
                batch_size,
                self.config.horizon,
                self.config.action_feature.shape[0],
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
        if sample.shape != (
            batch_size,
            self.config.horizon,
            self.config.action_feature.shape[0],
        ):
            raise ValueError("Diffusion noise has an incompatible shape.")

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], timestep, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(
                model_output,
                timestep,
                sample,
                generator=generator,
            ).prev_sample
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        latents = self.observation_encoder(batch)
        global_cond = self._prepare_global_conditioning(latents)
        trajectory = self.conditional_sample(latents.shape[0], global_cond, noise=noise)
        start = self.config.n_obs_steps - 1
        return trajectory[:, start : start + self.config.n_action_steps]

    def generate_action_candidates(
        self,
        batch: dict[str, Tensor],
        num_candidates: int | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return executed action candidates and the current shared latent."""
        candidate_count = self.config.jepa_num_action_candidates if num_candidates is None else num_candidates
        if candidate_count < 1:
            raise ValueError("At least one candidate is required.")

        latents = self.observation_encoder(batch)
        current_latent = latents[:, self.config.n_obs_steps - 1]
        global_cond = self._prepare_global_conditioning(latents).repeat_interleave(candidate_count, dim=0)
        batch_size = latents.shape[0]

        flat_noise = None
        if noise is not None:
            if noise.ndim == 4:
                expected_prefix = (batch_size, candidate_count)
                if noise.shape[:2] != expected_prefix:
                    raise ValueError(
                        f"Candidate noise must start with {expected_prefix}. Got {noise.shape[:2]}."
                    )
                flat_noise = noise.flatten(0, 1)
            else:
                flat_noise = noise

        trajectories = self.conditional_sample(
            batch_size * candidate_count,
            global_cond,
            noise=flat_noise,
        ).reshape(batch_size, candidate_count, self.config.horizon, -1)
        start = self.config.n_obs_steps - 1
        chunks = trajectories[:, :, start : start + self.config.n_action_steps]
        return chunks, current_latent

    def rollout_action_candidates(self, initial_latent: Tensor, candidates: Tensor) -> Tensor:
        if candidates.ndim != 4:
            raise ValueError("Action candidates must be shaped (B, K, H, A).")
        batch_size, candidate_count, horizon = candidates.shape[:3]
        if horizon < 1 or candidates.shape[-1] != self.config.action_feature.shape[0]:
            raise ValueError("Action candidates have an incompatible horizon or action dimension.")
        if initial_latent.shape != (batch_size, self.config.jepa_latent_dim):
            raise ValueError("Initial candidate latent has an incompatible shape.")
        flat_latent = initial_latent.repeat_interleave(candidate_count, dim=0)
        flat_actions = candidates.flatten(0, 1)
        rollout = self.world_model_predictor.rollout(flat_latent, flat_actions)
        return rollout.reshape(batch_size, candidate_count, horizon, -1)

    def score_action_candidates(
        self,
        initial_latent: Tensor,
        candidates: Tensor,
        goal_latent: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Score candidates against an explicit goal; lower is better."""
        horizon = self.config.jepa_candidate_horizon
        if candidates.shape[2] < horizon:
            raise ValueError(
                f"Candidate chunk has {candidates.shape[2]} steps, but scoring requires {horizon}."
            )
        rollout = self.rollout_action_candidates(initial_latent, candidates[:, :, :horizon])
        if goal_latent.ndim == 1:
            goal_latent = goal_latent.unsqueeze(0).expand_as(initial_latent)
        if goal_latent.shape != initial_latent.shape:
            raise ValueError(f"Goal latent must have shape {initial_latent.shape}. Got {goal_latent.shape}.")

        goal_cost = (rollout[:, :, -1] - goal_latent[:, None]).square().mean(dim=-1)
        if horizon > 1:
            smoothness_cost = candidates[:, :, 1:horizon].sub(candidates[:, :, : horizon - 1]).square()
            smoothness_cost = smoothness_cost.mean(dim=(-1, -2))
        else:
            smoothness_cost = torch.zeros_like(goal_cost)
        action_bound_cost = F.relu(candidates[:, :, :horizon].abs() - 1).square().mean(dim=(-1, -2))
        total = (
            self.config.jepa_goal_weight * goal_cost
            + self.config.jepa_smoothness_weight * smoothness_cost
            + self.config.jepa_action_bound_weight * action_bound_cost
        )
        return total, {
            "goal_cost": goal_cost,
            "smoothness_cost": smoothness_cost,
            "action_bound_cost": action_bound_cost,
            "latent_rollout": rollout,
        }

    def select_action_candidate(
        self,
        batch: dict[str, Tensor],
        goal_latent: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        candidates, current_latent = self.generate_action_candidates(batch, noise=noise)
        scores, diagnostics = self.score_action_candidates(current_latent, candidates, goal_latent)
        selected_index = scores.argmin(dim=1)
        gather_index = selected_index[:, None, None, None].expand(
            -1,
            1,
            candidates.shape[2],
            candidates.shape[3],
        )
        selected = candidates.gather(1, gather_index).squeeze(1)
        diagnostics.update({"scores": scores, "selected_index": selected_index})
        return selected, diagnostics

    def _observation_valid_mask(self, batch: dict[str, Tensor], sequence_length: int) -> Tensor:
        masks = []
        for key in self.config.input_features:
            pad_key = f"{key}_is_pad"
            if pad_key in batch and isinstance(batch[pad_key], Tensor):
                mask = batch[pad_key]
                if mask.ndim == 2 and mask.shape[1] >= sequence_length:
                    masks.append(~mask[:, :sequence_length])
        if masks:
            return torch.stack(masks, dim=0).all(dim=0)
        return torch.ones(
            batch[OBS_STATE].shape[0],
            sequence_length,
            dtype=torch.bool,
            device=batch[OBS_STATE].device,
        )

    def _diffusion_loss(self, batch: dict[str, Tensor], global_cond: Tensor) -> Tensor:
        trajectory = batch[ACTION]
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        prediction = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
        target = noise if self.config.prediction_type == "epsilon" else trajectory
        loss = F.mse_loss(prediction, target, reduction="none")
        if self.config.do_mask_loss_for_padding:
            loss = loss * (~batch["action_is_pad"]).unsqueeze(-1)
        return loss.mean()

    def _current_world_model_weight(self) -> float:
        if self.config.jepa_loss_ramp_steps == 0:
            return self.config.jepa_world_model_loss_weight
        ramp = min(self.num_updates.item() / self.config.jepa_loss_ramp_steps, 1.0)
        return self.config.jepa_world_model_loss_weight * ramp

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if not set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"}):
            raise ValueError("Diffusion-JEPA training requires state, action, and action padding tensors.")
        if OBS_IMAGES not in batch and OBS_ENV_STATE not in batch:
            raise ValueError("Diffusion-JEPA requires image or environment-state observations.")
        if batch[ACTION].shape[1] != self.config.horizon:
            raise ValueError("Training action horizon does not match the policy config.")

        latents = self.observation_encoder(batch)
        diffusion_loss = self._diffusion_loss(batch, self._prepare_global_conditioning(latents))
        world_weight = self._current_world_model_weight()
        if self.config.jepa_world_model_loss_weight == 0:
            return diffusion_loss, {
                "diffusion_loss": diffusion_loss.item(),
                "world_model_weight": 0.0,
            }

        current_index = self.config.n_obs_steps - 1
        end_index = current_index + self.config.jepa_prediction_horizon + 1
        if latents.shape[1] < end_index:
            raise ValueError(
                "Training observations do not contain all JEPA future targets. "
                f"Need {end_index} steps, got {latents.shape[1]}."
            )
        world_latents = latents[:, current_index:end_index]
        action_start = self.config.n_obs_steps - 1
        world_actions = batch[ACTION][
            :,
            action_start : action_start + self.config.jepa_prediction_horizon,
        ]
        predicted_next = self.world_model_predictor(world_latents[:, :-1], world_actions)

        observation_valid = self._observation_valid_mask(batch, end_index)[:, current_index:end_index]
        action_valid = ~batch["action_is_pad"][
            :,
            action_start : action_start + self.config.jepa_prediction_horizon,
        ]
        transition_valid = observation_valid[:, :-1] & observation_valid[:, 1:] & action_valid
        per_transition_loss = (predicted_next - world_latents[:, 1:]).square().mean(dim=-1)
        valid_count = transition_valid.sum()
        if valid_count.item() == 0:
            prediction_loss = predicted_next.sum() * 0
        else:
            prediction_loss = (per_transition_loss * transition_valid).sum() / valid_count

        sigreg_loss = self.sigreg(world_latents, observation_valid)
        world_model_loss = prediction_loss + self.config.jepa_sigreg_weight * sigreg_loss
        total_loss = diffusion_loss + world_weight * world_model_loss
        metrics = {
            "diffusion_loss": diffusion_loss.item(),
            "lewm_prediction_loss": prediction_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
            "weighted_world_model_loss": (world_weight * world_model_loss).item(),
            "world_model_weight": world_weight,
            "latent_feature_std": world_latents.detach().float().std(dim=(0, 1)).mean().item(),
        }
        for step in range(per_transition_loss.shape[1]):
            step_valid = transition_valid[:, step]
            step_loss = per_transition_loss[:, step][step_valid]
            metrics[f"lewm_prediction_loss_step_{step + 1}"] = (
                step_loss.mean().item() if step_loss.numel() else 0.0
            )
        return total_loss, metrics

    @torch.no_grad()
    def update(self) -> None:
        self.num_updates.add_(1)


class DiffusionJEPAPolicy(PreTrainedPolicy):
    """Diffusion Policy with a jointly trained LeWorldModel latent world model."""

    config_class = DiffusionJEPAConfig
    name = "diffusion_jepa"

    def __init__(self, config: DiffusionJEPAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.diffusion_jepa = DiffusionJEPAModel(config)
        self._goal_latent: Tensor | None = None
        self.reset()

    def get_optim_params(self):
        return self.diffusion_jepa.parameters()

    def reset(self) -> None:
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        self._goal_latent = None

    def set_goal_latent(self, goal_latent: Tensor) -> None:
        """Set an explicit goal latent required by candidate selection."""
        self._goal_latent = goal_latent

    @torch.no_grad()
    def encode_goal_observation(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode a goal observation so it can be passed to ``set_goal_latent``."""
        temporal_batch = self._stack_images(dict(batch))
        for key in (OBS_STATE, OBS_ENV_STATE, OBS_IMAGES):
            if key not in temporal_batch:
                continue
            value = temporal_batch[key]
            feature_rank = 1 if key in (OBS_STATE, OBS_ENV_STATE) else 4
            if value.ndim == feature_rank + 1:
                temporal_batch[key] = value.unsqueeze(1)
        return self.diffusion_jepa.observation_encoder(temporal_batch)[:, -1]

    @torch.no_grad()
    def set_goal_observation(self, batch: dict[str, Tensor]) -> None:
        """Encode and store an explicit goal observation for candidate selection."""
        self.set_goal_latent(self.encode_goal_observation(batch))

    def _stack_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.config.image_features:
            return batch
        result = dict(batch)
        result[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return result

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        queued_batch = {
            key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues
        }
        if self.config.use_jepa_candidate_selection:
            if self._goal_latent is None:
                raise ValueError(
                    "JEPA candidate selection requires an explicit goal. Call `set_goal_latent` first."
                )
            actions, _ = self.diffusion_jepa.select_action_candidate(
                queued_batch,
                self._goal_latent.to(queued_batch[OBS_STATE]),
                noise=noise,
            )
            return actions
        return self.diffusion_jepa.generate_actions(queued_batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = dict(batch)
        batch.pop(ACTION, None)
        batch = self._stack_images(batch)
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        batch = self._stack_images(batch)
        return self.diffusion_jepa.compute_loss(batch)

    @torch.no_grad()
    def update(self) -> None:
        self.diffusion_jepa.update()
