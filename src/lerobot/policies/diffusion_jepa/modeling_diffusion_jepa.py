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


class DiffusionVisionEncoder(nn.Module):
    """Observation encoder used exclusively by the Diffusion policy."""

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

        self.feature_dim = raw_feature_dim

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
        """Return per-step features used as Diffusion global conditioning."""
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

        return torch.cat(features, dim=-1)


class JEPAEncoder(DiffusionVisionEncoder):
    """Independent training-only encoder for current and future JEPA states."""

    def __init__(self, config: DiffusionJEPAConfig):
        super().__init__(config)
        self.adapter = nn.Sequential(
            nn.Linear(self.feature_dim, config.jepa_latent_dim),
            nn.SiLU(),
            nn.LayerNorm(config.jepa_latent_dim),
        )
        self.projector = nn.Linear(config.jepa_latent_dim, config.jepa_latent_dim, bias=False)
        self.projector_norm = nn.BatchNorm1d(config.jepa_latent_dim)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        features = super().forward(batch)
        projected = self.projector(self.adapter(features))
        return _stepwise_batch_norm(self.projector_norm, projected)


class DiffusionJEPAModel(nn.Module):
    """Diffusion action policy with a training-only LeWorldModel objective."""

    def __init__(self, config: DiffusionJEPAConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = DiffusionVisionEncoder(config)
        self.jepa_encoder = JEPAEncoder(config)
        self.unet = DiffusionConditionalUnet1d(
            config,
            global_cond_dim=self.vision_encoder.feature_dim * config.n_obs_steps,
        )
        self.latent_action_head = nn.Sequential(
            nn.LayerNorm(config.down_dims[0]),
            nn.Linear(config.down_dims[0], config.jepa_action_latent_dim),
            nn.SiLU(),
            nn.Linear(config.jepa_action_latent_dim, config.jepa_action_latent_dim),
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

    def _prepare_global_conditioning(self, policy_features: Tensor) -> Tensor:
        """Flatten policy history without depending on the JEPA branch."""
        if policy_features.shape[1] < self.config.n_obs_steps:
            raise ValueError(
                f"Diffusion-JEPA needs {self.config.n_obs_steps} observation steps. "
                f"Got {policy_features.shape[1]}."
            )
        return policy_features[:, : self.config.n_obs_steps].flatten(start_dim=1)

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
        # The deployed path deliberately never evaluates the JEPA encoder,
        # latent-action head, world model, or SIGReg modules.
        policy_features = self.vision_encoder(batch)
        global_cond = self._prepare_global_conditioning(policy_features)
        trajectory = self.conditional_sample(policy_features.shape[0], global_cond, noise=noise)
        start = self.config.n_obs_steps - 1
        return trajectory[:, start : start + self.config.n_action_steps]

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

    def _diffusion_loss(
        self,
        batch: dict[str, Tensor],
        global_cond: Tensor,
        *,
        return_latent_actions: bool,
    ) -> tuple[Tensor, Tensor | None]:
        trajectory = batch[ACTION]
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        if return_latent_actions:
            prediction, denoising_features = self.unet(
                noisy_trajectory,
                timesteps,
                global_cond=global_cond,
                return_features=True,
            )
            latent_actions = self.latent_action_head(denoising_features)
        else:
            prediction = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
            latent_actions = None
        target = noise if self.config.prediction_type == "epsilon" else trajectory
        loss = F.mse_loss(prediction, target, reduction="none")
        if self.config.do_mask_loss_for_padding:
            loss = loss * (~batch["action_is_pad"]).unsqueeze(-1)
        return loss.mean(), latent_actions

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

        policy_features = self.vision_encoder(batch)
        global_cond = self._prepare_global_conditioning(policy_features)
        world_weight = self._current_world_model_weight()
        diffusion_loss, latent_actions = self._diffusion_loss(
            batch,
            global_cond,
            return_latent_actions=self.config.jepa_world_model_loss_weight > 0,
        )
        if self.config.jepa_world_model_loss_weight == 0:
            return diffusion_loss, {
                "diffusion_loss": diffusion_loss.item(),
                "world_model_weight": 0.0,
            }

        if latent_actions is None:
            raise RuntimeError("JEPA training requires Diffusion latent actions.")
        jepa_latents = self.jepa_encoder(batch)
        current_index = self.config.n_obs_steps - 1
        end_index = current_index + self.config.jepa_prediction_horizon + 1
        if jepa_latents.shape[1] < end_index:
            raise ValueError(
                "Training observations do not contain all JEPA future targets. "
                f"Need {end_index} steps, got {jepa_latents.shape[1]}."
            )
        world_latents = jepa_latents[:, current_index:end_index]
        action_start = self.config.n_obs_steps - 1
        world_latent_actions = latent_actions[
            :,
            action_start : action_start + self.config.jepa_prediction_horizon,
        ]
        predicted_next = self.world_model_predictor(world_latents[:, :-1], world_latent_actions)

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
            "latent_action_norm": world_latent_actions.detach().float().norm(dim=-1).mean().item(),
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
    """Diffusion Policy trained with an auxiliary LeWorldModel objective."""

    config_class = DiffusionJEPAConfig
    name = "diffusion_jepa"

    def __init__(self, config: DiffusionJEPAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.diffusion_jepa = DiffusionJEPAModel(config)
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
