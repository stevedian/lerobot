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

import math
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
from lerobot.policies.diffusion_jepa.modeling_lewm_predictor import LeWorldModelPredictor
from lerobot.policies.diffusion_jepa.modeling_vjepa2 import FrozenVJEPA2Encoder
from lerobot.policies.diffusion_jepa.processor_diffusion_jepa import (
    JEPA_IMAGES,
    raw_jepa_image_key,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class DiffusionVisionEncoder(nn.Module):
    """Observation encoder used exclusively by the deployed Diffusion policy."""

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


class DiffusionLatentActionQueries(nn.Module):
    """Learnable latent-action queries conditioned only on policy observations."""

    def __init__(
        self,
        config: DiffusionJEPAConfig,
        *,
        policy_feature_dim: int,
        num_transitions: int,
    ):
        super().__init__()
        action_dim = config.jepa_action_latent_dim
        num_queries = num_transitions * config.jepa_action_tokens_per_transition
        num_heads = math.gcd(action_dim, config.jepa_predictor_heads)

        self.num_transitions = num_transitions
        self.num_tokens_per_transition = config.jepa_action_tokens_per_transition
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, action_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.context_projection = nn.Linear(policy_feature_dim, action_dim)
        self.cross_attention = nn.MultiheadAttention(
            action_dim,
            num_heads=num_heads,
            dropout=config.jepa_predictor_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(action_dim)
        self.norm2 = nn.LayerNorm(action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, int(action_dim * config.jepa_predictor_mlp_ratio)),
            nn.GELU(),
            nn.Dropout(config.jepa_predictor_dropout),
            nn.Linear(int(action_dim * config.jepa_predictor_mlp_ratio), action_dim),
        )

    def forward(self, policy_features: Tensor) -> Tensor:
        if policy_features.ndim != 3:
            raise ValueError("Latent-action queries expect policy features shaped (B, T, D).")
        context = self.context_projection(policy_features)
        queries = self.query_tokens.expand(policy_features.shape[0], -1, -1)
        attended = self.cross_attention(queries, context, context, need_weights=False)[0]
        queries = queries + attended
        queries = queries + self.mlp(self.norm2(queries))
        queries = self.norm1(queries)
        return queries.unflatten(
            1,
            (self.num_transitions, self.num_tokens_per_transition),
        )


class DiffusionJEPAModel(nn.Module):
    """Diffusion policy jointly trained by frozen-V-JEPA2 world prediction."""

    def __init__(
        self,
        config: DiffusionJEPAConfig,
        *,
        jepa_encoder: FrozenVJEPA2Encoder | None = None,
    ):
        super().__init__()
        self.config = config
        self.vision_encoder = DiffusionVisionEncoder(config)

        if config.jepa_world_model_loss_weight > 0:
            if not config.image_features:
                raise ValueError("The V-JEPA2 world-model objective requires RGB image observations.")
            self.jepa_encoder = jepa_encoder or FrozenVJEPA2Encoder(config)
            num_transitions = self.jepa_encoder.num_state_steps - 1
            if num_transitions != config.jepa_prediction_horizon:
                raise ValueError(
                    "`jepa_prediction_horizon` must equal the number of V-JEPA2 state transitions. "
                    f"Expected {num_transitions}, got {config.jepa_prediction_horizon}."
                )
            state_dim = self.jepa_encoder.output_dim_per_view * len(config.image_features)
            self.world_model_predictor = LeWorldModelPredictor(
                config,
                state_dim=state_dim,
                num_steps=num_transitions,
                tokens_per_state=self.jepa_encoder.tokens_per_state,
            )
        else:
            self.jepa_encoder = None
            self.world_model_predictor = None
            num_transitions = config.jepa_prediction_horizon

        self.latent_action_queries = DiffusionLatentActionQueries(
            config,
            policy_feature_dim=self.vision_encoder.feature_dim,
            num_transitions=num_transitions,
        )
        self.latent_action_conditioner = nn.Sequential(
            nn.LayerNorm(config.jepa_action_latent_dim),
            nn.Linear(config.jepa_action_latent_dim, config.jepa_action_latent_dim),
            nn.SiLU(),
        )
        global_cond_dim = (
            self.vision_encoder.feature_dim * config.n_obs_steps
            + config.jepa_action_latent_dim
        )
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
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

    def _policy_history(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Exclude all JEPA future observations from the deployed policy path."""
        history = {}
        for key, value in batch.items():
            if isinstance(value, Tensor) and value.ndim >= 2 and key != ACTION:
                history[key] = value[:, : self.config.n_obs_steps]
            else:
                history[key] = value
        return history

    def _prepare_global_conditioning(
        self,
        policy_features: Tensor,
        latent_actions: Tensor,
    ) -> Tensor:
        if policy_features.shape[1] != self.config.n_obs_steps:
            raise ValueError(
                f"Diffusion-JEPA needs {self.config.n_obs_steps} policy observation steps. "
                f"Got {policy_features.shape[1]}."
            )
        latent_condition = self.latent_action_conditioner(latent_actions.mean(dim=(1, 2)))
        return torch.cat([policy_features.flatten(start_dim=1), latent_condition], dim=-1)

    def _encode_policy_condition(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        policy_features = self.vision_encoder(self._policy_history(batch))
        latent_actions = self.latent_action_queries(policy_features)
        global_cond = self._prepare_global_conditioning(policy_features, latent_actions)
        return global_cond, latent_actions

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
        expected_shape = (
            batch_size,
            self.config.horizon,
            self.config.action_feature.shape[0],
        )
        if sample.shape != expected_shape:
            raise ValueError(f"Diffusion noise must have shape {expected_shape}. Got {sample.shape}.")

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
        # V-JEPA2 and the world-model predictor are deliberately absent here.
        global_cond, _ = self._encode_policy_condition(batch)
        trajectory = self.conditional_sample(global_cond.shape[0], global_cond, noise=noise)
        start = self.config.n_obs_steps - 1
        return trajectory[:, start : start + self.config.n_action_steps]

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

    def _video_frame_valid_mask(self, batch: dict[str, Tensor], start: int, end: int) -> Tensor:
        masks = []
        for key in self.config.image_features:
            pad_key = f"{key}_is_pad"
            if pad_key in batch:
                masks.append(~batch[pad_key][:, start:end])
        if masks:
            return torch.stack(masks).all(dim=0)
        return torch.ones(
            batch[OBS_IMAGES].shape[0],
            end - start,
            dtype=torch.bool,
            device=batch[OBS_IMAGES].device,
        )

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if not set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"}):
            raise ValueError("Diffusion-JEPA training requires state, action, and action padding tensors.")
        if batch[ACTION].shape[1] != self.config.horizon:
            raise ValueError("Training action horizon does not match the policy config.")

        global_cond, latent_actions = self._encode_policy_condition(batch)
        diffusion_loss = self._diffusion_loss(batch, global_cond)
        if self.config.jepa_world_model_loss_weight == 0:
            return diffusion_loss, {
                "diffusion_loss": diffusion_loss.item(),
                "world_model_weight": 0.0,
            }
        if self.jepa_encoder is None or self.world_model_predictor is None:
            raise RuntimeError("The JEPA training modules were not initialized.")

        current_index = self.config.n_obs_steps - 1
        video_end = current_index + self.config.jepa_num_frames
        jepa_images = batch.get(JEPA_IMAGES, batch[OBS_IMAGES])
        if jepa_images.shape[1] < video_end:
            raise ValueError(
                "Training observations do not contain the complete V-JEPA2 video clip. "
                f"Need {video_end} steps, got {jepa_images.shape[1]}."
            )
        world_states = self.jepa_encoder(jepa_images[:, current_index:video_end])
        predicted_next = self.world_model_predictor(world_states[:, :-1], latent_actions)

        frame_valid = self._video_frame_valid_mask(batch, current_index, video_end)
        state_valid = frame_valid.unfold(1, self.jepa_encoder.tubelet_size, self.jepa_encoder.tubelet_size)
        state_valid = state_valid.all(dim=-1)
        transition_valid = state_valid[:, :-1] & state_valid[:, 1:]
        per_transition_loss = (predicted_next - world_states[:, 1:]).abs().mean(dim=(-1, -2))
        valid_count = transition_valid.sum()
        if valid_count.item() == 0:
            world_model_loss = predicted_next.sum() * 0
        else:
            world_model_loss = (per_transition_loss * transition_valid).sum() / valid_count

        weighted_world_model_loss = self.config.jepa_world_model_loss_weight * world_model_loss
        total_loss = diffusion_loss + weighted_world_model_loss
        metrics = {
            "diffusion_loss": diffusion_loss.item(),
            "world_model_loss": world_model_loss.item(),
            "weighted_world_model_loss": weighted_world_model_loss.item(),
            "world_model_weight": self.config.jepa_world_model_loss_weight,
            "world_state_std": world_states.float().std(dim=(0, 1, 2)).mean().item(),
            "latent_action_norm": latent_actions.detach().float().norm(dim=-1).mean().item(),
        }
        for step in range(per_transition_loss.shape[1]):
            step_loss = per_transition_loss[:, step][transition_valid[:, step]]
            metrics[f"world_model_loss_step_{step + 1}"] = (
                step_loss.mean().item() if step_loss.numel() else 0.0
            )
        return total_loss, metrics


class DiffusionJEPAPolicy(PreTrainedPolicy):
    """Diffusion Policy trained with a VLA-JEPA-style latent world model."""

    config_class = DiffusionJEPAConfig
    name = "diffusion_jepa"

    def __init__(
        self,
        config: DiffusionJEPAConfig,
        *,
        jepa_encoder: FrozenVJEPA2Encoder | None = None,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.diffusion_jepa = DiffusionJEPAModel(config, jepa_encoder=jepa_encoder)
        self.reset()

    def get_optim_params(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def state_dict(self, *args, **kwargs):
        """Do not duplicate the external frozen V-JEPA2 checkpoint in policy saves."""
        state = super().state_dict(*args, **kwargs)
        frozen_prefix = "diffusion_jepa.jepa_encoder.model."
        for key in [name for name in state if name.startswith(frozen_prefix)]:
            del state[key]
        return state

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
        raw_keys = [raw_jepa_image_key(key) for key in self.config.image_features]
        if all(key in batch for key in raw_keys):
            result[JEPA_IMAGES] = torch.stack([batch[key] for key in raw_keys], dim=-4)
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
        pass
