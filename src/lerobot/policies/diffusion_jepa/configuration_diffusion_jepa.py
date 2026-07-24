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

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("diffusion_jepa")
@dataclass
class DiffusionJEPAConfig(DiffusionConfig):
    """Diffusion Policy jointly trained with a VLA-JEPA-style world objective.

    A trainable policy vision encoder produces latent-action queries used by both
    Diffusion and a training-only world-model predictor. A separate pretrained
    V-JEPA2 encoder supplies frozen video-state targets.
    """

    # A deterministic crop is the safe default for temporal prediction. Sequence-
    # consistent random augmentation can be added without changing this model API.
    crop_is_random: bool = False

    jepa_encoder_name_or_path: str | None = None
    jepa_num_frames: int = 8
    jepa_latent_dim: int = 1024
    jepa_predictor_layers: int = 12
    jepa_predictor_heads: int = 8
    jepa_predictor_mlp_ratio: float = 4.0
    jepa_predictor_dropout: float = 0.0
    jepa_prediction_horizon: int = 3
    jepa_action_latent_dim: int = 1024
    jepa_action_tokens_per_transition: int = 8

    # VLA-JEPA uses 0.1 for robot-data joint training.
    jepa_world_model_loss_weight: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.jepa_num_frames < 2:
            raise ValueError(f"`jepa_num_frames` must be at least 2. Got {self.jepa_num_frames}.")
        if self.jepa_latent_dim <= 0:
            raise ValueError(f"`jepa_latent_dim` must be positive. Got {self.jepa_latent_dim}.")
        if self.jepa_predictor_layers <= 0:
            raise ValueError(f"`jepa_predictor_layers` must be positive. Got {self.jepa_predictor_layers}.")
        if self.jepa_predictor_heads <= 0:
            raise ValueError(f"`jepa_predictor_heads` must be positive. Got {self.jepa_predictor_heads}.")
        if self.jepa_latent_dim % self.jepa_predictor_heads != 0:
            raise ValueError(
                "`jepa_latent_dim` must be divisible by `jepa_predictor_heads`. "
                f"Got {self.jepa_latent_dim=} and {self.jepa_predictor_heads=}."
            )
        if self.jepa_predictor_mlp_ratio <= 0:
            raise ValueError(
                f"`jepa_predictor_mlp_ratio` must be positive. Got {self.jepa_predictor_mlp_ratio}."
            )
        if not 0 <= self.jepa_predictor_dropout < 1:
            raise ValueError(
                f"`jepa_predictor_dropout` must be in [0, 1). Got {self.jepa_predictor_dropout}."
            )
        if self.jepa_action_latent_dim <= 0:
            raise ValueError(
                f"`jepa_action_latent_dim` must be positive. Got {self.jepa_action_latent_dim}."
            )
        if self.jepa_action_tokens_per_transition <= 0:
            raise ValueError(
                "`jepa_action_tokens_per_transition` must be positive. "
                f"Got {self.jepa_action_tokens_per_transition}."
            )

        if self.jepa_prediction_horizon <= 0:
            raise ValueError(
                "`jepa_prediction_horizon` must be positive. "
                f"Got {self.jepa_prediction_horizon}."
            )
        if self.jepa_world_model_loss_weight < 0:
            raise ValueError(
                "`jepa_world_model_loss_weight` must be non-negative. "
                f"Got {self.jepa_world_model_loss_weight}."
            )

    @property
    def observation_delta_indices(self) -> list[int]:
        if self.jepa_world_model_loss_weight == 0:
            return super().observation_delta_indices
        return list(range(1 - self.n_obs_steps, self.jepa_num_frames))
