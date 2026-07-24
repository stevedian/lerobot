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
    """Diffusion Policy jointly trained with a LeWorldModel-style JEPA objective.

    The Diffusion policy and JEPA world model use separate observation encoders.
    During training, a latent-action head on the Diffusion U-Net conditions the
    world model. Inference evaluates only the policy vision encoder and U-Net.
    """

    # A deterministic crop is the safe default for temporal prediction. Sequence-
    # consistent random augmentation can be added without changing this model API.
    crop_is_random: bool = False

    jepa_latent_dim: int = 192
    jepa_predictor_layers: int = 4
    jepa_predictor_heads: int = 6
    jepa_predictor_mlp_ratio: float = 4.0
    jepa_predictor_dropout: float = 0.1
    jepa_prediction_horizon: int = 4
    jepa_action_latent_dim: int = 192

    jepa_world_model_loss_weight: float = 0.02
    jepa_sigreg_weight: float = 0.1
    jepa_sigreg_num_projections: int = 512
    jepa_sigreg_num_frequencies: int = 17
    jepa_loss_ramp_steps: int = 20_000

    # Legacy fields kept so checkpoints from the earlier JEPA-conditioned
    # inference design can still be parsed. They are not used by this model.
    jepa_shared_encoder_gradient_scale: float = 0.1
    jepa_condition_residual_scale: float = 0.1
    use_jepa_candidate_selection: bool = False
    jepa_num_action_candidates: int = 8
    jepa_candidate_horizon: int = 4
    jepa_goal_weight: float = 1.0
    jepa_smoothness_weight: float = 0.01
    jepa_action_bound_weight: float = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()

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

        max_future_actions = self.horizon - self.n_obs_steps + 1
        if not 0 < self.jepa_prediction_horizon <= max_future_actions:
            raise ValueError(
                "`jepa_prediction_horizon` must fit the future part of the Diffusion horizon. "
                f"Got {self.jepa_prediction_horizon=} and {max_future_actions=}."
            )
        if self.jepa_world_model_loss_weight < 0:
            raise ValueError(
                "`jepa_world_model_loss_weight` must be non-negative. "
                f"Got {self.jepa_world_model_loss_weight}."
            )
        if self.jepa_sigreg_weight < 0:
            raise ValueError(f"`jepa_sigreg_weight` must be non-negative. Got {self.jepa_sigreg_weight}.")
        if self.jepa_sigreg_num_projections <= 0:
            raise ValueError("`jepa_sigreg_num_projections` must be positive.")
        if self.jepa_sigreg_num_frequencies < 2:
            raise ValueError("`jepa_sigreg_num_frequencies` must be at least 2.")
        if self.jepa_loss_ramp_steps < 0:
            raise ValueError("`jepa_loss_ramp_steps` must be non-negative.")
        if self.use_jepa_candidate_selection:
            raise ValueError(
                "`use_jepa_candidate_selection` is no longer supported: JEPA is training-only and "
                "inference always uses Diffusion."
            )

    @property
    def observation_delta_indices(self) -> list[int]:
        if self.jepa_world_model_loss_weight == 0:
            return super().observation_delta_indices
        return list(range(1 - self.n_obs_steps, self.jepa_prediction_horizon + 1))
