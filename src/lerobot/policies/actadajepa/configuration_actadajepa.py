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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("actadajepa")
@dataclass
class ACTAdaJEPAConfig(ACTConfig):
    """ACT policy augmented with an AdaJEPA-style latent transition objective.

    The policy keeps ACT's action chunking behavior and adds a lightweight latent
    world-model head trained from adjacent observation transitions:

        z_t = encoder(obs_t)
        z_hat_t+1 = predictor(z_t, action_t)
        loss_jepa = mse(z_hat_t+1, stop_grad(encoder(obs_t+1)))

    This does not turn ACT into an MPC planner by itself. It gives the policy a
    trainable latent dynamics model that can be used for AdaJEPA-style online
    adaptation or planning integrations.
    """

    # Weight of the JEPA latent transition prediction objective.
    jepa_loss_weight: float = 0.1

    # Dataset offsets used to fetch observation pairs for the JEPA objective.
    # ACT still receives the t=0 observation; t=1 is used only as the latent target.
    jepa_observation_delta_indices: list[int] | None = field(default_factory=lambda: [0, 1])

    # Hidden dimension of the JEPA predictor MLP. Defaults to ACT's feed-forward width.
    jepa_predictor_hidden_dim: int | None = None

    # Stop-gradient target branch, following JEPA / AdaJEPA anti-collapse practice.
    jepa_stop_gradient_target: bool = True

    # Residual MPC inference. Disabled by default to preserve ACTAdaJEPA's
    # existing behavior unless explicitly enabled at eval/deployment time.
    use_residual_mpc: bool = False
    mpc_num_iters: int = 5
    mpc_lr: float = 1e-2
    mpc_horizon: int | None = None
    mpc_residual_clip: float = 0.05

    mpc_goal_weight: float = 1.0
    mpc_bc_weight: float = 0.1
    mpc_smooth_weight: float = 0.05
    mpc_uncertainty_weight: float = 0.0
    mpc_force_weight: float = 0.0

    # Goal latent source for Residual MPC. During inference, callers may pass a
    # tensor under this key in the observation batch. If absent, `mpc_goal_latent`
    # is used. If both are absent, the ACT rollout endpoint is used as a neutral
    # target, making the optimizer conservative.
    mpc_goal_batch_key: str = "jepa_goal_latent"
    mpc_goal_latent: list[float] | None = None

    # Adaptive execution horizon. Disabled by default. When enabled,
    # ACTAdaJEPA still predicts a full action chunk, but only enqueues the
    # prefix considered reliable by AdaJEPA rollout signals and optional
    # contact/force observations.
    use_adaptive_horizon: bool = False
    min_execution_horizon: int = 1
    max_execution_horizon: int | None = None
    uncertainty_threshold: float | None = 0.2
    prediction_error_threshold: float | None = 0.2

    contact_replan: bool = True
    contact_replan_horizon: int = 1
    contact_batch_key: str = "contact"
    force_batch_key: str = "observation.force"
    force_threshold: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.jepa_loss_weight < 0:
            raise ValueError(f"`jepa_loss_weight` must be non-negative. Got {self.jepa_loss_weight}.")
        if self.jepa_observation_delta_indices is not None:
            if len(self.jepa_observation_delta_indices) < 2:
                raise ValueError("`jepa_observation_delta_indices` must contain at least current and next.")
            if self.jepa_observation_delta_indices[0] != 0:
                raise ValueError("`jepa_observation_delta_indices` must start at 0 for ACT compatibility.")
        if self.mpc_num_iters < 0:
            raise ValueError(f"`mpc_num_iters` must be non-negative. Got {self.mpc_num_iters}.")
        if self.mpc_lr < 0:
            raise ValueError(f"`mpc_lr` must be non-negative. Got {self.mpc_lr}.")
        if self.mpc_horizon is not None and self.mpc_horizon <= 0:
            raise ValueError(f"`mpc_horizon` must be positive when set. Got {self.mpc_horizon}.")
        if self.mpc_residual_clip < 0:
            raise ValueError(f"`mpc_residual_clip` must be non-negative. Got {self.mpc_residual_clip}.")
        for name in [
            "mpc_goal_weight",
            "mpc_bc_weight",
            "mpc_smooth_weight",
            "mpc_uncertainty_weight",
            "mpc_force_weight",
        ]:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"`{name}` must be non-negative. Got {value}.")
        if self.min_execution_horizon <= 0:
            raise ValueError(
                f"`min_execution_horizon` must be positive. Got {self.min_execution_horizon}."
            )
        if self.max_execution_horizon is not None and self.max_execution_horizon <= 0:
            raise ValueError(
                f"`max_execution_horizon` must be positive when set. Got {self.max_execution_horizon}."
            )
        if self.contact_replan_horizon <= 0:
            raise ValueError(
                f"`contact_replan_horizon` must be positive. Got {self.contact_replan_horizon}."
            )
        for name in ["uncertainty_threshold", "prediction_error_threshold", "force_threshold"]:
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"`{name}` must be non-negative when set. Got {value}.")

    @property
    def observation_delta_indices(self) -> list[int] | None:
        if self.jepa_loss_weight == 0:
            return None
        return self.jepa_observation_delta_indices
