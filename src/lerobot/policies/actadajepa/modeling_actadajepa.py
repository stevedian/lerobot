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
"""ACT + AdaJEPA latent transition policy."""

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.actadajepa.configuration_actadajepa import ACTAdaJEPAConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

PlanInfo = dict[str, float | int | str | Tensor]


class ACTAdaJEPAPolicy(ACTPolicy):
    """ACT behavior cloning with an AdaJEPA-style latent dynamics auxiliary loss."""

    config_class = ACTAdaJEPAConfig
    name = "actadajepa"

    def __init__(self, config: ACTAdaJEPAConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.model = ACTAdaJEPA(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def reset(self):
        super().reset()
        self._last_action = None
        self._last_plan_info: dict[str, float | int | str] = {}

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select one action, optionally correcting ACT chunks with AdaJEPA Residual MPC."""
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions, plan_info = self.plan_action_chunk(batch)
            self._last_plan_info = self._public_plan_info(plan_info)
            action = self.temporal_ensembler.update(actions)
            self._last_action = action.detach()
            return action

        if len(self._action_queue) == 0:
            actions, plan_info = self.plan_action_chunk(batch)
            execution_horizon, horizon_info = self.choose_execution_horizon(
                batch=batch,
                plan_info=plan_info,
                action_chunk_size=actions.shape[1],
            )
            plan_info.update(horizon_info)
            self._last_plan_info = self._public_plan_info(plan_info)
            actions = actions[:, :execution_horizon]
            self._action_queue.extend(actions.transpose(0, 1))

        action = self._action_queue.popleft()
        self._last_action = action.detach()
        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions from the current observation.

        Training batches may contain observation pairs with shape ``(B, T, ...)``.
        Inference observations keep the usual ``(B, ...)`` shape. ACT always
        receives only the current observation.
        """
        self.eval()
        current_batch, _, _ = self._split_current_next_observations(batch)

        if self.config.image_features:
            current_batch = dict(current_batch)
            current_batch[OBS_IMAGES] = [current_batch[key] for key in self.config.image_features]

        actions = self.model(current_batch)[0]
        return actions

    @torch.no_grad()
    def plan_action_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, PlanInfo]:
        """Predict an ACT chunk and optionally apply AdaJEPA residual correction."""
        act_actions = self.predict_action_chunk(batch)
        if not self.config.use_residual_mpc or self.config.mpc_num_iters == 0:
            plan_info: PlanInfo = {"mpc_enabled": 0, "mpc_horizon": 0}
            self._add_adaptive_horizon_signals(batch, act_actions, plan_info)
            return act_actions, plan_info

        corrected_actions, plan_info = self.optimize_residual_actions(batch, act_actions)
        self._add_adaptive_horizon_signals(batch, corrected_actions, plan_info)
        return corrected_actions, plan_info

    def optimize_residual_actions(
        self,
        batch: dict[str, Tensor],
        act_actions: Tensor,
    ) -> tuple[Tensor, PlanInfo]:
        """Optimize a small action residual through the AdaJEPA latent dynamics model.

        The policy weights are kept fixed. Only a temporary `delta_action`
        tensor is optimized, then clipped around ACT's original action chunk.
        """
        current_batch = self._prepare_current_jepa_batch(batch)
        horizon = self.config.mpc_horizon or self.config.n_action_steps
        horizon = min(horizon, act_actions.shape[1])
        if horizon <= 0:
            return act_actions, {"mpc_enabled": 0, "mpc_horizon": 0}

        act_prefix = act_actions[:, :horizon].detach()
        with torch.enable_grad():
            initial_latent = self.model.encode_jepa_observation(current_batch).detach()
            target_latent, target_source = self._resolve_mpc_target_latent(
                batch=batch,
                initial_latent=initial_latent,
                act_prefix=act_prefix,
            )

            delta = torch.zeros_like(act_prefix, requires_grad=True)
            last_costs: dict[str, Tensor] = {}

            for _ in range(self.config.mpc_num_iters):
                corrected_prefix = act_prefix + delta
                rollout = self.model.rollout_jepa_latents(initial_latent, corrected_prefix)
                total_cost, last_costs = self._residual_mpc_cost(
                    corrected_prefix=corrected_prefix,
                    delta=delta,
                    rollout=rollout,
                    target_latent=target_latent,
                )
                grad = torch.autograd.grad(total_cost, delta, allow_unused=False)[0]
                with torch.no_grad():
                    delta -= self.config.mpc_lr * grad
                    if self.config.mpc_residual_clip > 0:
                        delta.clamp_(-self.config.mpc_residual_clip, self.config.mpc_residual_clip)
                delta.requires_grad_(True)

            corrected_prefix = (act_prefix + delta.detach()).to(dtype=act_actions.dtype)

        corrected_actions = act_actions.clone()
        corrected_actions[:, :horizon] = corrected_prefix
        residual_norm = delta.detach().norm(dim=-1).mean()
        info: dict[str, float | int | str] = {
            "mpc_enabled": 1,
            "mpc_horizon": horizon,
            "mpc_target_source": target_source,
            "mpc_residual_norm": float(residual_norm.item()),
        }
        for name, value in last_costs.items():
            info[f"mpc_{name}"] = float(value.detach().item())
        return corrected_actions, info

    def choose_execution_horizon(
        self,
        batch: dict[str, Tensor],
        plan_info: PlanInfo,
        action_chunk_size: int,
    ) -> tuple[int, dict[str, float | int | str]]:
        """Choose how many actions from the current chunk should be executed before replanning."""
        max_horizon = min(self.config.n_action_steps, action_chunk_size)
        if self.config.max_execution_horizon is not None:
            max_horizon = min(max_horizon, self.config.max_execution_horizon)
        min_horizon = min(self.config.min_execution_horizon, max_horizon)

        if not self.config.use_adaptive_horizon:
            return max_horizon, {
                "adaptive_horizon_enabled": 0,
                "adaptive_execution_horizon": max_horizon,
                "adaptive_replan_reason": "fixed_n_action_steps",
            }

        execution_horizon = max_horizon
        reason = "max_horizon"

        uncertainty = plan_info.get("_adaptive_uncertainty")
        if (
            isinstance(uncertainty, Tensor)
            and self.config.uncertainty_threshold is not None
            and uncertainty.numel() > 0
        ):
            threshold_index = self._first_threshold_index(uncertainty, self.config.uncertainty_threshold)
            if threshold_index is not None:
                execution_horizon = min(execution_horizon, threshold_index + 1)
                reason = "uncertainty_threshold"

        prediction_error = plan_info.get("_adaptive_prediction_error")
        if (
            isinstance(prediction_error, Tensor)
            and self.config.prediction_error_threshold is not None
            and prediction_error.numel() > 0
        ):
            threshold_index = self._first_threshold_index(
                prediction_error, self.config.prediction_error_threshold
            )
            if threshold_index is not None:
                execution_horizon = min(execution_horizon, threshold_index + 1)
                reason = "prediction_error_threshold"

        if self.config.contact_replan and self._detect_contact_or_force(batch):
            execution_horizon = min(execution_horizon, self.config.contact_replan_horizon)
            reason = "contact_or_force"

        execution_horizon = max(min_horizon, min(execution_horizon, max_horizon))
        return execution_horizon, {
            "adaptive_horizon_enabled": 1,
            "adaptive_execution_horizon": execution_horizon,
            "adaptive_min_execution_horizon": min_horizon,
            "adaptive_max_execution_horizon": max_horizon,
            "adaptive_replan_reason": reason,
        }

    def _add_adaptive_horizon_signals(
        self,
        batch: dict[str, Tensor],
        actions: Tensor,
        plan_info: PlanInfo,
    ) -> None:
        if not self.config.use_adaptive_horizon:
            return

        horizon = min(actions.shape[1], self.config.n_action_steps)
        if self.config.max_execution_horizon is not None:
            horizon = min(horizon, self.config.max_execution_horizon)
        if horizon <= 0:
            return

        current_batch = self._prepare_current_jepa_batch(batch)
        action_prefix = actions[:, :horizon].detach()
        with torch.no_grad():
            initial_latent = self.model.encode_jepa_observation(current_batch)
            rollout = self.model.rollout_jepa_latents(initial_latent, action_prefix)

        previous_latents = torch.cat([initial_latent[:, None], rollout[:, :-1]], dim=1)
        uncertainty = (rollout - previous_latents).norm(dim=-1).mean(dim=0)
        plan_info["_adaptive_uncertainty"] = uncertainty.detach()
        plan_info["adaptive_uncertainty_mean"] = float(uncertainty.mean().item())
        plan_info["adaptive_uncertainty_max"] = float(uncertainty.max().item())

        target_latent, target_source = self._resolve_explicit_mpc_target_latent(batch, initial_latent)
        if target_latent is None:
            plan_info["adaptive_prediction_error_source"] = "none"
            return

        prediction_error = (rollout - target_latent[:, None]).pow(2).mean(dim=-1).mean(dim=0)
        plan_info["_adaptive_prediction_error"] = prediction_error.detach()
        plan_info["adaptive_prediction_error_source"] = target_source
        plan_info["adaptive_prediction_error_mean"] = float(prediction_error.mean().item())
        plan_info["adaptive_prediction_error_final"] = float(prediction_error[-1].item())

    def _resolve_explicit_mpc_target_latent(
        self,
        batch: dict[str, Tensor],
        initial_latent: Tensor,
    ) -> tuple[Tensor | None, str]:
        goal_key = self.config.mpc_goal_batch_key
        if goal_key in batch and isinstance(batch[goal_key], Tensor):
            goal_latent = batch[goal_key].to(device=initial_latent.device, dtype=initial_latent.dtype)
            return self._expand_goal_latent(goal_latent, initial_latent).detach(), "batch_goal_latent"

        if self.config.mpc_goal_latent is not None:
            goal_latent = torch.tensor(
                self.config.mpc_goal_latent,
                device=initial_latent.device,
                dtype=initial_latent.dtype,
            )
            return self._expand_goal_latent(goal_latent, initial_latent).detach(), "config_goal_latent"

        return None, "none"

    def _first_threshold_index(self, values: Tensor, threshold: float) -> int | None:
        exceeded = torch.nonzero(values > threshold, as_tuple=False)
        if exceeded.numel() == 0:
            return None
        return int(exceeded[0, 0].item())

    def _detect_contact_or_force(self, batch: dict[str, Tensor]) -> bool:
        contact = batch.get(self.config.contact_batch_key)
        if isinstance(contact, Tensor) and bool(contact.detach().bool().any().item()):
            return True

        if self.config.force_threshold is None:
            return False

        force = batch.get(self.config.force_batch_key)
        if not isinstance(force, Tensor):
            return False

        force = force.detach().to(dtype=torch.float32)
        force_norm = force.norm().unsqueeze(0) if force.ndim == 1 else force.flatten(start_dim=1).norm(dim=-1)
        return bool((force_norm > self.config.force_threshold).any().item())

    def _public_plan_info(self, plan_info: PlanInfo) -> dict[str, float | int | str]:
        return {key: value for key, value in plan_info.items() if not isinstance(value, Tensor)}

    def _prepare_current_jepa_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        current_batch, _, _ = self._split_current_next_observations(batch)
        if self.config.image_features:
            current_batch = dict(current_batch)
            current_batch[OBS_IMAGES] = [current_batch[key] for key in self.config.image_features]
        return current_batch

    def _resolve_mpc_target_latent(
        self,
        batch: dict[str, Tensor],
        initial_latent: Tensor,
        act_prefix: Tensor,
    ) -> tuple[Tensor, str]:
        goal_key = self.config.mpc_goal_batch_key
        if goal_key in batch and isinstance(batch[goal_key], Tensor):
            goal_latent = batch[goal_key].to(device=initial_latent.device, dtype=initial_latent.dtype)
            goal_latent = self._expand_goal_latent(goal_latent, initial_latent)
            return goal_latent.detach(), "batch_goal_latent"

        if self.config.mpc_goal_latent is not None:
            goal_latent = torch.tensor(
                self.config.mpc_goal_latent,
                device=initial_latent.device,
                dtype=initial_latent.dtype,
            )
            goal_latent = self._expand_goal_latent(goal_latent, initial_latent)
            return goal_latent.detach(), "config_goal_latent"

        with torch.no_grad():
            act_rollout = self.model.rollout_jepa_latents(initial_latent, act_prefix)
        return act_rollout[:, -1].detach(), "act_rollout_endpoint"

    def _expand_goal_latent(self, goal_latent: Tensor, initial_latent: Tensor) -> Tensor:
        if goal_latent.ndim == 1:
            goal_latent = goal_latent.unsqueeze(0)
        if goal_latent.shape[0] == 1 and initial_latent.shape[0] != 1:
            goal_latent = goal_latent.expand(initial_latent.shape[0], -1)
        if goal_latent.shape != initial_latent.shape:
            raise ValueError(
                "`mpc_goal_latent` must have shape "
                f"{tuple(initial_latent.shape)} or {(initial_latent.shape[-1],)}. "
                f"Got {tuple(goal_latent.shape)}."
            )
        return goal_latent

    def _residual_mpc_cost(
        self,
        corrected_prefix: Tensor,
        delta: Tensor,
        rollout: Tensor,
        target_latent: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        goal_cost = F.mse_loss(rollout[:, -1], target_latent)
        bc_cost = delta.pow(2).mean()
        smooth_cost = self._smoothness_cost(corrected_prefix)
        uncertainty_cost = rollout.new_zeros(())
        force_cost = rollout.new_zeros(())

        total_cost = (
            self.config.mpc_goal_weight * goal_cost
            + self.config.mpc_bc_weight * bc_cost
            + self.config.mpc_smooth_weight * smooth_cost
            + self.config.mpc_uncertainty_weight * uncertainty_cost
            + self.config.mpc_force_weight * force_cost
        )
        return total_cost, {
            "total_cost": total_cost,
            "goal_cost": goal_cost,
            "bc_cost": bc_cost,
            "smooth_cost": smooth_cost,
            "uncertainty_cost": uncertainty_cost,
            "force_cost": force_cost,
        }

    def _smoothness_cost(self, actions: Tensor) -> Tensor:
        costs = []
        if self._last_action is not None:
            previous_action = self._last_action.to(device=actions.device, dtype=actions.dtype)
            costs.append((actions[:, 0] - previous_action).pow(2).mean())
        if actions.shape[1] > 1:
            costs.append((actions[:, 1:] - actions[:, :-1]).pow(2).mean())
        if not costs:
            return actions.new_zeros(())
        return torch.stack(costs).mean()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run ACT loss plus JEPA latent transition prediction loss."""
        current_batch, next_batch, next_is_valid = self._split_current_next_observations(batch)

        if self.config.image_features:
            current_batch = dict(current_batch)
            current_batch[OBS_IMAGES] = [current_batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(current_batch)

        l1_loss = (
            F.l1_loss(current_batch[ACTION], actions_hat, reduction="none")
            * ~current_batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        if self.config.jepa_loss_weight > 0 and next_batch:
            if self.config.image_features:
                next_batch = dict(next_batch)
                next_batch[OBS_IMAGES] = [next_batch[key] for key in self.config.image_features]

            jepa_action = current_batch[ACTION][:, 0]
            if "action_is_pad" in current_batch:
                next_is_valid = next_is_valid & ~current_batch["action_is_pad"][:, 0]

            jepa_loss = self.model.jepa_prediction_loss(current_batch, jepa_action, next_batch, next_is_valid)
            loss_dict["jepa_loss"] = jepa_loss.item()
            loss = loss + self.config.jepa_loss_weight * jepa_loss

        return loss, loss_dict

    def _split_current_next_observations(
        self, batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        """Split observation tensors shaped as ``(B, T, ...)`` into current and next observations."""
        current_batch = dict(batch)
        next_batch: dict[str, Tensor] = {}
        valid_masks = []

        for key, feature in self.config.input_features.items():
            if key not in batch or not isinstance(batch[key], Tensor):
                continue

            value = batch[key]
            has_time_axis = value.ndim == len(feature.shape) + 2
            if not has_time_axis:
                continue

            current_batch[key] = value[:, 0]
            if value.shape[1] < 2:
                continue

            next_batch[key] = value[:, 1]
            pad_key = f"{key}_is_pad"
            if pad_key in batch and isinstance(batch[pad_key], Tensor):
                valid_masks.append(~batch[pad_key][:, 1])

        if valid_masks:
            next_is_valid = torch.stack(valid_masks, dim=0).all(dim=0)
        else:
            any_tensor = next((v for v in batch.values() if isinstance(v, Tensor)), None)
            if any_tensor is None:
                raise ValueError("ACTAdaJEPA received a batch without tensor values.")
            next_is_valid = torch.ones(any_tensor.shape[0], dtype=torch.bool, device=any_tensor.device)

        return current_batch, next_batch, next_is_valid


class ACTAdaJEPA(ACT):
    """ACT network with a shared-encoder JEPA transition head."""

    def __init__(self, config: ACTAdaJEPAConfig):
        super().__init__(config)
        self.config = config

        action_dim = self.config.action_feature.shape[0]
        predictor_hidden_dim = config.jepa_predictor_hidden_dim or config.dim_feedforward

        self.jepa_latent_norm = nn.LayerNorm(config.dim_model)
        self.jepa_action_encoder = nn.Sequential(
            nn.Linear(action_dim, config.dim_model),
            nn.LayerNorm(config.dim_model),
            nn.GELU(),
        )
        self.jepa_predictor = nn.Sequential(
            nn.Linear(config.dim_model * 2, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, config.dim_model),
        )

    def encode_jepa_observation(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode an observation into the latent used by the JEPA predictor."""
        tokens = []

        if self.config.robot_state_feature:
            tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.reduce(cam_features, "b c h w -> b c", "mean")
                tokens.append(cam_features)

        if not tokens:
            raise ValueError("ACTAdaJEPA requires at least one image, robot state, or environment state.")

        latent = torch.stack(tokens, dim=0).mean(dim=0)
        return self.jepa_latent_norm(latent)

    def predict_next_jepa_latent(self, batch: dict[str, Tensor], action: Tensor) -> Tensor:
        current_latent = self.encode_jepa_observation(batch)
        return self.predict_next_jepa_latent_from_latent(current_latent, action)

    def predict_next_jepa_latent_from_latent(self, latent: Tensor, action: Tensor) -> Tensor:
        action_latent = self.jepa_action_encoder(action)
        return self.jepa_predictor(torch.cat([latent, action_latent], dim=-1))

    def rollout_jepa_latents(self, initial_latent: Tensor, actions: Tensor) -> Tensor:
        """Roll out AdaJEPA latent dynamics under an action chunk.

        Args:
            initial_latent: Current latent observation with shape `(B, dim_model)`.
            actions: Action sequence with shape `(B, H, action_dim)`.

        Returns:
            Predicted future latents with shape `(B, H, dim_model)`.
        """
        latents = []
        latent = initial_latent
        for step in range(actions.shape[1]):
            latent = self.predict_next_jepa_latent_from_latent(latent, actions[:, step])
            latents.append(latent)
        if not latents:
            return initial_latent.new_empty(initial_latent.shape[0], 0, initial_latent.shape[-1])
        return torch.stack(latents, dim=1)

    def jepa_prediction_loss(
        self,
        current_batch: dict[str, Tensor],
        action: Tensor,
        next_batch: dict[str, Tensor],
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        pred_next_latent = self.predict_next_jepa_latent(current_batch, action)
        target_next_latent = self.encode_jepa_observation(next_batch)
        if self.config.jepa_stop_gradient_target:
            target_next_latent = target_next_latent.detach()

        per_sample_loss = F.mse_loss(pred_next_latent, target_next_latent, reduction="none").mean(dim=-1)
        if valid_mask is None:
            return per_sample_loss.mean()

        valid = valid_mask.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
        if valid.sum() == 0:
            return per_sample_loss.new_zeros(())
        return (per_sample_loss * valid).sum() / valid.sum().clamp_min(1)
