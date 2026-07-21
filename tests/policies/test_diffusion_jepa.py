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

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.policies.diffusion_jepa.configuration_diffusion_jepa import DiffusionJEPAConfig
from lerobot.policies.diffusion_jepa.modeling_diffusion_jepa import DiffusionJEPAModel, DiffusionJEPAPolicy
from lerobot.policies.diffusion_jepa.sigreg import SIGReg
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _make_config(**overrides) -> DiffusionJEPAConfig:
    kwargs = {
        "input_features": {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (3,)),
            OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (6,)),
        },
        "output_features": {ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        "n_obs_steps": 2,
        "horizon": 4,
        "n_action_steps": 2,
        "down_dims": (16, 32),
        "diffusion_step_embed_dim": 16,
        "n_groups": 4,
        "num_train_timesteps": 4,
        "num_inference_steps": 2,
        "jepa_latent_dim": 12,
        "jepa_predictor_layers": 2,
        "jepa_predictor_heads": 3,
        "jepa_predictor_mlp_ratio": 2.0,
        "jepa_predictor_dropout": 0.0,
        "jepa_prediction_horizon": 2,
        "jepa_world_model_loss_weight": 0.1,
        "jepa_sigreg_weight": 0.1,
        "jepa_sigreg_num_projections": 16,
        "jepa_sigreg_num_frequencies": 5,
        "jepa_loss_ramp_steps": 0,
        "jepa_num_action_candidates": 3,
        "jepa_candidate_horizon": 2,
        "jepa_smoothness_weight": 0.0,
        "jepa_action_bound_weight": 0.0,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return DiffusionJEPAConfig(**kwargs)


def _make_batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    # Observation offsets are [-1, 0, 1, 2], while action offsets are
    # [-1, 0, 1, 2]. JEPA must therefore start from action index 1.
    return {
        OBS_STATE: torch.randn(batch_size, 4, 3),
        OBS_ENV_STATE: torch.randn(batch_size, 4, 6),
        ACTION: torch.randn(batch_size, 4, 2).clamp(-1, 1),
        f"{OBS_STATE}_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        f"{OBS_ENV_STATE}_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


def test_config_requests_history_and_future_observations() -> None:
    config = _make_config()
    assert config.observation_delta_indices == [-1, 0, 1, 2]
    assert config.action_delta_indices == [-1, 0, 1, 2]

    with pytest.raises(ValueError, match="divisible"):
        _make_config(jepa_latent_dim=10)
    with pytest.raises(ValueError, match="executed action chunk"):
        _make_config(jepa_candidate_horizon=3)
    with pytest.raises(ValueError, match="residual_scale"):
        _make_config(jepa_condition_residual_scale=-0.1)


def test_joint_loss_backpropagates_through_one_shared_encoder() -> None:
    model = DiffusionJEPAModel(_make_config())
    model.train()
    loss, metrics = model.compute_loss(_make_batch())
    loss.backward()

    assert set(metrics).issuperset(
        {
            "diffusion_loss",
            "lewm_prediction_loss",
            "sigreg_loss",
            "latent_feature_std",
            "jepa_condition_residual_norm",
        }
    )
    assert metrics["jepa_condition_residual_norm"] == 0
    assert any(parameter.grad is not None for parameter in model.observation_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.world_model_predictor.parameters())
    assert model.jepa_condition_residual.weight.grad is not None
    assert model.jepa_condition_residual.weight.grad.abs().sum() > 0
    assert not any("target_encoder" in name for name, _ in model.named_modules())


def test_zero_initialized_jepa_residual_preserves_baseline_condition() -> None:
    model = DiffusionJEPAModel(_make_config())
    model.eval()
    batch = _make_batch()
    raw_features, latents = model.observation_encoder.forward_with_raw(batch)

    global_condition = model._prepare_global_conditioning(raw_features, latents)
    expected_baseline_condition = raw_features[:, : model.config.n_obs_steps].flatten(start_dim=1)

    torch.testing.assert_close(global_condition, expected_baseline_condition)
    assert torch.count_nonzero(model.jepa_condition_residual.weight) == 0
    assert torch.count_nonzero(model.jepa_condition_residual.bias) == 0


def test_jepa_residual_is_additive_after_adapter_update() -> None:
    model = DiffusionJEPAModel(_make_config())
    model.eval()
    batch = _make_batch()
    raw_features, latents = model.observation_encoder.forward_with_raw(batch)
    with torch.no_grad():
        model.jepa_condition_residual.weight.fill_(0.01)
        model.jepa_condition_residual.bias.fill_(0.02)

    global_condition = model._prepare_global_conditioning(raw_features, latents)
    residual = model.jepa_condition_residual(latents[:, : model.config.n_obs_steps])
    expected = (
        raw_features[:, : model.config.n_obs_steps] + model.config.jepa_condition_residual_scale * residual
    ).flatten(start_dim=1)
    torch.testing.assert_close(global_condition, expected)


def test_zero_residual_scale_keeps_exact_baseline_condition() -> None:
    model = DiffusionJEPAModel(_make_config(jepa_condition_residual_scale=0.0))
    model.eval()
    batch = _make_batch()
    raw_features, latents = model.observation_encoder.forward_with_raw(batch)
    with torch.no_grad():
        model.jepa_condition_residual.weight.normal_()
        model.jepa_condition_residual.bias.normal_()

    global_condition = model._prepare_global_conditioning(raw_features, latents)
    expected = raw_features[:, : model.config.n_obs_steps].flatten(start_dim=1)
    torch.testing.assert_close(global_condition, expected)


def test_zero_initialized_residual_matches_diffusion_actions_with_shared_unet() -> None:
    config = _make_config(num_inference_steps=2)
    baseline = DiffusionModel(config)
    model = DiffusionJEPAModel(config)
    model.unet.load_state_dict(baseline.unet.state_dict())
    baseline.eval()
    model.eval()

    batch = {
        OBS_STATE: torch.randn(2, config.n_obs_steps, 3),
        OBS_ENV_STATE: torch.randn(2, config.n_obs_steps, 6),
    }
    noise = torch.randn(2, config.horizon, 2)

    torch.manual_seed(0)
    baseline_actions = baseline.generate_actions(batch, noise=noise.clone())
    torch.manual_seed(0)
    jepa_actions = model.generate_actions(batch, noise=noise.clone())
    torch.testing.assert_close(baseline_actions, jepa_actions, rtol=0, atol=0)


def test_world_model_uses_actions_starting_at_current_timestep() -> None:
    model = DiffusionJEPAModel(_make_config())
    batch = _make_batch()
    captured_actions = None

    def capture_actions(_module, inputs, _output):
        nonlocal captured_actions
        captured_actions = inputs[1].detach().clone()

    hook = model.world_model_predictor.register_forward_hook(capture_actions)
    model.compute_loss(batch)
    hook.remove()

    assert captured_actions is not None
    torch.testing.assert_close(captured_actions, batch[ACTION][:, 1:3])


def test_shared_encoder_does_not_leak_future_through_batch_norm() -> None:
    model = DiffusionJEPAModel(_make_config())
    model.train()
    batch = _make_batch()
    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch[OBS_STATE][:, 2:] += 100
    changed_batch[OBS_ENV_STATE][:, 2:] -= 100

    torch.manual_seed(0)
    original = model.observation_encoder(batch)
    torch.manual_seed(0)
    changed = model.observation_encoder(changed_batch)
    torch.testing.assert_close(original[:, :2], changed[:, :2])


def test_predictor_is_causal_and_action_adaln_learns() -> None:
    model = DiffusionJEPAModel(_make_config())
    predictor = model.world_model_predictor
    predictor.eval()
    latents = torch.randn(2, 2, 12)
    actions = torch.randn(2, 2, 2, requires_grad=True)

    original = predictor(latents, actions)
    changed_latents = latents.clone()
    changed_latents[:, 1] += 100
    changed = predictor(changed_latents, actions)
    torch.testing.assert_close(original[:, 0], changed[:, 0])

    original.square().mean().backward()
    modulation = predictor.blocks[0].action_modulation[-1]
    assert modulation.weight.grad is not None
    assert modulation.weight.grad.abs().sum() > 0


def test_sigreg_penalizes_collapse_and_has_finite_gradients() -> None:
    sigreg = SIGReg(num_projections=64, num_frequencies=17)
    collapsed = torch.zeros(2048, 1, 8)
    gaussian = torch.randn(2048, 1, 8)

    torch.manual_seed(0)
    collapsed_loss = sigreg(collapsed)
    torch.manual_seed(0)
    gaussian_loss = sigreg(gaussian)
    assert collapsed_loss > gaussian_loss

    embeddings = torch.randn(8, 3, 8, requires_grad=True)
    loss = sigreg(embeddings)
    loss.backward()
    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_candidate_rollout_and_goal_scoring_are_batched() -> None:
    model = DiffusionJEPAModel(_make_config())
    model.eval()
    # Break the intentional zero action-conditioning initialization so this unit
    # test can construct candidates with distinct predicted consequences.
    with torch.no_grad():
        for block in model.world_model_predictor.blocks:
            block.action_modulation[-1].weight.normal_(std=0.1)

    initial_latent = torch.randn(2, 12)
    candidates = torch.randn(2, 3, 2, 2).clamp(-1, 1)
    rollout = model.rollout_action_candidates(initial_latent, candidates)
    assert rollout.shape == (2, 3, 2, 12)

    goal = rollout[:, 0, -1]
    scores, diagnostics = model.score_action_candidates(initial_latent, candidates, goal)
    assert scores.shape == (2, 3)
    assert diagnostics["latent_rollout"].shape == (2, 3, 2, 12)
    assert torch.equal(scores.argmin(dim=1), torch.zeros(2, dtype=torch.long))


def test_loss_weight_ramp_is_checkpointed_in_model_state() -> None:
    model = DiffusionJEPAModel(_make_config(jepa_loss_ramp_steps=2))
    assert model._current_world_model_weight() == 0
    model.update()
    assert model._current_world_model_weight() == pytest.approx(0.05)
    assert "num_updates" in model.state_dict()


def test_policy_factory_and_inference_action_queue(tmp_path) -> None:
    assert get_policy_class("diffusion_jepa") is DiffusionJEPAPolicy
    assert isinstance(make_policy_config("diffusion_jepa"), DiffusionJEPAConfig)

    policy = DiffusionJEPAPolicy(_make_config())
    policy.eval()
    observation = {
        OBS_STATE: torch.randn(2, 3),
        OBS_ENV_STATE: torch.randn(2, 6),
    }
    noise = torch.randn(2, 4, 2)
    action = policy.select_action(observation, noise=noise)
    assert action.shape == (2, 2)

    save_directory = tmp_path / "diffusion_jepa"
    policy.save_pretrained(save_directory)
    loaded = DiffusionJEPAPolicy.from_pretrained(save_directory, config=policy.config)
    torch.testing.assert_close(list(policy.parameters()), list(loaded.parameters()), rtol=0, atol=0)
    assert loaded.diffusion_jepa.num_updates.item() == policy.diffusion_jepa.num_updates.item()


def test_candidate_selection_requires_and_accepts_explicit_goal() -> None:
    policy = DiffusionJEPAPolicy(_make_config(use_jepa_candidate_selection=True))
    policy.eval()
    observation = {
        OBS_STATE: torch.randn(2, 3),
        OBS_ENV_STATE: torch.randn(2, 6),
    }
    noise = torch.randn(2, 3, 4, 2)
    with pytest.raises(ValueError, match="explicit goal"):
        policy.select_action(observation, noise=noise)

    policy.reset()
    policy.set_goal_observation(observation)
    action = policy.select_action(observation, noise=noise)
    assert action.shape == (2, 2)


def test_image_observations_use_the_shared_encoder() -> None:
    image_key = "observation.images.front"
    config = _make_config(
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (3,)),
            image_key: PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        },
        crop_shape=None,
    )
    policy = DiffusionJEPAPolicy(config)
    batch = _make_batch(batch_size=2)
    batch.pop(OBS_ENV_STATE)
    batch.pop(f"{OBS_ENV_STATE}_is_pad")
    batch[image_key] = torch.rand(2, 4, 3, 32, 32)
    batch[f"{image_key}_is_pad"] = torch.zeros(2, 4, dtype=torch.bool)

    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert "lewm_prediction_loss" in metrics
