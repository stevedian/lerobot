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
from lerobot.policies.diffusion_jepa.modeling_diffusion_jepa import (
    DiffusionJEPAModel,
    DiffusionJEPAPolicy,
)
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
        "jepa_action_latent_dim": 8,
        "jepa_world_model_loss_weight": 0.1,
        "jepa_sigreg_weight": 0.1,
        "jepa_sigreg_num_projections": 16,
        "jepa_sigreg_num_frequencies": 5,
        "jepa_loss_ramp_steps": 0,
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
    with pytest.raises(ValueError, match="action_latent_dim"):
        _make_config(jepa_action_latent_dim=0)
    with pytest.raises(ValueError, match="training-only"):
        _make_config(use_jepa_candidate_selection=True)


def test_joint_loss_connects_diffusion_latent_action_to_independent_jepa_branch() -> None:
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
            "latent_action_norm",
        }
    )
    assert model.vision_encoder is not model.jepa_encoder
    assert any(parameter.grad is not None for parameter in model.jepa_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.world_model_predictor.parameters())
    assert model.latent_action_head[-1].weight.grad is not None
    assert model.latent_action_head[-1].weight.grad.abs().sum() > 0
    assert not any("target_encoder" in name for name, _ in model.named_modules())


def test_jepa_encoder_never_changes_diffusion_condition() -> None:
    model = DiffusionJEPAModel(_make_config())
    model.eval()
    batch = _make_batch()
    policy_features = model.vision_encoder(batch)
    global_condition = model._prepare_global_conditioning(policy_features)
    expected_condition = policy_features[:, : model.config.n_obs_steps].flatten(start_dim=1)
    torch.testing.assert_close(global_condition, expected_condition)

    with torch.no_grad():
        for parameter in model.jepa_encoder.parameters():
            parameter.normal_()
        for parameter in model.world_model_predictor.parameters():
            parameter.normal_()

    torch.testing.assert_close(model._prepare_global_conditioning(policy_features), global_condition)


def test_diffusion_only_path_matches_baseline_with_same_unet() -> None:
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


def test_action_generation_does_not_evaluate_jepa_modules() -> None:
    model = DiffusionJEPAModel(_make_config())
    model.eval()
    batch = {
        OBS_STATE: torch.randn(2, model.config.n_obs_steps, 3),
        OBS_ENV_STATE: torch.randn(2, model.config.n_obs_steps, 6),
    }
    noise = torch.randn(2, model.config.horizon, 2)
    called = []

    hooks = [
        module.register_forward_hook(lambda _module, _inputs, _output, name=name: called.append(name))
        for name, module in {
            "jepa_encoder": model.jepa_encoder,
            "latent_action_head": model.latent_action_head,
            "world_model_predictor": model.world_model_predictor,
            "sigreg": model.sigreg,
        }.items()
    ]
    try:
        model.generate_actions(batch, noise=noise)
    finally:
        for hook in hooks:
            hook.remove()

    assert called == []


def test_world_model_uses_diffusion_latent_actions_starting_at_current_timestep() -> None:
    model = DiffusionJEPAModel(_make_config())
    batch = _make_batch()
    head_output = None
    predictor_input = None

    def capture_head(_module, _inputs, output):
        nonlocal head_output
        head_output = output.detach().clone()

    def capture_predictor(_module, inputs, _output):
        nonlocal predictor_input
        predictor_input = inputs[1].detach().clone()

    head_hook = model.latent_action_head.register_forward_hook(capture_head)
    predictor_hook = model.world_model_predictor.register_forward_hook(capture_predictor)
    model.compute_loss(batch)
    head_hook.remove()
    predictor_hook.remove()

    assert head_output is not None
    assert predictor_input is not None
    torch.testing.assert_close(predictor_input, head_output[:, 1:3])
    assert predictor_input.shape[-1] == model.config.jepa_action_latent_dim


def test_jepa_encoder_does_not_leak_future_through_batch_norm() -> None:
    model = DiffusionJEPAModel(_make_config())
    model.train()
    batch = _make_batch()
    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch[OBS_STATE][:, 2:] += 100
    changed_batch[OBS_ENV_STATE][:, 2:] -= 100

    torch.manual_seed(0)
    original = model.jepa_encoder(batch)
    torch.manual_seed(0)
    changed = model.jepa_encoder(changed_batch)
    torch.testing.assert_close(original[:, :2], changed[:, :2])


def test_predictor_is_causal_and_latent_action_adaln_learns() -> None:
    model = DiffusionJEPAModel(_make_config())
    predictor = model.world_model_predictor
    predictor.eval()
    latents = torch.randn(2, 2, 12)
    latent_actions = torch.randn(2, 2, 8, requires_grad=True)

    original = predictor(latents, latent_actions)
    changed_latents = latents.clone()
    changed_latents[:, 1] += 100
    changed = predictor(changed_latents, latent_actions)
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


def test_image_observations_use_separate_policy_and_jepa_encoders() -> None:
    image_key = "observation.images.front"
    config = _make_config(
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (3,)),
            image_key: PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        },
        crop_shape=None,
    )
    policy = DiffusionJEPAPolicy(config)
    assert policy.diffusion_jepa.vision_encoder.rgb_encoder is not policy.diffusion_jepa.jepa_encoder.rgb_encoder
    policy_parameter_ids = {id(parameter) for parameter in policy.diffusion_jepa.vision_encoder.parameters()}
    jepa_parameter_ids = {id(parameter) for parameter in policy.diffusion_jepa.jepa_encoder.parameters()}
    assert policy_parameter_ids.isdisjoint(jepa_parameter_ids)
    batch = _make_batch(batch_size=2)
    batch.pop(OBS_ENV_STATE)
    batch.pop(f"{OBS_ENV_STATE}_is_pad")
    batch[image_key] = torch.rand(2, 4, 3, 32, 32)
    batch[f"{image_key}_is_pad"] = torch.zeros(2, 4, dtype=torch.bool)

    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert "lewm_prediction_loss" in metrics
