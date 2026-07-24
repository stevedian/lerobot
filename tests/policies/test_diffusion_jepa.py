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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion_jepa.configuration_diffusion_jepa import DiffusionJEPAConfig
from lerobot.policies.diffusion_jepa.modeling_diffusion_jepa import (
    DiffusionJEPAModel,
    DiffusionJEPAPolicy,
)
from lerobot.policies.diffusion_jepa.modeling_lewm_predictor import (
    LeWorldModelPredictor,
    build_action_block_causal_attention_mask,
)
from lerobot.policies.diffusion_jepa.modeling_vjepa2 import FrozenVJEPA2Encoder
from lerobot.policies.diffusion_jepa.processor_diffusion_jepa import (
    JEPA_IMAGES,
    make_diffusion_jepa_pre_post_processors,
    raw_jepa_image_key,
)
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

IMAGE_KEY = "observation.images.front"


class FakeVJEPA2(nn.Module):
    """Small deterministic stand-in for shape and gradient-boundary tests."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=6,
            image_size=8,
            patch_size=4,
            tubelet_size=2,
        )
        self.scale = nn.Parameter(torch.ones(()))

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames = pixel_values_videos.shape[:2]
        num_steps = num_frames // self.config.tubelet_size
        pooled = pixel_values_videos.mean(dim=(2, 3, 4))
        pooled = pooled.view(batch_size, num_steps, self.config.tubelet_size).mean(dim=-1)
        channels = torch.arange(
            self.config.hidden_size,
            dtype=pooled.dtype,
            device=pooled.device,
        )
        tokens = pooled[:, :, None, None] + channels[None, None, None]
        tokens = tokens.expand(batch_size, num_steps, 4, self.config.hidden_size)
        return (tokens * self.scale).flatten(1, 2)


def _make_config(**overrides) -> DiffusionJEPAConfig:
    kwargs = {
        "input_features": {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (3,)),
            IMAGE_KEY: PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
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
        "crop_shape": None,
        "jepa_num_frames": 4,
        "jepa_latent_dim": 12,
        "jepa_predictor_layers": 2,
        "jepa_predictor_heads": 3,
        "jepa_predictor_mlp_ratio": 2.0,
        "jepa_predictor_dropout": 0.0,
        "jepa_prediction_horizon": 1,
        "jepa_action_latent_dim": 8,
        "jepa_action_tokens_per_transition": 2,
        "jepa_world_model_loss_weight": 0.1,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return DiffusionJEPAConfig(**kwargs)


def _make_frozen_encoder(config: DiffusionJEPAConfig) -> FrozenVJEPA2Encoder:
    return FrozenVJEPA2Encoder(config, model=FakeVJEPA2())


def _make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    # Observation offsets are [-1, 0, 1, 2, 3]. V-JEPA2 receives [0, 1, 2, 3].
    sequence_length = 5
    return {
        OBS_STATE: torch.randn(batch_size, sequence_length, 3),
        IMAGE_KEY: torch.rand(batch_size, sequence_length, 3, 32, 32),
        f"{OBS_STATE}_is_pad": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        f"{IMAGE_KEY}_is_pad": torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        ACTION: torch.randn(batch_size, 4, 2).clamp(-1, 1),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


def _stack_images(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(batch)
    result[OBS_IMAGES] = batch[IMAGE_KEY].unsqueeze(2)
    raw_key = raw_jepa_image_key(IMAGE_KEY)
    if raw_key in batch:
        result[JEPA_IMAGES] = batch[raw_key].unsqueeze(2)
    return result


def test_config_requests_policy_history_and_complete_video_clip() -> None:
    config = _make_config()
    assert config.observation_delta_indices == [-1, 0, 1, 2, 3]
    assert config.action_delta_indices == [-1, 0, 1, 2]

    with pytest.raises(ValueError, match="at least 2"):
        _make_config(jepa_num_frames=1)
    with pytest.raises(ValueError, match="divisible"):
        _make_config(jepa_latent_dim=10)
    with pytest.raises(ValueError, match="action_tokens_per_transition"):
        _make_config(jepa_action_tokens_per_transition=0)


def test_frozen_vjepa2_preserves_patch_tokens_and_concatenates_views() -> None:
    config = _make_config()
    encoder = _make_frozen_encoder(config)
    encoder.train()
    videos = torch.rand(2, 4, 2, 3, 12, 10)
    videos[:, :, 1] *= 0.5

    output = encoder(videos)

    assert output.shape == (2, 2, 4, 12)
    assert not encoder.training
    assert not encoder.model.training
    assert not output.requires_grad
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert not torch.equal(output[..., :6], output[..., 6:])


def test_vjepa2_requires_a_checkpoint_when_not_injected() -> None:
    with pytest.raises(ValueError, match="jepa_encoder_name_or_path"):
        FrozenVJEPA2Encoder(_make_config())


def test_preprocessor_preserves_raw_rgb_before_diffusion_normalization() -> None:
    config = _make_config(jepa_world_model_loss_weight=0)
    stats = {
        OBS_STATE: {
            "min": torch.full((3,), -1.0),
            "max": torch.full((3,), 1.0),
        },
        IMAGE_KEY: {
            "mean": torch.full((3, 1, 1), 0.5),
            "std": torch.full((3, 1, 1), 0.25),
        },
        ACTION: {
            "min": torch.full((2,), -1.0),
            "max": torch.full((2,), 1.0),
        },
    }
    preprocessor, _ = make_diffusion_jepa_pre_post_processors(config, dataset_stats=stats)
    image = torch.rand(3, 32, 32)
    processed = preprocessor(
        {
            OBS_STATE: torch.zeros(3),
            IMAGE_KEY: image,
            ACTION: torch.zeros(2),
        }
    )

    torch.testing.assert_close(processed[raw_jepa_image_key(IMAGE_KEY)].squeeze(0), image)
    assert not torch.equal(processed[IMAGE_KEY].squeeze(0), image)


def test_vjepa2_rejects_diffusion_normalized_rgb() -> None:
    encoder = _make_frozen_encoder(_make_config())
    with pytest.raises(ValueError, match="raw RGB"):
        encoder(-torch.ones(1, 4, 1, 3, 12, 10))


def test_block_causal_mask_allows_full_current_block_and_blocks_future() -> None:
    mask = build_action_block_causal_attention_mask(
        num_steps=3,
        tokens_per_state=4,
        num_action_tokens=2,
    )
    assert mask.shape == (18, 18)
    assert not mask[:6, :6].any()
    assert mask[:6, 6:].all()
    assert not mask[6:12, :12].any()
    assert mask[6:12, 12:].all()


def test_token_world_model_is_causal() -> None:
    config = _make_config(jepa_prediction_horizon=3)
    predictor = LeWorldModelPredictor(
        config,
        state_dim=6,
        num_steps=3,
        tokens_per_state=4,
    ).eval()
    states = torch.randn(2, 3, 4, 6)
    latent_actions = torch.randn(2, 3, 2, 8)

    original = predictor(states, latent_actions)
    changed_states = states.clone()
    changed_states[:, 2] += 100
    changed_actions = latent_actions.clone()
    changed_actions[:, 2] -= 100
    changed = predictor(changed_states, changed_actions)

    torch.testing.assert_close(original[:, :2], changed[:, :2])
    assert original.shape == states.shape


def test_joint_loss_connects_policy_latent_actions_to_token_world_model() -> None:
    config = _make_config()
    model = DiffusionJEPAModel(config, jepa_encoder=_make_frozen_encoder(config))
    model.train()
    loss, metrics = model.compute_loss(_stack_images(_make_batch()))
    loss.backward()

    assert set(metrics).issuperset(
        {
            "diffusion_loss",
            "world_model_loss",
            "weighted_world_model_loss",
            "world_state_std",
            "latent_action_norm",
        }
    )
    assert model.jepa_encoder is not None
    assert all(parameter.grad is None for parameter in model.jepa_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.vision_encoder.parameters())
    assert model.latent_action_queries.query_tokens.grad is not None
    assert model.latent_action_queries.query_tokens.grad.abs().sum() > 0
    assert any(parameter.grad is not None for parameter in model.world_model_predictor.parameters())
    assert any(parameter.grad is not None for parameter in model.unet.parameters())


def test_world_loss_uses_l1_on_frozen_shifted_targets() -> None:
    config = _make_config()
    model = DiffusionJEPAModel(config, jepa_encoder=_make_frozen_encoder(config))
    batch = _stack_images(_make_batch())
    captured_states = None
    captured_predictions = None

    def capture_encoder(_module, _inputs, output):
        nonlocal captured_states
        captured_states = output.detach()

    def capture_predictor(_module, _inputs, output):
        nonlocal captured_predictions
        captured_predictions = output.detach()

    encoder_hook = model.jepa_encoder.register_forward_hook(capture_encoder)
    predictor_hook = model.world_model_predictor.register_forward_hook(capture_predictor)
    _, metrics = model.compute_loss(batch)
    encoder_hook.remove()
    predictor_hook.remove()

    expected = (captured_predictions - captured_states[:, 1:]).abs().mean()
    assert metrics["world_model_loss"] == pytest.approx(expected.item())


def test_policy_condition_never_reads_future_observations() -> None:
    config = _make_config()
    model = DiffusionJEPAModel(config, jepa_encoder=_make_frozen_encoder(config)).eval()
    batch = _stack_images(_make_batch())
    changed = {key: value.clone() for key, value in batch.items()}
    changed[OBS_STATE][:, config.n_obs_steps :] += 100
    changed[OBS_IMAGES][:, config.n_obs_steps :] = 0

    original_condition, original_actions = model._encode_policy_condition(batch)
    changed_condition, changed_actions = model._encode_policy_condition(changed)

    torch.testing.assert_close(original_condition, changed_condition)
    torch.testing.assert_close(original_actions, changed_actions)


def test_inference_runs_policy_latent_actions_but_not_jepa_modules() -> None:
    config = _make_config()
    model = DiffusionJEPAModel(config, jepa_encoder=_make_frozen_encoder(config)).eval()
    batch = _stack_images(_make_batch())
    inference_batch = {
        OBS_STATE: batch[OBS_STATE][:, : config.n_obs_steps],
        OBS_IMAGES: batch[OBS_IMAGES][:, : config.n_obs_steps],
    }
    called = []
    hooks = [
        module.register_forward_hook(lambda _module, _inputs, _output, name=name: called.append(name))
        for name, module in {
            "latent_action_queries": model.latent_action_queries,
            "jepa_encoder": model.jepa_encoder,
            "world_model_predictor": model.world_model_predictor,
        }.items()
    ]
    try:
        actions = model.generate_actions(
            inference_batch,
            noise=torch.randn(2, config.horizon, 2),
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert actions.shape == (2, config.n_action_steps, 2)
    assert called == ["latent_action_queries"]


def test_optimizer_excludes_frozen_vjepa2_parameters() -> None:
    config = _make_config()
    policy = DiffusionJEPAPolicy(config, jepa_encoder=_make_frozen_encoder(config))
    optimizer_parameter_ids = {id(parameter) for parameter in policy.get_optim_params()}
    frozen_parameter_ids = {
        id(parameter) for parameter in policy.diffusion_jepa.jepa_encoder.parameters()
    }

    assert optimizer_parameter_ids.isdisjoint(frozen_parameter_ids)
    assert id(policy.diffusion_jepa.latent_action_queries.query_tokens) in optimizer_parameter_ids


def test_world_model_can_be_disabled_for_diffusion_only_training() -> None:
    config = _make_config(jepa_world_model_loss_weight=0)
    model = DiffusionJEPAModel(config)
    batch = _stack_images(_make_batch())
    loss, metrics = model.compute_loss(batch)

    assert torch.isfinite(loss)
    assert metrics["world_model_weight"] == 0
    assert model.jepa_encoder is None
    assert model.world_model_predictor is None


def test_policy_factory_action_queue_and_checkpoint_round_trip(tmp_path) -> None:
    assert get_policy_class("diffusion_jepa") is DiffusionJEPAPolicy
    assert isinstance(make_policy_config("diffusion_jepa"), DiffusionJEPAConfig)

    config = _make_config()
    policy = DiffusionJEPAPolicy(config, jepa_encoder=_make_frozen_encoder(config)).eval()
    observation = {
        OBS_STATE: torch.randn(2, 3),
        IMAGE_KEY: torch.rand(2, 3, 32, 32),
    }
    action = policy.select_action(
        observation,
        noise=torch.randn(2, config.horizon, 2),
    )
    assert action.shape == (2, 2)

    save_directory = tmp_path / "diffusion_jepa"
    assert not any(
        key.startswith("diffusion_jepa.jepa_encoder.model.")
        for key in policy.state_dict()
    )
    policy.save_pretrained(save_directory)
    loaded = DiffusionJEPAPolicy.from_pretrained(
        save_directory,
        config=policy.config,
        jepa_encoder=_make_frozen_encoder(config),
    )
    torch.testing.assert_close(list(policy.parameters()), list(loaded.parameters()), rtol=0, atol=0)
