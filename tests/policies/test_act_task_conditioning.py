import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _make_config(**overrides) -> ACTConfig:
    kwargs = {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        "chunk_size": 2,
        "n_action_steps": 2,
        "dim_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "use_vae": False,
        "task_id_num_classes": 3,
    }
    kwargs.update(overrides)
    return ACTConfig(**kwargs)


def test_task_id_is_appended_as_one_hot_to_robot_state() -> None:
    policy = ACTPolicy(_make_config())
    batch = {
        OBS_STATE: torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "task_index": torch.tensor([0, 2]),
    }

    conditioned = policy.model._task_conditioned_robot_state(batch)

    expected = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0], [4.0, 5.0, 6.0, 0.0, 0.0, 1.0]]
    )
    torch.testing.assert_close(conditioned, expected)
    assert policy.model.encoder_robot_state_input_proj.in_features == 6


def test_task_id_override_ignores_dataset_task_index() -> None:
    policy = ACTPolicy(_make_config(task_id_override=1))
    batch = {
        OBS_STATE: torch.zeros(2, 3),
        "task_index": torch.tensor([0, 2]),
    }

    conditioned = policy.model._task_conditioned_robot_state(batch)

    torch.testing.assert_close(conditioned[:, 3:], torch.tensor([[0.0, 1.0, 0.0]] * 2))


def test_disabled_task_conditioning_preserves_original_state() -> None:
    policy = ACTPolicy(_make_config(task_id_num_classes=0))
    robot_state = torch.randn(2, 3)

    conditioned = policy.model._task_conditioned_robot_state({OBS_STATE: robot_state})

    assert conditioned is robot_state
    assert policy.model.encoder_robot_state_input_proj.in_features == 3


def test_task_conditioned_forward_shape() -> None:
    policy = ACTPolicy(_make_config()).eval()
    batch = {
        OBS_STATE: torch.randn(2, 3),
        OBS_ENV_STATE: torch.randn(2, 2),
        "task_index": torch.tensor([0, 2]),
    }

    actions = policy.predict_action_chunk(batch)

    assert actions.shape == (2, 2, 2)


def test_task_conditioned_vae_training_forward_shape() -> None:
    policy = ACTPolicy(_make_config(use_vae=True, n_vae_encoder_layers=1)).train()
    batch = {
        OBS_STATE: torch.randn(2, 3),
        OBS_ENV_STATE: torch.randn(2, 2),
        ACTION: torch.randn(2, 2, 2),
        "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
        "task_index": torch.tensor([1, 2]),
    }

    actions, (mu, log_sigma_x2) = policy.model(batch)

    assert actions.shape == (2, 2, 2)
    assert mu.shape == (2, policy.config.latent_dim)
    assert log_sigma_x2.shape == (2, policy.config.latent_dim)
    assert policy.model.vae_encoder_robot_state_input_proj.in_features == 6


def test_missing_or_out_of_range_task_id_is_rejected() -> None:
    policy = ACTPolicy(_make_config())

    with pytest.raises(KeyError, match="task_index"):
        policy.model._task_conditioned_robot_state({OBS_STATE: torch.zeros(1, 3)})
    with pytest.raises(ValueError, match="Task IDs must be"):
        policy.model._task_conditioned_robot_state(
            {OBS_STATE: torch.zeros(1, 3), "task_index": torch.tensor([3])}
        )


def test_invalid_task_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="task_id_override"):
        _make_config(task_id_override=3)
