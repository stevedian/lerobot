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

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.diffusion_jepa.configuration_diffusion_jepa import DiffusionJEPAConfig


class FrozenVJEPA2Encoder(nn.Module):
    """Frozen V-JEPA2 world-state encoder used only by the training objective.

    The encoder preserves spatial patch tokens and concatenates synchronized
    camera views along the embedding dimension, following VLA-JEPA Eq. (1).
    """

    def __init__(
        self,
        config: DiffusionJEPAConfig,
        *,
        model: nn.Module | None = None,
        processor=None,
    ):
        super().__init__()
        if model is None:
            if not config.jepa_encoder_name_or_path:
                raise ValueError(
                    "`jepa_encoder_name_or_path` is required when the JEPA world-model loss is enabled."
                )
            try:
                from transformers import AutoModel, AutoVideoProcessor
            except ImportError as exc:
                raise ImportError(
                    "Diffusion-JEPA with V-JEPA2 requires the `transformers-dep` extra."
                ) from exc
            model = AutoModel.from_pretrained(config.jepa_encoder_name_or_path)
            processor = AutoVideoProcessor.from_pretrained(config.jepa_encoder_name_or_path)

        required_config_fields = ("hidden_size", "image_size", "patch_size", "tubelet_size")
        missing = [name for name in required_config_fields if not hasattr(model.config, name)]
        if missing:
            raise ValueError(f"V-JEPA2 model config is missing required fields: {missing}.")

        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()

        self.hidden_size = int(model.config.hidden_size)
        self.image_size = int(model.config.image_size)
        self.patch_size = int(model.config.patch_size)
        self.tubelet_size = int(model.config.tubelet_size)
        if config.jepa_num_frames % self.tubelet_size:
            raise ValueError(
                "`jepa_num_frames` must be divisible by the V-JEPA2 tubelet size. "
                f"Got {config.jepa_num_frames} and {self.tubelet_size}."
            )
        self.num_state_steps = config.jepa_num_frames // self.tubelet_size
        self.tokens_per_state = (self.image_size // self.patch_size) ** 2

        image_mean = getattr(processor, "image_mean", (0.485, 0.456, 0.406))
        image_std = getattr(processor, "image_std", (0.229, 0.224, 0.225))
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    @property
    def output_dim_per_view(self) -> int:
        return self.hidden_size

    def train(self, mode: bool = True):
        """Keep the pretrained target encoder in evaluation mode."""
        super().train(False)
        self.model.eval()
        return self

    def _resize_and_center_crop(self, videos: Tensor) -> Tensor:
        """Apply the V-JEPA2 resize/center-crop recipe on the current device."""
        batch_size, num_frames, channels, height, width = videos.shape
        if channels != 3:
            raise ValueError(f"V-JEPA2 expects RGB videos. Got {channels} channels.")

        resize_short_edge = int(self.image_size * 256 / 224)
        scale = resize_short_edge / min(height, width)
        resized_height = max(self.image_size, round(height * scale))
        resized_width = max(self.image_size, round(width * scale))
        videos = F.interpolate(
            videos.flatten(0, 1),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).unflatten(0, (batch_size, num_frames))

        top = (resized_height - self.image_size) // 2
        left = (resized_width - self.image_size) // 2
        videos = videos[..., top : top + self.image_size, left : left + self.image_size]
        mean = self.image_mean.to(device=videos.device, dtype=videos.dtype)
        std = self.image_std.to(device=videos.device, dtype=videos.dtype)
        return (videos - mean) / std

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Encode ``(B, T, V, C, H, W)`` videos as ``(B, T', P, V*D)`` tokens."""
        if images.ndim != 6:
            raise ValueError(
                "Frozen V-JEPA2 expects images shaped (B, T, V, C, H, W). "
                f"Got {tuple(images.shape)}."
            )
        batch_size, num_frames, num_views, channels, _, _ = images.shape
        if num_frames != self.num_state_steps * self.tubelet_size:
            raise ValueError(
                f"V-JEPA2 expects {self.num_state_steps * self.tubelet_size} frames, "
                f"got {num_frames}."
            )

        videos = images.permute(0, 2, 1, 3, 4, 5).flatten(0, 1)
        if not videos.is_floating_point():
            videos = videos.float().div_(255)
        else:
            videos = videos.float()
            min_value = videos.amin().item()
            max_value = videos.amax().item()
            if min_value < -1e-4:
                raise ValueError(
                    "V-JEPA2 must receive raw RGB in [0, 1], not Diffusion-normalized images. "
                    f"Observed minimum {min_value:.4f}."
                )
            if max_value > 1.0001:
                if max_value <= 255:
                    videos = videos.div(255)
                else:
                    raise ValueError(
                        "V-JEPA2 RGB values must be in [0, 1] or [0, 255]. "
                        f"Observed maximum {max_value:.4f}."
                    )
        videos = self._resize_and_center_crop(videos)

        embeddings = self.model.get_vision_features(pixel_values_videos=videos)
        expected_tokens = self.num_state_steps * self.tokens_per_state
        if embeddings.shape[1] != expected_tokens:
            raise ValueError(
                "Unexpected V-JEPA2 token count. "
                f"Expected {expected_tokens}, got {embeddings.shape[1]}."
            )
        embeddings = embeddings.view(
            batch_size,
            num_views,
            self.num_state_steps,
            self.tokens_per_state,
            self.hidden_size,
        )
        return embeddings.permute(0, 2, 3, 1, 4).flatten(-2)
