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
from typing import Any

import torch

from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.policies.diffusion_jepa.configuration_diffusion_jepa import DiffusionJEPAConfig
from lerobot.processor import ProcessorStep, ProcessorStepRegistry, TransitionKey

JEPA_IMAGES = "_diffusion_jepa_raw_images"
JEPA_RAW_IMAGE_PREFIX = "_diffusion_jepa_raw_image."


def raw_jepa_image_key(image_key: str) -> str:
    return f"{JEPA_RAW_IMAGE_PREFIX}{image_key}"


@dataclass
@ProcessorStepRegistry.register(name="preserve_diffusion_jepa_raw_images")
class PreserveRawImagesProcessorStep(ProcessorStep):
    """Copy raw RGB observations before Diffusion's dataset normalization."""

    image_keys: list[str]

    def __call__(self, transition):
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition

        new_transition = transition.copy()
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        for key in self.image_keys:
            image = observation.get(key)
            if isinstance(image, torch.Tensor):
                complementary_data[raw_jepa_image_key(key)] = image.clone()
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": self.image_keys}


def make_diffusion_jepa_pre_post_processors(
    config: DiffusionJEPAConfig,
    dataset_stats=None,
):
    """Build Diffusion processors while preserving raw video for frozen V-JEPA2."""
    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        config,
        dataset_stats=dataset_stats,
    )
    # Diffusion's normalizer is the final preprocessor step. Preserve RGB after
    # batching/device placement but immediately before that normalization.
    preprocessor.steps.insert(
        -1,
        PreserveRawImagesProcessorStep(image_keys=list(config.image_features)),
    )
    return preprocessor, postprocessor
