from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Type

import torch
from torch.nn import Parameter

from nerfimpa.models.custom_vanilla_model import CustomVanillaModel
from nerfimpa.models.custom_vanilla_model import CustomVanillaModelConfig

from nerfimpa.fields.siren_field import SirenField

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.encodings import NeRFEncoding, Identity
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc


@dataclass
class SirenModelConfig(CustomVanillaModelConfig):
    """Siren Model Config"""

    _target: Type = field(default_factory=lambda: SirenModel)


class SirenModel(CustomVanillaModel):
    """Siren NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config:SirenModelConfig

    def __init__(
        self,
        config: SirenModelConfig,
        **kwargs,
    ) -> None:
        self.temporal_distortion = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        position_encoding = Identity(in_dim=3)
        direction_encoding = Identity(in_dim=3)

        self.field_coarse = SirenField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_fine = SirenField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

