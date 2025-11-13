from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch
from torch.nn import Parameter

from nerfimpa.models.custom_vanilla_model import CustomVanillaModel
from nerfimpa.models.custom_vanilla_model import CustomVanillaModelConfig

from nerfimpa.fields.siren_field import SirenField

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.encodings import NeRFEncoding, Identity, SHEncoding
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
    w0: Optional[float] = 10.0
    w0_hidden: float = 1.0
    use_positional_encoding: bool = True
    use_directional_encoding: bool = True
    base_mlp_layer_width: int = 512
    base_mlp_num_layers: int = 4
    use_siren_color_head: bool = False


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

        position_encoding = (
            NeRFEncoding(in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8, include_input=True)
            if self.config.use_positional_encoding
            else Identity(in_dim=3)
        )

        direction_encoding = (
            NeRFEncoding(in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True)
            if self.config.use_directional_encoding
            else Identity(in_dim=3)
        )

        self.field_coarse = SirenField(
            self.scene_box.aabb,
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            skip_connections=(),
            base_mlp_num_layers=self.config.base_mlp_num_layers,
            base_mlp_layer_width=self.config.base_mlp_layer_width,
            head_mlp_layer_width=128,
            w0=self.config.w0,
            w0_hidden=self.config.w0_hidden,
            use_siren_color_head=self.config.use_siren_color_head
        )

        self.field_fine = SirenField(
            self.scene_box.aabb,
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            skip_connections=(),
            base_mlp_num_layers=self.config.base_mlp_num_layers,
            base_mlp_layer_width=self.config.base_mlp_layer_width,
            head_mlp_layer_width=128,
            w0=self.config.w0,
            w0_hidden=self.config.w0_hidden,
            use_siren_color_head=self.config.use_siren_color_head
        )
