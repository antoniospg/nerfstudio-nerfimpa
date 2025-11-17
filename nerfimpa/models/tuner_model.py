from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch
from torch.nn import Parameter

from nerfimpa.models.custom_vanilla_model import CustomVanillaModel
from nerfimpa.models.custom_vanilla_model import CustomVanillaModelConfig

from nerfimpa.fields.tuner_field import TunerField
from nerfimpa.field_components.tuner_encoding import TunerEncoding
from nerfimpa.utils.utils import to_gray, fft2_power_spectrum_gray, gray_to_rgb

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
class TunerModelConfig(CustomVanillaModelConfig):
    """Siren Model Config"""

    _target: Type = field(default_factory=lambda: TunerModel)


class TunerModel(CustomVanillaModel):
    """Siren NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config:TunerModelConfig

    def __init__(
        self,
        config: TunerModelConfig,
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

        # Desired bandwidth
        B = 2 ** 7
        b = 2 ** 3

        position_encoding = TunerEncoding(
            in_dim=3,
            hidden_width=128,
            m=256, B=B, b=b, low_frac=0.7,
            learned_bounds=False, c_low=1.0, c_high=0.05, reg_lambda=0.05,
            include_input=True
        )
        direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True)

        self.field_coarse = TunerField(
            self.scene_box.aabb,
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        self.field_fine = TunerField(
            self.scene_box.aabb,
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)
        coarse_pred, coarse_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        fine_pred, fine_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )

        rgb_loss_coarse = self.rgb_loss(coarse_image, coarse_pred)
        rgb_loss_fine = self.rgb_loss(fine_image, fine_pred)

        # Tuner l1 regularization factor
        tuner_reg = torch.zeros((), device=image.device)
        for field in (self.field_coarse, self.field_fine):
            enc = getattr(field, "position_encoding", None)
            if enc is not None and hasattr(enc, "regularizer"):
                tuner_reg = tuner_reg + enc.regularizer()

        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine, "tuner_l1": tuner_reg}

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict
