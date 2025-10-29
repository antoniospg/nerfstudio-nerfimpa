"""
Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfimpa.models.custom_vanilla_model import CustomVanillaModel
from nerfimpa.models.custom_vanilla_model import CustomVanillaModelConfig
from nerfimpa.fields.custom_vanilla_field import CustomVanillaField

from nerfimpa.models.siren_model import SirenModel
from nerfimpa.models.siren_model import SirenModelConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from nerfstudio.field_components.encodings import NeRFEncoding

custom_vanilla_nerf = MethodSpecification(
    config=TrainerConfig(
        method_name="custom-vanilla-nerf",
        experiment_name="custom-vanilla-nerf",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=120000,
        mixed_precision=True,
        steps_per_eval_all_images=1200000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=512,
                eval_num_rays_per_batch=64,
                cache_images_type="uint8"
            ),
            model=CustomVanillaModelConfig(_target=CustomVanillaModel),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
            "temporal_distortion": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
        vis="viewer",
    ),
    description="Custom vanilla nerf",
)

siren_nerf = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf",
        experiment_name="siren-nerf",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=120000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=512,
                eval_num_rays_per_batch=64,
                cache_images_type="uint8"
            ),
            model=SirenModelConfig(_target=SirenModel),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-4, eps=1e-08),
                "scheduler": None,
            },
        },
        vis="viewer",
    ),
    description="Custom vanilla nerf",
)
