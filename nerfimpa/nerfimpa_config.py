"""
Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfimpa.models.custom_vanilla_model import CustomVanillaModel
from nerfimpa.models.custom_vanilla_model import CustomVanillaModelConfig
from nerfimpa.fields.custom_vanilla_field import CustomVanillaField

from nerfimpa.models.siren_model import SirenModel, SirenModelConfig
from nerfimpa.models.tuner_model import TunerModel, TunerModelConfig

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
        experiment_name="custom-vanilla-nerf-baseline",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=40000,
        mixed_precision=True,
        steps_per_eval_all_images=1200000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=CustomVanillaModelConfig(_target=CustomVanillaModel, eval_num_rays_per_chunk=1024),
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
        experiment_name="siren-nerf-baseline",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=40000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(_target=SirenModel, eval_num_rays_per_chunk=1024),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
        vis="viewer",
    ),
    description="Custom vanilla nerf",
)

siren_nerf_nope = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-nope",
        experiment_name="siren-nerf-nope",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                use_positional_encoding=False,
                w0=30,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_w5 = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-w5",
        experiment_name="siren-nerf-w5",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                w0=5,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_w20 = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-w20",
        experiment_name="siren-nerf-w20",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                w0=20,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_sirencolor = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-sirencolor",
        experiment_name="siren-nerf-sirencolor",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                use_siren_color_head=True,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_lr1em4 = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-lr1em4",
        experiment_name="siren-nerf-lr1em4",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_lw256 = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-lw256",
        experiment_name="siren-nerf-lw256",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                base_mlp_layer_width=256,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_nl5 = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-nl5",
        experiment_name="siren-nerf-nl5",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                base_mlp_num_layers=5,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_nl6 = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-nl6",
        experiment_name="siren-nerf-nl6",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                base_mlp_num_layers=6,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_noclamp = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-noclamp",
        experiment_name="siren-nerf-noclamp",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                use_positional_encoding=False,
                w0=30
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": None,
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_baseline2 = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-baseline2",
        experiment_name="siren-nerf-baseline2",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=40000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                w0=10
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-6, max_steps=40000),
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_no_w0 = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-no-w0",
        experiment_name="siren-nerf-no-w0",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                w0=None,
                w0_hidden=1.0,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-6, max_steps=40000),
            },
        },
    ),
    description="Custom vanilla nerf",
)

siren_nerf_no_w0_hidden5 = MethodSpecification(
    config=TrainerConfig(
        method_name="siren-nerf-no-w0-hidden5",
        experiment_name="siren-nerf-no-w0-hidden5",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=SirenModelConfig(
                _target=SirenModel,
                eval_num_rays_per_chunk=1024,
                w0=None,
                w0_hidden=5.0,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-6, max_steps=40000),
            },
        },
    ),
    description="Custom vanilla nerf",
)

tuner_encoding = MethodSpecification(
    config=TrainerConfig(
        method_name="tuner-encoding",
        experiment_name="tuner-encoding",
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=20000,
        mixed_precision=True,
        steps_per_eval_all_images=130000000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=64,
            ),
            model=TunerModelConfig(
                _target=TunerModel,
                eval_num_rays_per_chunk=1024,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-6, max_steps=40000),
            },
        },
    ),
    description="Custom vanilla nerf",
)
