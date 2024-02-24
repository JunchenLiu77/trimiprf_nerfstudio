from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from .trimiprf import TriMipRFModelConfig

trimiprf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="trimiprf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=DynamicBatchPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(), train_num_rays_per_batch=8192
            ),
            model=TriMipRFModelConfig(
                eval_num_rays_per_chunk=8192,
                alpha_thre=0.0,
                cone_angle=0.0,
                disable_scene_contraction=True,
                near_plane=2.0,
                far_plane=6.0,
                background_color="white",
                use_gradient_scaling=True,
                gpu_limitation=4000000,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001, max_steps=200000
                ),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="TriMipRF model",
)
