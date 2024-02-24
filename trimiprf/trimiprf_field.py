# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Fields for Tri-MipRF (https://github.com/wbhu/Tri-MipRF/)
"""

from typing import Dict, Optional, Tuple, Literal

import torch
from torch import Tensor, nn
import math
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

from .trimiprf_encoding import TriMipEncoding


class TriMipRFField(Field):
    """TriMipRF field.

    Args:
        aabb: parameters of scene aabb bounds
        n_levels: number of mipmap levels
        feat_grid_resolution: resolution of the feature grid
        occ_grid_resolution: resolution of the occupancy grid
        feat_dim: dimension of the feature grid
        geo_feat_dim: dimension of the geometric feature
        net_depth_base: number of hidden layers for the base mlp
        net_depth_color: number of hidden layers for the color mlp
        net_width: width of the hidden layers
        implementation: implementation method
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb: SceneBox,
        n_levels: int = 8,
        feat_grid_resolution: int = 512,
        occ_grid_resolution: int = 128,
        feat_dim: int = 16,
        geo_feat_dim: int = 15,
        net_depth_base: int = 2,
        net_depth_color: int = 4,
        net_width: int = 128,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        average_init_density: float = 1.0,
        gpu_limitation: int = 4000000,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        assert (
            self.aabb[0, :].max() == self.aabb[0, :].min()
            and self.aabb[1, :].max() == self.aabb[1, :].min()
        )
        self.max_radius = (self.aabb[1, :] - self.aabb[0, :]).max() / 2
        self.geo_feat_dim = geo_feat_dim
        self.log2_feat_grid_resolution = math.log2(feat_grid_resolution)
        self.log2_occ_grid_resolution = math.log2(occ_grid_resolution)
        self.average_init_density = average_init_density
        self.spatial_distortion = spatial_distortion

        self.encoding = TriMipEncoding(n_levels, feat_grid_resolution, feat_dim, gpu_limitation=gpu_limitation)
        self.mlp_base = MLP(
            in_dim=self.encoding.get_out_dim(),
            out_dim=geo_feat_dim + 1,
            layer_width=net_width,
            num_layers=net_depth_base,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )
        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + geo_feat_dim,
            out_dim=3,
            layer_width=net_width,
            num_layers=net_depth_color,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    @staticmethod
    def compute_ball_radius(distances, radius, cos):
        inverse_cos = 1.0 / cos
        tmp = (inverse_cos * inverse_cos - 1).sqrt() - radius
        sample_ball_radius = distances * radius * cos / (tmp * tmp + 1.0).sqrt()
        return sample_ball_radius

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        frustums = ray_samples.frustums
        positions = frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        if torch.all(frustums.directions == torch.ones_like(positions)) or torch.all(
            ray_samples.frustums.starts == ray_samples.frustums.ends
        ):
            # used in update_every_n_steps, ugly but works
            levels = torch.empty_like(positions[..., 0:1]).fill_(
                -self.log2_occ_grid_resolution
            )
        else:
            distances = (frustums.starts + frustums.ends) / 2
            cone_radius = torch.sqrt(frustums.pixel_area) / 1.7724538509055159
            cos = torch.matmul(
                frustums.directions,
                torch.tensor([[0.0, 0.0, -1.0]], device=frustums.directions.device).T,
            )
            sample_ball_radius = self.compute_ball_radius(distances, cone_radius, cos)
            levels = torch.log2(sample_ball_radius / self.max_radius)
        levels += self.log2_feat_grid_resolution
        feat = self.encoding((positions, levels))
        res = self.mlp_base(feat)
        if ray_samples.frustums.shape[0] == 0:
            h = res
        else:
            h = res.view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(
            h, [1, self.geo_feat_dim], dim=-1
        )
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.average_init_density * trunc_exp(
            density_before_activation.to(positions) - 1.0
        )
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        directions = get_normalized_directions(ray_samples.frustums.directions)
        d = self.direction_encoding(directions)
        outputs_shape = ray_samples.frustums.directions.shape[:-1]
        h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
