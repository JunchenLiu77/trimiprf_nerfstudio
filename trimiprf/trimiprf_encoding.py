import torch
import nvdiffrast.torch

from jaxtyping import Shaped
from torch import Tensor, nn

from nerfstudio.field_components import Encoding


class TriMipEncoding(Encoding):
    def __init__(
        self,
        n_levels: int,
        resolution: int,
        feat_dim: int,
        include_xyz: bool = False,
        gpu_limitation: int = 4000000,
    ):
        super().__init__(in_dim=3)
        self.n_levels = n_levels
        self.resolution = resolution
        self.feat_dim = feat_dim
        self.include_xyz = include_xyz

        self.register_parameter(
            "fm",
            nn.Parameter(torch.zeros(3, resolution, resolution, feat_dim)),
        )
        self.init_parameters()
        self.out_dim = self.feat_dim * 3 + (3 if include_xyz else 0)
        self.gpu_limitation = gpu_limitation

    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def forward(
        self, in_tensor: Shaped[Tensor, "*bs input_dim"]
    ) -> Shaped[Tensor, "*bs output_dim"]:
        x, level = in_tensor
        # x in [0, 1], level in [0, max_level]
        # x is Nx3, level is Nx1
        if x.shape[0] == 0:
            return torch.zeros([x.shape[0], self.feat_dim * 3]).to(x)
        # RuntimeError: Cuda error: 9[cudaLaunchKernel(func_tbl[func_idx], gridSize, blockSize, args, 0, stream);]
        # indicates that the input is too large, split the input into smaller chunks, ugly but works
        feats = []
        for i in range(0, x.shape[0], self.gpu_limitation):
            end = min(i + self.gpu_limitation, x.shape[0])
            decomposed_x = torch.stack(
                [
                    x[i:end, None, [1, 2]],
                    x[i:end, None, [0, 2]],
                    x[i:end, None, [0, 1]],
                ],
                dim=0,
            )
            level_expanded = torch.broadcast_to(
                level[i:end, :], decomposed_x.shape[:3]
            ).contiguous()
            feats.append(
                nvdiffrast.torch.texture(
                    self.fm,
                    decomposed_x,
                    mip_level_bias=level_expanded,
                    boundary_mode="clamp",
                    max_mip_level=self.n_levels - 1,
                )
            )
        feat = torch.cat(feats, dim=1)
        enc = (
            feat.permute(1, 2, 0, 3).contiguous().view(x.shape[0], self.feat_dim * 3)
        )  # Nx(3C)
        if self.include_xyz:
            enc = torch.cat([x, enc], dim=-1)
        return enc

    def get_out_dim(self) -> int:
        return self.out_dim
