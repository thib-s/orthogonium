import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils import parametrize as parametrize

from orthogonium.reparametrizers import L2Normalize
from orthogonium.reparametrizers import OrthoParams


class OrthoLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        ortho_params: OrthoParams = OrthoParams(),
    ):
        super(OrthoLinear, self).__init__(in_features, out_features, bias=bias)
        torch.nn.init.orthogonal_(self.weight)
        parametrize.register_parametrization(
            self,
            "weight",
            ortho_params.spectral_normalizer(
                weight_shape=(self.out_features, self.in_features)
            ),
        )
        parametrize.register_parametrization(
            self, "weight", ortho_params.orthogonalizer(weight_shape=self.weight.shape)
        )

    def singular_values(self):
        svs = np.linalg.svd(
            self.weight.detach().cpu().numpy(), full_matrices=False, compute_uv=False
        )
        stable_rank = np.sum((np.mean(svs) ** 2)) / (svs.max() ** 2)
        return svs.min(), svs.max(), stable_rank


class UnitNormLinear(nn.Linear):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """LInear layer where each output unit is normalized to have Frobenius norm 1"""
        super(UnitNormLinear, self).__init__(*args, **kwargs)
        torch.nn.init.orthogonal_(self.weight)
        parametrize.register_parametrization(
            self,
            "weight",
            L2Normalize(dtype=self.weight.dtype, dim=1),
        )

    def singular_values(self):
        svs = np.linalg.svd(
            self.weight.detach().cpu().numpy(), full_matrices=False, compute_uv=False
        )
        stable_rank = np.sum(np.mean(svs) ** 2) / (svs.max() ** 2)
        return svs.min(), svs.max(), stable_rank
