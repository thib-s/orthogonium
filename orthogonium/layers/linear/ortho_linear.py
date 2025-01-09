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
        """
        Initializes an orthogonal linear layer with customizable orthogonalization parameters.

        Attributes:
            in_features : int
                Number of input features.
            out_features : int
                Number of output features.
            bias : bool
                Whether to include a bias term in the layer. Default is True.
            ortho_params : OrthoParams
                Parameters for orthogonalization and spectral normalization. Default is the
                default instance of OrthoParams.

        Parameters:
            in_features : int
                The size of each input sample.
            out_features : int
                The size of each output sample.
            bias : bool
                Indicates if the layer should include a learnable bias parameter.
            ortho_params : OrthoParams
                An object containing orthogonalization and normalization configurations.

        Notes
        -----
        The layer is initialized with orthogonal weights using `torch.nn.init.orthogonal_`.
        Weight parameters are further parametrized for both spectral normalization and
        orthogonal constraints using the provided `OrthoParams` object.
        """
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
        """
        A custom PyTorch Linear layer that ensures weights are normalized to unit norm along a specified dimension.

        This class extends the torch.nn.Linear module and modifies the weight
        matrix to maintain orthogonal initialization and unit norm
        normalization during training. In this specific case, each output can be viewed as the result of a 1-Lipschitz
        function. This means that the whole function in more than 1-Lipschitz but that each output taken independently
        is 1-Lipschitz.

        Attributes:
            weight: The learnable weight tensor with orthogonal initialization
                and enforced unit norm parametrization.

        Args:
            *args: Variable length positional arguments passed to the base
                Linear class.
            **kwargs: Variable length keyword arguments passed to the base
                Linear class.
        """
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
