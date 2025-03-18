"""
# SSL derived 1-Lipschitz Layers

This module implements several 1-Lipschitz residual blocks, inspired by and extending
the SDP-based Lipschitz Layers (SLL) from [1]. Specifically:

- **`SDPBasedLipschitzResBlock`**  
  The original version of the 1-Lipschitz convolutional residual block. It enforces Lipschitz
  constraints by rescaling activation outputs according to an estimate of the operator norm.

- **`SLLxAOCLipschitzResBlock`**  
  An extended version of the SLL approach described in [1], combined with additional orthogonal
  convolutions to handle stride, kernel-size, or channel-dimension changes. It fuses multiple
  convolutions via the block convolution, thereby preserving the 1-Lipschitz property while enabling
  strided downsampling or modifying input/output channels.

- **`AOCLipschitzResBlock`**  
  A variant of the original Lipschitz block where the core convolution is replaced by an
  `AdaptiveOrthoConv2d`. It maintains the 1-Lipschitz property with orthogonal weight
  parameterization while providing efficient convolution implementations.

## References

[1] Alexandre Araujo, Aaron J Havens, Blaise Delattre, Alexandre Allauzen, and Bin Hu. A unified alge-
braic perspective on lipschitz neural networks. In The Eleventh International Conference on Learning
Representations, 2023
[2] Thibaut Boissin, Franck Mamalet, Thomas Fel, Agustin Martin Picard, Thomas Massena, Mathieu Serrurier,
An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures

## Notes on the SLL approach

In [1], the SLL layer for convolutions is a 1-Lipschitz residual operation defined approximately as:

$$
y = x - \mathbf{K}^T \\star (t \\times  \sigma(\\mathbf{K} \\star x + b)),
$$

where $\mathbf{K}$ represents a toeplitz (convolution) matrix that represent a 1-Lipschitz operator.
This is done in practice by computing a normalization vector $\mathbf{t}$ and rescaling the
activation outputs by $\mathbf{t}$.

By default, the SLL formulation does **not** allow strides or changes in the number of channels.  
To address these issues, `SLLxAOCLipschitzResBlock` adds extra orthogonal convolutions before and/or
after the main SLL operation. These additional convolutions can be merged via block convolution
(Proposition 1 in [2]) to maintain 1-Lipschitz behavior while enabling stride and/or channel changes.

When $\mathbf{K}$, $\mathbf{K}_{pre}$, and $\mathbf{K}_{post}$ each correspond to 2×2 convolutions,
the resulting block effectively contains two 3×3 convolutions in one branch and a single 4×4 stride-2
convolution in the skip branch—quite similar to typical ResNet blocks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.utils import parametrize

from orthogonium.layers import AdaptiveOrthoConv2d
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import fast_matrix_conv
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import transpose_kernel
from orthogonium.layers.conv.AOL.aol import AOLReparametrizer, safe_inv
from orthogonium.reparametrizers import OrthoParams


class SLLxAOCLipschitzResBlock(nn.Module):
    def __init__(
        self, cin, cout, inner_dim_factor, kernel_size=3, stride=2, groups=1, **kwargs
    ):
        """
        Extended SLL-based convolutional residual block. Supports arbitrary kernel sizes,
        strides, and changes in the number of channels by integrating additional
        orthogonal convolutions *and* fusing them via `\mathbconv` [1].

        The forward pass follows:

        $$
        y = (\mathbf{K}_{post} \circledast \mathbf{K}_{pre}) \\star x - (\mathbf{K}_{post} \circledast \mathbf{K}^T) \\star (t \\times  \sigma(( \mathbf{K} \circledast \mathbf{K}_{pre}) \\star x + b)),
        $$

        where $\mathbf{K}_{pre}$ and $\mathbf{K}_{post}$ are obtained with AOC.


        <img src="../../assets/SLL_3.png" alt="illustration of SLL x AOC" width="600">



        where the kernel `\kernel{K}` may effectively be expanded by pre/post AOC layers to
        handle stride and channel changes. This approach is described in "Improving
        SDP-based Lipschitz Layers" of [1].

        **Args**:
          - `cin` (int): Number of input channels.
          - `inner_dim_factor` (float): Multiplier for the internal channel dimension.
          - `kernel_size` (int, optional): Base kernel size for the SLL portion. Default is 3.
          - `stride` (int, optional): Stride for the skip connection. Default is 2.
          - `groups` (int, optional): Number of groups for the convolution. Default is 1.
          - `**kwargs`: Additional options (unused).



        References:
            - Boissin, T., Mamalet, F., Fel, T., Picard, A. M., Massena, T., & Serrurier, M. (2025).
            An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures.
            <https://arxiv.org/abs/2501.07930>
        """
        super().__init__()
        inner_kernel_size = kernel_size - (stride - 1)
        self.skip_kernel_size = stride + (stride // 2)
        inner_dim = int(cout * inner_dim_factor)
        self.activation = nn.ReLU()
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        self.kernel = nn.Parameter(
            torch.randn(
                inner_dim, cin // self.groups, inner_kernel_size, inner_kernel_size
            )
        )
        parametrize.register_parametrization(
            self,
            "kernel",
            AOLReparametrizer(
                inner_dim,
                groups=groups,
            ),
        )
        self.bias = nn.Parameter(torch.empty(1, inner_dim, 1, 1))
        self.q = nn.Parameter(torch.ones(inner_dim, 1, 1, 1))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.pre_conv = AdaptiveOrthoConv2d(
            cin, cin, kernel_size=stride, stride=1, bias=False, padding=0, groups=groups
        )
        self.post_conv = AdaptiveOrthoConv2d(
            cin,
            cout,
            kernel_size=stride,
            stride=stride,
            bias=False,
            padding=0,
            groups=groups,
        )

    def forward(self, x):
        # compute t
        # print(self.pre_conv.weight.shape, self.kernel.shape, self.post_conv.weight.shape)
        kernel_1a = fast_matrix_conv(
            self.pre_conv.weight, self.kernel, groups=self.groups
        )
        with parametrize.cached():
            kernel_1b = fast_matrix_conv(
                transpose_kernel(self.kernel, groups=self.groups),
                self.post_conv.weight,
                groups=self.groups,
            )
            kernel_2 = fast_matrix_conv(
                self.pre_conv.weight, self.post_conv.weight, groups=self.groups
            )
            # first branch
            # fuse pre conv with kernel
            res = F.conv2d(x, kernel_1a, padding=self.padding, groups=self.groups)
            res = res + self.bias
            res = self.activation(res)
            res = 2 * F.conv2d(
                res,
                kernel_1b,
                padding=self.padding,
                stride=self.stride,
                groups=self.groups,
            )
            # residual branch
            x = F.conv2d(
                x,
                kernel_2,
                padding=self.skip_kernel_size // 2,
                stride=self.stride,
                groups=self.groups,
            )
        # skip connection
        out = x - res
        return out


class SDPBasedLipschitzResBlock(nn.Module):
    def __init__(self, cin, inner_dim_factor, kernel_size=3, groups=1, **kwargs):
        """
         Original 1-Lipschitz convolutional residual block, based on the SDP-based Lipschitz
        layer (SLL) approach [1]. It has a structure akin to:

        out = x - 2 * ConvTranspose( t * ReLU(Conv(x) + bias) )

        where `t` is a channel-wise scaling factor ensuring a Lipschitz constant ≤ 1.

        !!! note
            By default, `SDPBasedLipschitzResBlock` assumes `cin == cout` and does **not** handle
            stride changes outside the skip connection (i.e., typically used when stride=1 or 2
            for downsampling in a standard residual architecture).

        **Args**:
          - `cin` (int): Number of input channels.
          - `cout` (int): Number of output channels.
          - `inner_dim_factor` (float): Multiplier for the intermediate dimensionality.
          - `kernel_size` (int, optional): Size of the convolution kernel. Default is 3.
          - `groups` (int, optional): Number of groups for the convolution. Default is 1.
          - `**kwargs`: Additional keyword arguments (unused).


        References:
            - Araujo, A., Havens, A. J., Delattre, B., Allauzen, A., & Hu, B.
            A Unified Algebraic Perspective on Lipschitz Neural Networks.
            In The Eleventh International Conference on Learning Representations.
            <https://arxiv.org/abs/2303.03169>
        """
        super().__init__()

        inner_dim = int(cin * inner_dim_factor)
        self.activation = nn.ReLU()
        self.groups = groups

        self.padding = kernel_size // 2

        self.kernel = nn.Parameter(
            torch.randn(inner_dim, cin // groups, kernel_size, kernel_size)
        )
        parametrize.register_parametrization(
            self,
            "kernel",
            AOLReparametrizer(
                inner_dim,
                groups=groups,
            ),
        )
        self.bias = nn.Parameter(torch.empty(1, inner_dim, 1, 1))
        self.q = nn.Parameter(torch.ones(inner_dim, 1, 1, 1))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        res = F.conv2d(x, self.kernel, padding=self.padding, groups=self.groups)
        res = res + self.bias
        res = self.activation(res)
        with parametrize.cached():
            res = 2 * F.conv_transpose2d(
                res, self.kernel, padding=self.padding, groups=self.groups
            )
        out = x - res
        return out


class SDPBasedLipschitzDense(nn.Module):
    def __init__(self, in_features, out_features, inner_dim, **kwargs):
        """
        A 1-Lipschitz fully-connected layer (dense version). Similar to the convolutional
        SLL approach, but operates on vectors:

        $$
        y = x - K^T \\times (t \\times \sigma(K \\times x + b)),
        $$

        **Args**:
          - `in_features` (int): Input size.
          - `out_features` (int): Output size (must match `in_features` to remain 1-Lipschitz).
          - `inner_dim` (int): The internal dimension used for the transform.


        References:
            - Araujo, A., Havens, A. J., Delattre, B., Allauzen, A., & Hu, B.
            A Unified Algebraic Perspective on Lipschitz Neural Networks.
            In The Eleventh International Conference on Learning Representations.
            <https://arxiv.org/abs/2303.03169>
        """
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else in_features
        self.activation = nn.ReLU()

        self.weight = nn.Parameter(torch.empty(inner_dim, in_features))
        self.bias = nn.Parameter(torch.empty(1, inner_dim))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def compute_t(self):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = torch.abs(
            torch.einsum("i,ik,kj,j -> ij", q_inv, self.weight, self.weight.T, q)
        ).sum(1)
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        res = F.linear(x, self.weight)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.linear(res, self.weight.T)
        out = x - res
        return out


class AOCLipschitzResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inner_dim_factor: int,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "circular",
        ortho_params: OrthoParams = OrthoParams(),
    ):
        """
        A Lipschitz residual block in which the main convolution is replaced by
        `AdaptiveOrthoConv2d` (AOC). This preserves 1-Lipschitz (or lower) behavior through
        an orthogonal parameterization, without explicitly computing a scaling factor `t`.

        $$
        y = x - \mathbf{K}^T \\star (\sigma(\\mathbf{K} \\star x + b)),
        $$

        **Args**:
          - `in_channels` (int): Number of input channels.
          - `inner_dim_factor` (int): Multiplier for internal representation size.
          - `kernel_size` (_size_2_t): Convolution kernel size.
          - `dilation` (_size_2_t, optional): Default is 1.
          - `groups` (int, optional): Default is 1.
          - `bias` (bool, optional): If True, adds a learnable bias. Default is True.
          - `padding_mode` (str, optional): `'circular'` or `'zeros'`. Default is `'circular'`.
          - `ortho_params` (OrthoParams, optional): Orthogonal parameterization settings. Default is `OrthoParams()`.


        References:
            - [1] Araujo, A., Havens, A. J., Delattre, B., Allauzen, A., & Hu, B.
            A Unified Algebraic Perspective on Lipschitz Neural Networks.
            In The Eleventh International Conference on Learning Representations.
            <https://arxiv.org/abs/2303.03169>
            - [2] Boissin, T., Mamalet, F., Fel, T., Picard, A. M., Massena, T., & Serrurier, M. (2025).
            An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures.
            <https://arxiv.org/abs/2501.07930>
        """
        super().__init__()

        inner_dim = int(in_channels * inner_dim_factor)
        self.activation = nn.ReLU()

        if padding_mode not in ["circular", "zeros"]:
            raise ValueError("padding_mode must be either 'circular' or 'zeros'")
        if padding_mode == "circular":
            self.padding = 0  # will be handled by the padding function
        else:
            self.padding = kernel_size // 2

        self.in_conv = AdaptiveOrthoConv2d(
            in_channels,
            inner_dim,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            ortho_params=ortho_params,
        )
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

    def forward(self, x):
        kernel = self.in_conv.weight
        # conv
        res = x
        if self.padding_mode == "circular":
            res = F.pad(
                res,
                (self.padding,) * 4,
                mode="circular",
                value=0,
            )
        res = F.conv2d(
            res,
            kernel,
            bias=self.in_conv.bias,
            padding=0,
            groups=self.groups,
        )
        # activation
        res = self.activation(res)
        # conv transpose
        if self.padding_mode == "circular":
            res = F.pad(
                res,
                (self.padding,) * 4,
                mode="circular",
                value=0,
            )
        res = 2 * F.conv_transpose2d(res, kernel, padding=0, groups=self.groups)
        # residual
        out = x - res
        return out
