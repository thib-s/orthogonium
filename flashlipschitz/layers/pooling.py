import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class ScaledAvgPool2d(nn.AvgPool2d):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: bool = None,
        k_coef_lip: float = 1.0,
    ):
        """
        Layer from the deel-torchlip project: https://github.com/deel-ai/deel-torchlip/blob/master/deel/torchlip/modules/pooling.py
        Average pooling operation for spatial data, but with a lipschitz bound.

        Args:
            kernel_size: The size of the window.
            stride: The stride of the window. Must be None or equal to
                ``kernel_size``. Default value is ``kernel_size``.
            padding: Implicit zero-padding to be added on both sides. Must
                be zero.
            ceil_mode: When True, will use ceil instead of floor to compute the output
                shape.
            count_include_pad: When True, will include the zero-padding in the averaging
                calculation.
            divisor_override: If specified, it will be used as divisor, otherwise
                ``kernel_size`` will be used.
            k_coef_lip: The Lipschitz factor to ensure. The output will be scaled
                by this factor.

        This documentation reuse the body of the original torch.nn.AveragePooling2D
        doc.
        """
        torch.nn.AvgPool2d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        if isinstance(kernel_size, tuple):
            self.scalingFactor = math.sqrt(np.prod(np.asarray(kernel_size)))
        else:
            self.scalingFactor = kernel_size

        if self.stride != self.kernel_size:
            raise RuntimeError("stride must be equal to kernel_size.")
        if np.sum(self.padding) != 0:
            raise RuntimeError(f"{type(self)} does not support padding.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.AvgPool2d.forward(self, input) * self.scalingFactor
