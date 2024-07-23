from dataclasses import dataclass
from typing import Callable
from typing import Tuple
from typing import Union

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn as nn

from flashlipschitz.classparam import ClassParam


class L2Normalize(nn.Module):
    def __init__(self, dtype, dim=None):
        super(L2Normalize, self).__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, x):
        return x / (torch.norm(x, dim=self.dim, keepdim=True, dtype=self.dtype) + 1e-8)

    def right_inverse(self, x):
        return x / (torch.norm(x, dim=self.dim, keepdim=True, dtype=self.dtype) + 1e-8)


class BatchedPowerIteration(nn.Module):
    def __init__(self, weight_shape, power_it_niter=3, eps=1e-12):
        """
        This module is a batched version of the Power Iteration algorithm.
        It is used to normalize the kernel of a convolutional layer.

        Args:
            weight_shape (tuple): shape of the kernel, the last dimension will be normalized.
            power_it_niter (int, optional): number of iterations. Defaults to 3.
            eps (float, optional): small value to avoid division by zero. Defaults to 1e-12.
        """
        super(BatchedPowerIteration, self).__init__()
        self.weight_shape = weight_shape
        self.power_it_niter = power_it_niter
        self.eps = eps
        # init u
        # u will be weight_shape[:-2] + (weight_shape[:-2], 1)
        # v will be weight_shape[:-2] + (weight_shape[:-1], 1,)
        self.u = nn.Parameter(
            torch.Tensor(torch.randn(*weight_shape[:-2], weight_shape[-2], 1)),
            requires_grad=False,
        )
        self.v = nn.Parameter(
            torch.Tensor(torch.randn(*weight_shape[:-2], weight_shape[-1], 1)),
            requires_grad=False,
        )
        parametrize.register_parametrization(
            self, "u", L2Normalize(dtype=self.u.dtype, dim=(-2))
        )
        parametrize.register_parametrization(
            self, "v", L2Normalize(dtype=self.v.dtype, dim=(-2))
        )

    def forward(self, X, init_u=None, n_iters=3, return_uv=True):
        for _ in range(n_iters):
            self.v = X.transpose(-1, -2) @ self.u
            self.u = X @ self.v
        # stop gradient on u and v
        u = self.u.detach()
        v = self.v.detach()
        # but keep gradient on s
        s = u.transpose(-1, -2) @ X @ v
        return X / (s + self.eps)

    def right_inverse(self, normalized_kernel):
        # we assume that the kernel is normalized
        return normalized_kernel.to(self.u.dtype)


class BatchedBjorckOrthogonalization(nn.Module):
    def __init__(self, weight_shape, beta=0.5, niters=7):
        self.weight_shape = weight_shape
        self.beta = beta
        self.niters = niters
        if weight_shape[-2] < weight_shape[-1]:
            self.wwtw_op = BatchedBjorckOrthogonalization.wwt_w_op
        else:
            self.wwtw_op = BatchedBjorckOrthogonalization.w_wtw_op
        super(BatchedBjorckOrthogonalization, self).__init__()

    @staticmethod
    def w_wtw_op(w):
        return w @ (w.transpose(-1, -2) @ w)

    @staticmethod
    def wwt_w_op(w):
        return (w @ w.transpose(-1, -2)) @ w

    def forward(self, w):
        for _ in range(self.niters):
            w = (1 + self.beta) * w - self.beta * self.wwtw_op(w)
        return w

    def right_inverse(self, w):
        return w


class BatchedExponentialOrthogonalization(nn.Module):
    def __init__(self, weight_shape, niters=7):
        self.weight_shape = weight_shape
        self.max_dim = max(weight_shape[-2:])
        self.niters = niters
        super(BatchedExponentialOrthogonalization, self).__init__()

    def forward(self, w):
        # fill w with zero to have a square matrix over the last two dimensions
        # if ((self.max_dim - w.shape[-1]) != 0) and ((self.max_dim - w.shape[-2]) != 0):
        w = torch.nn.functional.pad(
            w, (0, self.max_dim - w.shape[-1], 0, self.max_dim - w.shape[-2])
        )
        # makes w skew symmetric
        w = (w - w.transpose(-1, -2)) / 2
        acc = w
        res = torch.eye(acc.shape[-2], acc.shape[-1], device=w.device) + acc
        for i in range(2, self.niters):
            acc = torch.einsum("...ij,...jk->...ik", acc, w) / i
            res = res + acc
        # if transpose:
        #     res = res.transpose(-1, -2)
        res = res[..., : self.weight_shape[-2], : self.weight_shape[-1]]
        return res.contiguous()

    def right_inverse(self, w):
        return w


class BatchedQROrthogonalization(nn.Module):
    def __init__(self, weight_shape):
        super(BatchedQROrthogonalization, self).__init__()

    def forward(self, w):
        transpose = w.shape[-2] < w.shape[-1]
        if transpose:
            w = w.transpose(-1, -2)
        q, r = torch.linalg.qr(w, mode="reduced")
        # compute the sign of the diagonal of d
        diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
        # multiply the sign with the diagonal of r
        q = q * diag_sign
        if transpose:
            q = q.transpose(-1, -2)
        return q

    def right_inverse(self, w):
        return w


@dataclass
class OrthoParams:
    spectral_normalizer: Callable[Tuple[int, ...], nn.Module] = ClassParam(  # type: ignore
        BatchedPowerIteration, power_it_niter=3, eps=1e-6
    )
    orthogonalizer: Callable[Tuple[int, ...], nn.Module] = ClassParam(  # type: ignore
        BatchedBjorckOrthogonalization,
        beta=0.5,
        niters=12,
        # ClassParam(BatchedExponentialOrthogonalization, niters=12)
    )
    contiguous_optimization: bool = False


DEFAULT_ORTHO_PARAMS = OrthoParams()
DEFAULT_TEST_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=ClassParam(BatchedPowerIteration, power_it_niter=3, eps=1e-6),  # type: ignore
    # orthogonalizer=ClassParam(BatchedBjorckOrthogonalization, beta=0.5, niters=25),
    # orthogonalizer=ClassParam(BatchedQROrthogonalization),
    orthogonalizer=ClassParam(BatchedExponentialOrthogonalization, niters=12),  # type: ignore
    contiguous_optimization=False,
)
EXP_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=ClassParam(BatchedPowerIteration, power_it_niter=3, eps=1e-6),  # type: ignore
    orthogonalizer=ClassParam(BatchedExponentialOrthogonalization, niters=12),  # type: ignore
    contiguous_optimization=False,
)
QR_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=ClassParam(BatchedPowerIteration, power_it_niter=3, eps=1e-3),  # type: ignore
    orthogonalizer=ClassParam(BatchedQROrthogonalization),  # type: ignore
    contiguous_optimization=False,
)
