import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def power_iteration(A, init_u=None, n_iters=20, return_uv=True):
    shape = list(A.shape)
    shape[-1] = 1
    shape = tuple(shape)
    if init_u is None:
        u = torch.randn(*shape, dtype=A.dtype, device=A.device)
    else:
        assert tuple(init_u.shape) == shape, (init_u.shape, shape)
        u = init_u
    for _ in range(n_iters):
        v = A.transpose(-1, -2) @ u
        v /= v.norm(dim=-2, keepdim=True)
        u = A @ v
        u /= u.norm(dim=-2, keepdim=True)
    s = (u.transpose(-1, -2) @ A @ v).squeeze(-1).squeeze(-1)
    if return_uv:
        return u, s, v
    return s


def bjorck_orthonormalize(
    w, beta=0.5, iters=20, order=1, power_iteration_scaling=False, default_scaling=False
):
    if w.shape[-2] < w.shape[-1]:
        return bjorck_orthonormalize(
            w.transpose(-1, -2),
            beta=beta,
            iters=iters,
            order=order,
            power_iteration_scaling=power_iteration_scaling,
            default_scaling=default_scaling,
        ).transpose(-1, -2)

    if power_iteration_scaling:
        with torch.no_grad():
            s = power_iteration(w, return_uv=False)
        w = w / s.unsqueeze(-1).unsqueeze(-1)
    elif default_scaling:
        w = w / ((w.shape[0] * w.shape[1]) ** 0.5)
    assert order == 1, "only first order Bjorck is supported"
    for _ in range(iters):
        w_t_w = w.transpose(-1, -2) @ w
        w = (1 + beta) * w - beta * w @ w_t_w
    return w


def orthogonal_matrix(n):
    a = torch.randn((n, n))
    q, r = torch.qr(a)
    return q * torch.sign(torch.diag(r))


def symmetric_projection(n, ortho_matrix, mask=None):
    q = ortho_matrix
    # randomly zeroing out some columns
    if mask is None:
        mask = (torch.randn(n) > 0).float()
    c = q * mask
    return c.mm(c.t())


def block_orth(p1, p2):
    assert p1.shape == p2.shape
    n = p1.size(0)
    kernel2x2 = {}
    eye = torch.eye(n, device=p1.device, dtype=p1.dtype)
    kernel2x2[0, 0] = p1.mm(p2)
    kernel2x2[0, 1] = p1.mm(eye - p2)
    kernel2x2[1, 0] = (eye - p1).mm(p2)
    kernel2x2[1, 1] = (eye - p1).mm(eye - p2)
    return kernel2x2


def matrix_conv(m1, m2):
    n = (m1[0, 0]).size(0)
    if n != (m2[0, 0]).size(0):
        raise ValueError(
            "The entries in matrices m1 and m2 " "must have the same dimensions!"
        )
    k = int(np.sqrt(len(m1)))
    l = int(np.sqrt(len(m2)))
    result = {}
    size = k + l - 1
    # Compute matrix convolution between m1 and m2.
    for i in range(size):
        for j in range(size):
            result[i, j] = torch.zeros(
                (n, n), device=m1[0, 0].device, dtype=m1[0, 0].dtype
            )
            for index1 in range(min(k, i + 1)):
                for index2 in range(min(k, j + 1)):
                    if (i - index1) < l and (j - index2) < l:
                        result[i, j] += m1[index1, index2].mm(
                            m2[i - index1, j - index2]
                        )
    return result


def dict_to_tensor(x, k1, k2):
    return torch.stack([torch.stack([x[i, j] for j in range(k2)]) for i in range(k1)])


def convolution_orthogonal_generator_projs(ksize, cin, cout, ortho, sym_projs):
    flipped = False
    if ksize == 1:
        return ortho.t().unsqueeze(-1).unsqueeze(-1)
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
        ortho = ortho.t()
    p = block_orth(sym_projs[0], sym_projs[1])
    for _ in range(1, ksize - 1):
        p = matrix_conv(p, block_orth(sym_projs[_ * 2], sym_projs[_ * 2 + 1]))
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = ortho.mm(p[i, j])
    if flipped:
        return dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0)
    return dict_to_tensor(p, ksize, ksize).permute(3, 2, 1, 0)


def convolution_orthogonal_generator(ksize, cin, cout, P, Q):
    flipped = False
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
    orth = orthogonal_matrix(cout)[0:cin, :]
    if ksize == 1:
        return orth.unsqueeze(0).unsqueeze(0)

    p = block_orth(symmetric_projection(cout, P[0]), symmetric_projection(cout, Q[0]))
    for _ in range(ksize - 2):
        temp = block_orth(
            symmetric_projection(cout, P[_ + 1]), symmetric_projection(cout, Q[_ + 1])
        )
        p = matrix_conv(p, temp)
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = orth.mm(p[i, j])
    if flipped:
        return dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0)
    return dict_to_tensor(p, ksize, ksize).permute(3, 2, 1, 0)


def convolution_orthogonal_initializer(ksize, cin, cout):
    P, Q = [], []
    cmax = max(cin, cout)
    for i in range(ksize - 1):
        P.append(orthogonal_matrix(cmax))
        Q.append(orthogonal_matrix(cmax))
    P, Q = map(torch.stack, (P, Q))
    return convolution_orthogonal_generator(ksize, cin, cout, P, Q)


class BCOP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        padding_mode="circular",
        bias=True,
        bjorck_beta=0.5,
        bjorck_iters=12,
        bjorck_order=1,
        power_iteration_scaling=True,
    ):
        super(BCOP, self).__init__()
        assert (stride == 1) or (stride == 2)
        self.in_channels = in_channels * stride * stride
        self.out_channels = out_channels

        self.max_channels = max(self.out_channels, self.in_channels)

        self.stride = stride
        self.kernel_size = kernel_size
        self.num_kernels = 2 * self.kernel_size - 1

        self.bjorck_iters = bjorck_iters
        self.bjorck_beta = bjorck_beta
        self.bjorck_order = bjorck_order

        self.buffer_weight = None

        self.power_iteration_scaling = power_iteration_scaling

        # Define the unconstrained matrices Ms and Ns for Ps and Qs
        self.conv_matrices = nn.Parameter(
            torch.Tensor(self.num_kernels, self.max_channels, self.max_channels),
            requires_grad=True,
        )

        # The mask controls the rank of the symmetric projectors (full half rank).
        self.mask = nn.Parameter(
            torch.cat(
                (
                    torch.ones(self.num_kernels - 1, 1, self.max_channels // 2),
                    torch.zeros(
                        self.num_kernels - 1,
                        1,
                        self.max_channels - (self.max_channels // 2),
                    ),
                ),
                dim=-1,
            ),
            requires_grad=False,
        )

        # Bias parameters in the convolution
        self.enable_bias = bias
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels).cuda(), requires_grad=True
            )
        else:
            self.bias = None

        # Initialize the weights (self.weight is set to zero for streamline module)
        self.reset_parameters()

    def reset_parameters(self):
        for index in range(self.num_kernels):
            nn.init.orthogonal_(self.conv_matrices[index])

        stdv = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        self._input_shape = x.shape[2:]

        # orthogonalize all the matrices using Bjorck
        if self.training or self.buffer_weight is None:
            ortho = bjorck_orthonormalize(
                self.conv_matrices,
                beta=self.bjorck_beta,
                iters=self.bjorck_iters,
                power_iteration_scaling=self.power_iteration_scaling,
                default_scaling=not self.power_iteration_scaling,
            )

            # compute the symmetric projectors
            H = ortho[0, : self.in_channels, : self.out_channels]
            PQ = ortho[1:]
            if self.kernel_size > 1:
                PQ = PQ * self.mask
                PQ = PQ @ PQ.transpose(-1, -2)

            # compute the resulting convolution kernel using block convolutions
            weight = convolution_orthogonal_generator_projs(
                self.kernel_size, self.in_channels, self.out_channels, H, PQ
            )
            self.buffer_weight = weight
        else:
            weight = self.buffer_weight

        # detach the weight when we are using the cached weights from previous steps
        bias = self.bias

        # apply cyclic padding to the input and perform a standard convolution
        x_orig = x
        if self.stride > 1:
            x = rearrange(
                x,
                "b c (w k1) (h k2) -> b (c k1 k2) w h",
                k1=self.stride,
                k2=self.stride,
            )

        p4d = (
            self.kernel_size // 2,
            self.kernel_size // 2,
            self.kernel_size // 2,
            self.kernel_size // 2,
        )
        x_pad = F.pad(x, p4d, mode="circular")

        z = F.conv2d(x_pad, weight)
        if self.enable_bias:
            z = z + self.bias.view(1, -1, 1, 1)
        return z

    # def extra_repr(self):
    #     return "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, bias={enable_bias}, mask_half={mask_half}, ortho_mode={ortho_mode}".format(
    #         **self.__dict__
    #     )
