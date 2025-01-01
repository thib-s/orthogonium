from dataclasses import dataclass
from typing import Callable
from typing import Tuple
import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn as nn
from orthogonium.model_factory.classparam import ClassParam


class L2Normalize(nn.Module):
    def __init__(self, dtype, dim=None):
        """
        A class that performs L2 normalization for the given input tensor.

        L2 normalization is a process that normalizes the input over a specified
        dimension such that the sum of squares of the elements along that
        dimension equals 1. It ensures that the resulting tensor has a unit norm.
        This operation is widely used in machine learning and deep learning
        applications to standardize feature representations.

        Attributes:
            dim (Optional[int]): The specific dimension along which normalization
                is performed. If None, normalization is done over all dimensions.
            dtype (Any): The data type of the tensor to be normalized.

        Parameters:
            dtype: The data type of the tensor to be normalized.
            dim: An optional integer specifying the dimension along which to
                normalize. If not provided, the input will be normalized globally
                across all dimensions.
        """
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
        BatchedPowerIteration is a class that performs spectral normalization on weights
        using the power iteration method in a batched manner. It initializes singular
        vectors 'u' and 'v', which are used to approximate the largest singular value
        of the associated weight matrix during training. The L2 normalization is applied
        to stabilize these singular vector parameters.

        Attributes:
            weight_shape: tuple
                Shape of the weight tensor. Normalization is applied to the last two dimensions.
            power_it_niter: int
                Number of iterations to perform for the power iteration method.
            eps: float
                A small constant to ensure numerical stability during calculations. Used in the power iteration
                method to avoid dividing by zero.
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


class BatchedIdentity(nn.Module):
    def __init__(self, weight_shape):
        """
        Class representing a batched identity matrix with a specific weight shape. The
        matrix is initialized based on the provided shape of the weights. It is a
        convenient utility for applications where identity-like operations are
        required in a batched manner.

        Attributes:
            weight_shape (Tuple[int, int]): A tuple representing the shape of the
            weight matrix for each batch. (unused)

        Args:
            weight_shape: A tuple specifying the shape of the individual weight matrix.
        """
        super(BatchedIdentity, self).__init__()

    def forward(self, w):
        return w

    def right_inverse(self, w):
        return w


class BatchedBjorckOrthogonalization(nn.Module):
    def __init__(self, weight_shape, beta=0.5, niters=12, pass_through=False):
        """
        Initializes an instance of the BatchedBjorckOrthogonalization class.

        This constructor sets up the necessary attributes to perform batched
        Björck orthogonalization. It determines the suitable operation
        (w_wtw_op or wwt_w_op) based on the shape of the weight matrix.

        Attributes:
            weight_shape: Tuple
                The shape of the weight matrix to be orthogonalized.
            beta: float
                Coefficient to control the convergence of the orthogonalization
                process. Defaults to 0.5.
            niters: int
                Number of iterations for the orthogonalization procedure.
                Defaults to 7.
            wwtw_op: Callable
                The chosen method for calculating the orthogonalized weights.
                This is set based on the dimensions of the weight matrix.

        Parameters:
            weight_shape: Tuple
                Specifies the shape of the weight matrix.
            beta:
                Sets the convergence coefficient value. Defaults to 0.5.
            niters:
                Specifies the iteration count for the orthogonalization algorithm.
                Defaults to 12.
        """
        self.weight_shape = weight_shape
        self.beta = beta
        self.niters = niters
        self.pass_through = pass_through
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
        if self.pass_through:
            with torch.no_grad():
                for _ in range(self.niters):
                    w = (1 + self.beta) * w - self.beta * self.wwtw_op(w)
            # Final iteration without no_grad, using parameters:
            w = (1 + self.beta) * w - self.beta * self.wwtw_op(w)
        else:
            for _ in range(self.niters):
                w = (1 + self.beta) * w - self.beta * self.wwtw_op(w)
        return w

    def right_inverse(self, w):
        return w


class BatchedCholeskyOrthogonalization(nn.Module):
    def __init__(self, weight_shape, stable=False):
        """
        Initializes a BatchedCholeskyOrthogonalization instance. Depending on the stable
        flag, it selects the orthogonalization function to be used.

        Attributes:
        weight_shape: The shape of the weight matrix that this instance will work with.
        orth: Function used for performing batched Cholesky-based orthogonalization,
              chosen based on the stable parameter.

        Parameters:
        weight_shape: The shape of the weight matrix, typically a tuple of integers.
        stable: A boolean indicating whether to use the stable version of the
                orthogonalization function. Default is False.

        """
        self.weight_shape = weight_shape
        super(BatchedCholeskyOrthogonalization, self).__init__()
        if stable:
            self.orth = BatchedCholeskyOrthogonalization.CholeskyOrthfn_stable.apply
        else:
            self.orth = BatchedCholeskyOrthogonalization.CholeskyOrthfn.apply

    # @staticmethod
    # def orth(X):
    #     S = X @ X.mT
    #     eps = S.diagonal(dim1=1, dim2=2).mean(1).mul(1e-3).detach()
    #     eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device)
    #     S = S + eps.view(-1, 1, 1) * eye.unsqueeze(0)
    #     L = torch.linalg.cholesky(S)
    #     W = torch.linalg.solve_triangular(L, X, upper=False)
    #     return W

    class CholeskyOrthfn(torch.autograd.Function):
        @staticmethod
        # def forward(ctx, X):
        #     S = X @ X.mT
        #     eps = S.diagonal(dim1=1, dim2=2).mean(1).mul(1e-3)
        #     eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device)
        #     S = S + eps.view(-1, 1, 1) * eye.unsqueeze(0)
        #     L = torch.linalg.cholesky(S)
        #     W = torch.linalg.solve_triangular(L, X, upper=False)
        #     ctx.save_for_backward(W, L)
        #     return W
        def forward(ctx, X):
            S = X @ X.mT
            eps = 1e-3  # A common stable choice
            S = S + eps * torch.eye(
                S.size(-1), dtype=S.dtype, device=S.device
            ).unsqueeze(0)
            L = torch.linalg.cholesky(S)
            W = torch.linalg.solve_triangular(L, X, upper=False)
            ctx.save_for_backward(W, L)
            return W

        @staticmethod
        def backward(ctx, grad_output):
            W, L = ctx.saved_tensors
            LmT = L.mT.contiguous()
            gB = torch.linalg.solve_triangular(LmT, grad_output, upper=True)
            gA = (-gB @ W.mT).tril()
            gS = (LmT @ gA).tril()
            gS = gS + gS.tril(-1).mT
            gS = torch.linalg.solve_triangular(LmT, gS, upper=True)
            gX = gS @ W + gB
            return gX

    class CholeskyOrthfn_stable(torch.autograd.Function):
        @staticmethod
        def forward(ctx, X):
            S = X @ X.mT
            eps = 1e-3  # A common stable choice
            S = S + eps * torch.eye(
                S.size(-1), dtype=S.dtype, device=S.device
            ).unsqueeze(0)
            L = torch.linalg.cholesky(S)
            W = torch.linalg.solve_triangular(L, X, upper=False)
            ctx.save_for_backward(X, W, L)
            return W

        @staticmethod
        def backward(ctx, grad_output):
            X, W, L = ctx.saved_tensors
            gB = torch.linalg.solve_triangular(L.mT, grad_output, upper=True)
            gA = (-gB @ W.mT).tril()
            gS = (L.mT @ gA).tril()
            gS = gS + gS.tril(-1).mT
            gS = torch.linalg.solve_triangular(L.mT, gS, upper=True)
            gS = torch.linalg.solve_triangular(L, gS, upper=False, left=False)
            gX = gS @ X + gB
            return gX

    def forward(self, w):
        return self.orth(w).view(*self.weight_shape)

    def right_inverse(self, w):
        return w


class BatchedExponentialOrthogonalization(nn.Module):
    def __init__(self, weight_shape, niters=7):
        """
        Initialize a BatchedExponentialOrthogonalization instance. This class is used to
        perform stabilization and normalization of weights in neural network through
        exponential map-based orthogonalization. Suitable for batched operations.

        Attributes:
        weight_shape: Tuple[int, int]
            The shape of the weight matrix, usually two-dimensional.
        max_dim: int
            Maximum dimension from the last two dimensions of weight_shape. Used
            internally for computations.
        niters: int
            The number of iterations used for convergence in the orthogonalization
            process.

        Parameters:
        weight_shape: tuple
            A tuple representing the shape of the weight matrix.
        niters: int, optional
            The number of iterations for stabilization and orthogonalization. Default
            is 7.
        """
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
        """
        A class for performing batched QR orthogonalization.

        This class is used to initialize an instance of BatchedQROrthogonalization with the
        specified weight shape. It is designed to handle batched QR decomposition operations
        in applications where orthogonalization of batches of matrices is required.

        Attributes:
            weight_shape: The shape of the weights, which is expected to define the shape
            of the matrices to be orthogonalized.

        Args:
            weight_shape (tuple): Shape of the weights used for batched QR orthogonalization.
        """
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
        return q.contiguous()

    def right_inverse(self, w):
        return w


@dataclass
class OrthoParams:
    """
    Represents the parameters and configurations used for orthogonalization
    and spectral normalization.

    This class encapsulates the necessary modules and settings required
    for performing spectral normalization and orthogonalization of tensors
    in a parameterized way. It accommodates various implementations of
    normalizers and orthogonalization techniques to provide flexibility
    in their application. This way we can easily switch between different
    normalization techniques inside our layer despite that each normalization
    have different parameters.

    Attributes:
        spectral_normalizer (Callable[Tuple[int, ...], nn.Module]): A callable
            that produces a module for spectral normalization. Default is
            configured to use BatchedPowerIteration with specific parameters.
            This callable can be provided either as a `functool.partial` or as a
            `orthogonium.ClassParam`. It will recieve the shape of the weight tensor as its
            argument.
        orthogonalizer (Callable[Tuple[int, ...], nn.Module]): A callable
            that produces a module for orthogonalization. Default is
            configured to use BatchedBjorckOrthogonalization with specific
            parameters. This callable can be provided either as a `functool.partial` or as a
            `orthogonium.ClassParam`. It will recieve the shape of the weight tensor as its argument.
        contiguous_optimization (bool): Determines whether to perform
            optimization ensuring contiguous operations. Default is False.
    """

    # spectral_normalizer: Callable[Tuple[int, ...], nn.Module] = BatchedIdentity
    spectral_normalizer: Callable[Tuple[int, ...], nn.Module] = ClassParam(  # type: ignore
        BatchedPowerIteration, power_it_niter=3, eps=1e-6
    )
    orthogonalizer: Callable[Tuple[int, ...], nn.Module] = ClassParam(  # type: ignore
        BatchedBjorckOrthogonalization,
        beta=0.5,
        niters=12,
        pass_through=False,
        # ClassParam(BatchedExponentialOrthogonalization, niters=12)
        # BatchedCholeskyOrthogonalization,
        # BatchedQROrthogonalization,
    )
    contiguous_optimization: bool = False


DEFAULT_ORTHO_PARAMS = OrthoParams()
BJORCK_PASS_THROUGH_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=ClassParam(BatchedPowerIteration, power_it_niter=3, eps=1e-6),  # type: ignore
    orthogonalizer=ClassParam(
        BatchedBjorckOrthogonalization, beta=0.5, niters=12, pass_through=True
    ),
    contiguous_optimization=False,
)
DEFAULT_TEST_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=ClassParam(BatchedPowerIteration, power_it_niter=3, eps=1e-6),  # type: ignore
    orthogonalizer=ClassParam(BatchedBjorckOrthogonalization, beta=0.5, niters=25),
    # orthogonalizer=ClassParam(BatchedQROrthogonalization),
    # orthogonalizer=ClassParam(BatchedExponentialOrthogonalization, niters=12),  # type: ignore
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
CHOLESKY_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=BatchedIdentity,  # type: ignore
    orthogonalizer=ClassParam(BatchedCholeskyOrthogonalization),  # type: ignore
    contiguous_optimization=False,
)

CHOLESKY_STABLE_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=BatchedIdentity,
    orthogonalizer=ClassParam(BatchedCholeskyOrthogonalization, stable=True),
    contiguous_optimization=False,
)
