"""
Credit for this file goes to the original authors at https://github.com/blaisedelattre/lip4conv/.

Original Research Paper:
@misc{delattre2024spectralnormconvolutionallayers,
      title={Spectral Norm of Convolutional Layers with Circular and Zero Paddings},
      author={Blaise Delattre and Quentin Barthélemy and Alexandre Allauzen},
      year={2024},
      eprint={2402.00240},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.00240}
}

For further information, please refer to the GitHub repository and the associated publication.

Code were adapted to hande the batched version of this algorithm.
"""

import torch
import torch.nn.functional as F


def compute_delattre2023(X, n=None, n_iter=3):
    """Estimate spectral norm of convolutional layer with Delattre2023.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer for circular padding using [Section.3, Algo. 3] Delattre2023.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, k, k)
        Convolutional filter.
    n : None | int, default=None
        Size of input image. If None, n is set equal to k.
    n_iter : int, default=4
        Number of iterations.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.
    """
    cout, cin, k, _ = X.shape
    if n is None:
        n = k
    if cin > cout:
        X = X.transpose(0, 1)
        cin, cout = cout, cin

    crossed_term = torch.fft.rfft2(X, s=(n, n)).reshape(cout, cin, -1).permute(2, 0, 1)
    inverse_power = 1
    log_curr_norm = torch.zeros(crossed_term.shape[0]).to(crossed_term.device)
    for _ in range(n_iter):
        norm_crossed_term = crossed_term.norm(dim=(1, 2))
        crossed_term /= norm_crossed_term.reshape(-1, 1, 1)
        log_curr_norm = 2 * log_curr_norm + norm_crossed_term.log()
        crossed_term = torch.bmm(crossed_term.conj().transpose(1, 2), crossed_term)
        inverse_power /= 2
    sigma = (
        crossed_term.norm(dim=(1, 2)).pow(inverse_power)
        * ((2 * inverse_power * log_curr_norm).exp())
    ).max()

    return sigma


def compute_delattre2024(X, n_iter=4):
    """Estimate spectral norm of convolutional layer with Delattre2024.

    From a convolutional filter, this function estimates the spectral norm of
    the convolutional layer with zero padding using Delattre2024 [1]_.

    Parameters
    ----------
    X : ndarray, shape (cout, cint, k, k)
        Convolutional filter.
    n_iter : int, default=4
        Number of iterations.
    return_time : bool, default True
        Return computational time.

    Returns
    -------
    sigma : float
        Largest singular value.
    time : float
        If `return_time` is True, it returns the computational time.

    References
    ----------
    .. [1] `Spectral Norm of Convolutional Layers with Circular and Zero Paddings
        <https://arxiv.org/abs/2402.00240>`_
        B Delattre, Q Barthélemy & A Allauzen, arXiv, 2024
    """
    cout, cin, _, _ = X.shape
    if cin > cout:
        X = X.transpose(0, 1)
    rescale_weights = compute_spectral_rescaling_conv(X, n_iter)
    sigma = rescale_weights.max()
    return sigma


def compute_spectral_rescaling_conv(kernel, n_iter=1):
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least equal to 1, got {n_iter}")
    effective_iter = 0
    kkt = kernel
    log_curr_norm = 0
    for _ in range(n_iter):
        padding = kkt.shape[-1] - 1
        kkt_norm = kkt.norm().detach()
        kkt = kkt / kkt_norm
        log_curr_norm = 2 * (log_curr_norm + kkt_norm.log())
        kkt = F.conv2d(kkt, kkt, padding=padding)
        effective_iter += 1
    inverse_power = 2 ** (-effective_iter)
    t = torch.abs(kkt)
    t = t.sum(dim=(1, 2, 3)).pow(inverse_power)
    norm = torch.exp(log_curr_norm * inverse_power)
    t = t * norm
    return t
