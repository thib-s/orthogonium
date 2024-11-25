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
    """Estimate spectral norm of convolutional layers with Delattre2024.

    Parameters
    ----------
    X : tensor, shape (batch_size, cout, cin, k, k)
        Batch of convolutional filters.
    n_iter : int, default=4
        Number of iterations.

    Returns
    -------
    sigma : tensor, shape (batch_size,)
        Largest singular values for each kernel in the batch.
    time : float
        If `return_time` is True, it returns the computational time.
    """
    batch_size, cout, cin, _, _ = X.shape
    if cin > cout:
        X = X.transpose(1, 2).contiguous()  # Swap cout and cin dimensions
    else:
        X = X.contiguous()
    rescale_weights = compute_spectral_rescaling_conv(X, n_iter)
    sigma = rescale_weights.max(dim=1)[0]  # Max over output channels
    return sigma


def compute_spectral_rescaling_conv(kernel, n_iter=1):
    batch_size, cout, cin, kw, kh = kernel.shape
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least equal to 1, got {n_iter}")
    effective_iter = 0
    kkt = kernel
    log_curr_norm = torch.zeros(batch_size, device=kkt.device)  # Shape: (batch_size,)
    for _ in range(n_iter):
        # Compute norms over each kernel in the batch
        kkt_norm = (
            kkt.reshape(batch_size, -1).norm(dim=1, keepdim=False).detach()
        )  # Shape: (batch_size, 1)
        kkt = kkt / kkt_norm.view(-1, 1, 1, 1, 1)
        log_curr_norm = 2 * (log_curr_norm + kkt_norm.view(batch_size).log())
        # Perform grouped convolution
        kkt_resh = kkt.view(batch_size * cout, cin, kw, kh)
        kkt = F.conv2d(
            kkt.permute(1, 0, 2, 3, 4).reshape(cout, batch_size * cin, kw, kh),
            kkt.view(batch_size * cout, cin, kw, kh),
            padding=(kw - 1, kh - 1),
            groups=batch_size,
        )
        kkt = kkt.permute(1, 0, 2, 3).reshape(
            batch_size, -1, kkt.shape[0], kkt.shape[2], kkt.shape[3]
        )
        batch_size, cout, cin, kw, kh = kkt.shape
        effective_iter += 1
    inverse_power = 2 ** (-effective_iter)
    t = torch.abs(kkt)
    t = t.sum(dim=(2, 3, 4)).pow(inverse_power)
    norm = torch.exp(log_curr_norm * inverse_power)
    t = t * norm.unsqueeze(-1)
    return t  # Shape: (batch_size, cout)


def batch_group_conv2d(input, weight, padding=0):
    """
    Perform group convolution over a batch of inputs and weights.

    Parameters
    ----------
    input : tensor, shape (batch_size, cout, cin, H, W)
    weight : tensor, shape (batch_size, cout, cin, kH, kW)
    padding : int, default=0

    Returns
    -------
    output : tensor, shape (batch_size, cout, cout, H_out, W_out)
    """
    batch_size, cout, cin, H, W = input.shape
    _, _, _, kH, kW = weight.shape
    # Reshape for grouped convolution
    input = input.view(1, batch_size * cout, cin, H, W)
    weight = weight.view(batch_size * cout, cin, kH, kW)
    # Perform grouped convolution
    output = F.conv2d(
        input=input.view(batch_size * cout, cin, H, W),
        weight=weight,
        groups=batch_size,
        padding=padding,
    )
    # Reshape back to (batch_size, cout, cout, H_out, W_out)
    output = output.view(batch_size, cout, cout, output.shape[-2], output.shape[-1])
    return output
