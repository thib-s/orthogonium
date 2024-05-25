import math

import torch
import torch.nn as nn


def VRA(output, class_indices, L=1.0, eps=36 / 255, return_certs=False):
    """Compute the verified robust accuracy (VRA) of a model's output.

    Args:
        output : torch.Tensor
            The output of the model.
        class_indices : torch.Tensor
            The indices of the correct classes. Should not be one-hot encoded.
        L : float
            The Lipschitz constant of the model.
        eps : float
            The perturbation size.
        return_certs : bool
            Whether to return the certificates instead of the VRA.

    Returns:
        vra : torch.Tensor
            The VRA of the model.
    """
    batch_size = output.shape[0]
    batch_indices = torch.arange(batch_size)

    # get the values of the correct class
    output_class_indices = output[batch_indices, class_indices]
    # get the values of the top class that is not the correct class
    # create a mask indicating the correct class
    onehot = torch.zeros_like(output).cuda()
    onehot[torch.arange(output.shape[0]), class_indices] = 1.0
    # subtracting a large number from the correct class to ensure it is not the max
    # doing so will allow us to find the top of the output that is not the correct class
    output_trunc = output - onehot * 1e6
    output_nextmax = torch.max(output_trunc, dim=1)[0]
    # now we can compute the certificates
    output_diff = output_class_indices - output_nextmax
    certs = output_diff / (math.sqrt(2) * L)
    # now we can compute the vra
    # vra is percentage of certs > eps
    vra = (certs > eps).float()
    if return_certs:
        return certs
    return vra


# criterion = (
#     lambda yp, yt: -torch.nn.functional.cosine_similarity(
#         yp, torch.nn.functional.one_hot(yt, 1000)
#     ).mean()
#     + 0.1
#     * torch.clamp(
#         VRA(yp, yt, L=2 / 0.225, eps=36 / 255, return_certs=True), 0, 36 / 255
#     ).mean()
# )


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, yp, yt):
        return -torch.nn.functional.cosine_similarity(
            yp, torch.nn.functional.one_hot(yt, yp.shape[1])
        ).mean()


class Cosine_VRA_Loss(nn.Module):
    def __init__(self, gamma=0.1, L=1.0, eps=36 / 255):
        super(Cosine_VRA_Loss, self).__init__()
        self.gamma = gamma
        self.L = L
        self.eps = eps

    def forward(self, yp, yt):
        return -(
            (
                (1 - self.gamma)
                * torch.nn.functional.cosine_similarity(
                    yp, torch.nn.functional.one_hot(yt, yp.shape[1])
                )
            )
            + (
                self.gamma
                * torch.clamp(
                    VRA(yp, yt, L=self.L, eps=self.eps, return_certs=True), 0, self.eps
                )
            )
        ).mean()
