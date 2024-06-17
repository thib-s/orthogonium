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


class SoftHKRMulticlassLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=10.0,
        min_margin=1.0,
        alpha_mean=0.99,
        temperature=1.0,
    ):
        """
        The multiclass version of HKR with softmax. This is done by computing
        the HKR term over each class and averaging the results.

        Note that `y_true` could be either one-hot encoded, +/-1 values.


        Args:
            alpha (float): regularization factor (0 <= alpha <= 1),
                0 for KR only, 1 for hinge only
            min_margin (float): margin to enforce.
            temperature (float): factor for softmax  temperature
                (higher value increases the weight of the highest non y_true logits)
            alpha_mean (float): geometric mean factor
            one_hot_ytrue (bool): set to True when y_true are one hot encoded (0 or 1),
                and False when y_true already signed bases (for instance +/-1)
            reduction: passed to tf.keras.Loss constructor
            name (str): passed to tf.keras.Loss constructor

        """
        assert (alpha >= 0) and (alpha <= 1), "alpha must in [0,1]"
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.min_margin_v = min_margin
        self.alpha_mean = alpha_mean

        self.current_mean = torch.tensor((self.min_margin_v,), dtype=torch.float32)
        """    constraint=lambda x: torch.clamp(x, 0.005, 1000),
            name="current_mean",
        )"""

        self.temperature = temperature * self.min_margin_v
        if alpha == 1.0:  # alpha = 1.0 => hinge only
            self.fct = self.multiclass_hinge_soft
        else:
            if alpha == 0.0:  # alpha = 0.0 => KR only
                self.fct = self.kr_soft
            else:
                self.fct = self.hkr

        super(SoftHKRMulticlassLoss, self).__init__()

    def clamp_current_mean(self, x):
        return torch.clamp(x, 0.005, 1000)

    def _update_mean(self, y_pred):
        self.current_mean = self.current_mean.to(y_pred.device)
        current_global_mean = torch.mean(torch.abs(y_pred)).to(
            dtype=self.current_mean.dtype
        )
        current_global_mean = (
            self.alpha_mean * self.current_mean
            + (1 - self.alpha_mean) * current_global_mean
        )
        self.current_mean = self.clamp_current_mean(current_global_mean).detach()
        total_mean = current_global_mean
        total_mean = torch.clamp(total_mean, self.min_margin_v, 20000)
        return total_mean

    def computeTemperatureSoftMax(self, y_true, y_pred):
        total_mean = self._update_mean(y_pred)
        current_temperature = (
            torch.clamp(self.temperature / total_mean, 0.005, 250)
            .to(dtype=y_pred.dtype)
            .detach()
        )
        min_value = torch.tensor(torch.finfo(torch.float32).min, dtype=y_pred.dtype).to(
            device=y_pred.device
        )
        opposite_values = torch.where(
            y_true > 0, min_value, current_temperature * y_pred
        )
        F_soft_KR = torch.softmax(opposite_values, dim=-1)
        one_value = torch.tensor(1.0, dtype=F_soft_KR.dtype).to(device=y_pred.device)
        F_soft_KR = torch.where(y_true > 0, one_value, F_soft_KR)
        return F_soft_KR

    def signed_y_pred(self, y_true, y_pred):
        """Return for each item sign(y_true)*y_pred."""
        sign_y_true = torch.where(y_true > 0, 1, -1)  # switch to +/-1
        sign_y_true = sign_y_true.to(dtype=y_pred.dtype)
        return y_pred * sign_y_true

    def multiclass_hinge_preproc(self, signed_y_pred, min_margin):
        """From multiclass_hinge(y_true, y_pred, min_margin)
        simplified to use precalculated signed_y_pred"""
        # compute the elementwise hinge term
        hinge = torch.nn.functional.relu(min_margin / 2.0 - signed_y_pred)
        return hinge

    def multiclass_hinge_soft_preproc(self, signed_y_pred, F_soft_KR):
        hinge = self.multiclass_hinge_preproc(signed_y_pred, self.min_margin_v)
        b = hinge * F_soft_KR
        b = torch.sum(b, axis=-1)
        return b

    def multiclass_hinge_soft(self, y_true, y_pred):
        F_soft_KR = self.computeTemperatureSoftMax(y_true, y_pred)
        signed_y_pred = self.signed_y_pred(y_true, y_pred)
        return self.multiclass_hinge_soft_preproc(signed_y_pred, F_soft_KR)

    def kr_soft_preproc(self, signed_y_pred, F_soft_KR):
        kr = -signed_y_pred
        a = kr * F_soft_KR
        a = torch.sum(a, axis=-1)
        return a

    def kr_soft(self, y_true, y_pred):
        F_soft_KR = self.computeTemperatureSoftMax(y_true, y_pred)
        signed_y_pred = self.signed_y_pred(y_true, y_pred)
        return self.kr_soft_preproc(signed_y_pred, F_soft_KR)

    def hkr(self, y_true, y_pred):
        F_soft_KR = self.computeTemperatureSoftMax(y_true, y_pred)
        signed_y_pred = self.signed_y_pred(y_true, y_pred)

        loss_softkr = self.kr_soft_preproc(signed_y_pred, F_soft_KR)

        loss_softhinge = self.multiclass_hinge_soft_preproc(signed_y_pred, F_soft_KR)
        return (1 - self.alpha) * loss_softkr + self.alpha * loss_softhinge

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not (isinstance(input, torch.Tensor)):  # required for dtype.max
            input = torch.Tensor(input, dtype=input.dtype)
        if not (isinstance(target, torch.Tensor)):
            target = torch.Tensor(target, dtype=input.dtype)
        loss_batch = self.fct(target, input)
        return torch.mean(loss_batch)
