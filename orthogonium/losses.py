import math

import torch
import torch.nn as nn

from orthogonium.layers import OrthoLinear
from orthogonium.layers import UnitNormLinear


def check_last_linear_layer_type(model):
    """
    Determines the type of the last linear layer in a given model.

    This function inspects the architecture of the model and identifies the last
    linear layer of specific types (nn.Linear, OrthoLinear, UnitNormLinear). It
    then returns a string indicating the type of the last linear layer based on
    its class. This allows to determine the parameter to use for computing the
    VRA of a model's output.

    Args:
        model: The model containing layers to be inspected.

    Returns:
        str: A string indicating the type of the last linear layer.
             The possible values are:
                 - "global" if the layer is of type OrthoLinear.
                 - "classwise" if the layer is of type UnitNormLinear.
                 - "unknown" if the layer is of any other type or if no
                   linear layer is found.
    """
    # Find the last linear layer in the model
    last_linear_layer = None
    layers = list(model.children())
    for layer in reversed(layers):
        if (
            isinstance(layer, nn.Linear)
            or isinstance(layer, OrthoLinear)
            or isinstance(layer, UnitNormLinear)
        ):
            last_linear_layer = layer
            break

    # Check the type of the last linear layer
    if isinstance(last_linear_layer, OrthoLinear):
        return "global"
    elif isinstance(last_linear_layer, UnitNormLinear):
        return "classwise"
    else:
        return "unknown"


def VRA(
    output,
    class_indices,
    last_layer_type="classwise",
    L=1.0,
    eps=36 / 255,
    return_certs=False,
):
    """Compute the verified robust accuracy (VRA) of a model's output.

    Args:
        output : torch.Tensor
            The output of the model.
        class_indices : torch.Tensor
            The indices of the correct classes. Should not be one-hot encoded.
        last_layer_type : str
            The type of the last layer of the model. Should be either "classwise" (L-lip per class) or "global" (L-lip globally).
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
    if last_layer_type == "global":
        den = math.sqrt(2) * L
    elif last_layer_type == "classwise":
        den = 2 * L
    else:
        raise ValueError(
            "[VRA] last_layer_type should be either 'global' or 'classwise'"
        )
    certs = output_diff / den
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


class LossXent(nn.Module):
    def __init__(self, n_classes, offset=2.12132, temperature=0.25):
        """
        A custom loss function class for cross-entropy calculation.

        This class initializes a cross-entropy loss criterion along with additional
        parameters, such as an offset and a temperature factor, to allow a finer control over
        the accuracy/robustness tradeoff during training.

        Attributes:
            criterion (nn.CrossEntropyLoss): The PyTorch cross-entropy loss criterion.
            n_classes (int): The number of classes present in the dataset.
            offset (float): An offset value for customizing the loss computation.
            temperature (float): A temperature factor for scaling logits during loss calculation.

        Parameters:
            n_classes (int): The number of classes in the dataset.
            offset (float, optional): The offset value for loss computation. Default is 2.12132.
            temperature (float, optional): The temperature scaling factor. Default is 0.25.
        """
        super(LossXent, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.offset = offset
        self.temperature = temperature

    def __call__(self, outputs, labels):
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=self.n_classes)
        offset_outputs = outputs - self.offset * one_hot_labels
        offset_outputs /= self.temperature
        loss = self.criterion(offset_outputs, labels) * self.temperature
        return loss


class CosineLoss(nn.Module):
    def __init__(self):
        """
        A class that implements the Cosine Loss for measuring the cosine similarity
        between predictions and targets. Designed for use in scenarios involving
        angle-based loss calculations or similarity measurements.

        Attributes:
            None

        """
        super(CosineLoss, self).__init__()

    def forward(self, yp, yt):
        return -torch.nn.functional.cosine_similarity(
            yp, torch.nn.functional.one_hot(yt, yp.shape[1])
        ).mean()


class Cosine_VRA_Loss(nn.Module):
    def __init__(self, gamma=0.1, L=1.0, eps=36 / 255, last_layer_type="classwise"):
        super(Cosine_VRA_Loss, self).__init__()
        self.gamma = gamma
        self.L = L
        self.eps = eps
        self.last_layer_type = last_layer_type

    def SoftVRA(self, yp, yt):
        batch_size, nb_classes = yp.shape
        # create a mask indicating the correct class
        mask = yt.bool()
        # get the values of the correct class
        output_class_indices = yp[mask].view(-1, 1)
        # compute all the differences
        output_diff = output_class_indices - yp
        # select all the values that are not the correct class
        output_diff = output_diff[~mask].view(batch_size, nb_classes - 1)
        if self.last_layer_type == "global":
            den = math.sqrt(2) * self.L
        elif self.last_layer_type == "classwise":
            den = 2 * self.L
        certs = output_diff / den
        # now we can compute the vra
        # vra is percentage of certs > eps
        certs = torch.nn.functional.relu(self.eps - certs).float()
        # the loss is the mean of the certificates weighted by the softmax of the output_diff
        return torch.sum(certs * torch.nn.functional.softmax(certs, dim=1), dim=1)

    def forward(self, yp, yt, gamma=None):
        if gamma is None:
            gamma = self.gamma
        yt = torch.nn.functional.one_hot(yt, yp.shape[1])
        return (
            -((1 - gamma) * torch.nn.functional.cosine_similarity(yp, yt))
            + (gamma * self.SoftVRA(yp, yt))
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
        target = torch.nn.functional.one_hot(target, num_classes=input.shape[1])
        if not (isinstance(input, torch.Tensor)):  # required for dtype.max
            input = torch.Tensor(input, dtype=input.dtype)
        if not (isinstance(target, torch.Tensor)):
            target = torch.Tensor(target, dtype=input.dtype)
        loss_batch = self.fct(target, input)
        return torch.mean(loss_batch)
