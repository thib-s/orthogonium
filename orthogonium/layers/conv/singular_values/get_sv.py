import torch
from .delattre24 import compute_delattre2023, compute_delattre2024


def get_conv_sv(
    layer,
    n_iter,
    agg_groups=True,
    device=None,
    detach=True,
    imsize=None,
):
    """
    Computes the Lipschitz constant of a convolution layer. This uses the layer
    parameters to decide the correct function to call depending on the padding
    mode, kernel shape, and stride.

    Parameters:
        layer (torch.nn.Module): Convolutional layer to compute the Lipschitz constant for.
            It must have a `weight` attribute. This function is expected to work
            only for layers in this library since striding is handled using the fact
            that our layers may use two subkernels.
        n_iter (int): Number of iterations for the Delattre algorithm.
        agg_groups (bool, optional): If True, the Lipschitz constant is computed
            for the entire layer. If False, the Lipschitz constant is computed
            for each group separately. Default is True.
        device (torch.device, optional): Device to use for the computation. Default is None.
        detach (bool, optional): If True, the result is detached from the computation graph.
            Default is True.
        imsize (int, optional): Size of the input image (required for circular padding).
            Default is None.
    """

    def _compute_grouped_lip(weight, stride, padding_mode, imsize, n_iter):
        """
        Internal helper: if groups>1, reshapes the kernel to (g, co/g, ci/g, kw, kh)
        and applies _compute_lip_subkernel to each 'slice' in the batch.

        Returns a single aggregated value (multiplied across groups) if agg_groups=True,
        else a list of per-group results.
        """

        g = layer.groups
        # Original shape is (co, ci // g, kw, kh).
        # Reshape to (g, co // g, ci // g, kw, kh).
        co, ci_over_g, kw, kh = weight.shape
        assert co % g == 0, "Number of output channels must be divisible by groups."

        weight_reshaped = weight.view(g, co // g, ci_over_g, kw, kh)

        # Compute lipschitz subkernel for each group slice
        group_results = []
        for i in range(g):
            w_i = weight_reshaped[i]
            lip_val = _compute_lip_subkernel(
                w_i,
                stride[0],
                stride[1],
                padding_mode,
                imsize,
                n_iter,
            )
            group_results.append(lip_val)

        if agg_groups:
            # Multiply them across groups to get a single value
            lip_prod = group_results[0]
            for i in range(1, g):
                lip_prod = lip_prod * group_results[i]
            return lip_prod
        else:
            # Return list of group-wise results without aggregation
            return group_results

    def _maybe_detach(res):
        """Helper to detach results if required."""
        if detach and hasattr(res, "detach"):
            return res.detach()
        elif detach and isinstance(res, (list, tuple)):
            return [val.detach() if hasattr(val, "detach") else val for val in res]
        return res

    # Main body of get_conv_sv
    #
    # 1) Check whether layer has one weight or (weight_1, weight_2).
    # 2) Handle groups>1 via _compute_grouped_lip.
    #
    if hasattr(layer, "weight_1"):
        # The layer has two separate weight tensors
        res_1 = _compute_grouped_lip(
            layer.weight_1.to(device),
            stride=(1, 1),  # presumably no stride for weight_1
            padding_mode=layer.padding_mode,
            imsize=imsize,
            n_iter=n_iter,
        )
        res_2 = _compute_grouped_lip(
            layer.weight_2.to(device),
            stride=(layer.stride[0], layer.stride[1]),
            padding_mode=layer.padding_mode,
            imsize=imsize,
            n_iter=n_iter,
        )
        combined = res_1 * res_2
        return _maybe_detach(combined)

    else:
        # The layer has a single weight
        # If groups>1, _compute_grouped_lip will handle the reshape/apply logic
        res = _compute_grouped_lip(
            layer.weight.to(device),
            stride=(layer.stride[0], layer.stride[1]),
            padding_mode=layer.padding_mode,
            imsize=imsize,
            n_iter=n_iter,
        )
        return _maybe_detach(res)


def _compute_lip_subkernel(kernel, s1, s2, padding_mode, imsize=None, n_iter=4):
    """
    Computes the Lipschitz constant of a subkernel, depending on padding mode and shape.
    """
    ci, co, kw, kh = kernel.shape

    if kw == s1 and kh == s2:
        # Possibly a special case: RKO approach
        sigma = _rko_lip_cste(kernel)
    elif padding_mode == "circular":
        sigma = compute_delattre2023(kernel, n=imsize, n_iter=n_iter)
    elif padding_mode == "zeros":
        sigma = compute_delattre2024(kernel, n_iter=n_iter)
    else:
        raise ValueError(f"Padding mode {padding_mode} not supported.")

    return sigma


def _rko_lip_cste(weight):
    """
    Simple fallback method to compute a Lipschitz constant by flattening
    spatial dims and taking spectral norm (largest singular value).
    """
    oc, ic, kh, kw = weight.shape
    return torch.linalg.norm(weight.reshape(oc, ic * kw * kh), ord=2)
