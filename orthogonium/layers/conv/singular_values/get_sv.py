import torch
from .delattre24 import compute_delattre2023, compute_delattre2024


def get_conv_sv(
    layer,
    n_iter,
    agg_groups=True,
    return_stab_rank=True,
    device=None,
    detach=True,
    imsize=None,
):
    """
    Computes Lipschitz constant (and optional 'stability rank') of a convolution layer. This use the layer paramaters
    to decide the correct function to call depending the the padding mode, the shape of the kernel and the stride.
    Under the hood it uses the methods described in [1] and [2].

    Parameters:
        layer (torch.nn.Module): Convolutional layer to compute the Lipschitz constant for. It must have a weight
            attribute. This function is expected to work only for layer in this library as striding is handled using
            the fact that our layers uses two subkernels in this situation.
        n_iter (int): Number of iterations for the Delattre algorithm.
        agg_groups (bool, optional): If True, the Lipschitz constant is computed for the entire layer. If False, the
            Lipschitz constant is computed for each group separately. Default is True.
        return_stab_rank (bool, optional): If True, the function also returns the 'stability rank' of the layer
            (normalized to be a fraction of the full rank). When agg_groups = True the average stable rank over groups
            is returned. Default is True.
        device (torch.device, optional): Device to use for the computation. Default is None.
        detach (bool, optional): If True, the result is detached from the computation graph. Default is True.
        imsize (int, optional): Size of the input image. Required for circular padding. Default is None.

    Returns:
        float or tuple: Lipschitz constant of the layer. If return_stab_rank is True, the function returns a tuple

    Warnings:
        There is currently an issue when estimating the lipschitz constant of a layer with circular padding and
        asymmetric padding (ie. even kernel size and no stride). The function may return a lipschitz constant lower
        than the actual value.

    References:
        - [1] Delattre, B., Barthélemy, Q., Araujo, A., & Allauzen, A. (2023, July).
        Efficient bound of Lipschitz constant for convolutional layers by gram iteration.
        In International Conference on Machine Learning (pp. 7513-7532). PMLR.
        <https://arxiv.org/abs/2305.16173>
        - [2] Delattre, B., Barthélemy, Q., & Allauzen, A. (2024).
        Spectral Norm of Convolutional Layers with Circular and Zero Paddings.
        <https://arxiv.org/abs/2402.00240>
    """

    def _compute_grouped_lip(
        weight, stride, padding_mode, return_stab_rank, imsize, n_iter
    ):
        """
        Internal helper: if groups>1, reshapes the kernel to (g, co/g, ci/g, kw, kh)
        and applies _compute_lip_subkernel to each 'slice' in the batch.

        Returns either a list of per-group results or a single aggregated value
        (e.g., multiplied across groups) based on agg_groups.
        """

        g = layer.groups
        # Original shape is (co, ci // g, kw, kh).
        # Reshape to (g, co // g, ci // g, kw, kh).
        co, ci_over_g, kw, kh = weight.shape
        # sanity checks: co should be multiple of g, etc.
        assert co % g == 0, "Number of output channels must be divisible by groups."

        weight_reshaped = weight.view(g, co // g, ci_over_g, kw, kh)

        # Compute lip subkernel for each group slice
        group_results = []
        for i in range(g):
            w_i = weight_reshaped[i]
            lip_val = _compute_lip_subkernel(
                w_i,
                stride[0],
                stride[1],
                padding_mode,
                return_stab_rank,
                imsize,
                n_iter,
            )
            group_results.append(lip_val)

        # If we want to aggregate across all groups, define how to combine them
        if agg_groups:
            if return_stab_rank:
                # In this scenario, each lip_val is a tuple: (lip_factor, stab_rank)
                lip_prod = group_results[0][0]
                stab_prod = group_results[0][1] / g
                for i in range(1, g):
                    lip_prod = lip_prod * group_results[i][0]
                    stab_prod = stab_prod + group_results[i][1] / g
                return (lip_prod, stab_prod)
            else:
                # If we only compute one scalar per group, multiply them
                lip_prod = group_results[0]
                for i in range(1, g):
                    lip_prod = lip_prod * group_results[i]
                return lip_prod
        else:
            # Return list of group-wise results without aggregation
            return group_results

    def _maybe_detach(res):
        """
        Helper to detach results if required.
        Return type is preserved (scalar vs tuple).
        """
        if detach:
            if return_stab_rank and isinstance(res, tuple):
                return tuple([r.detach() for r in res])
            elif isinstance(res, (list, tuple)):
                return [r.detach() if hasattr(r, "detach") else r for r in res]
            else:
                return res.detach() if hasattr(res, "detach") else res
        return res

    #
    # Main body of get_lipschitz_conv
    #
    # 1) Check whether layer has one weight or (weight_1, weight_2).
    # 2) Check groups>1 and handle accordingly.
    #
    if hasattr(layer, "weight_1"):
        # The layer has two separate weight tensors
        res_1 = _compute_grouped_lip(
            layer.weight_1.to(device),
            stride=(1, 1),  # presumably no stride for weight_1
            padding_mode=layer.padding_mode,
            return_stab_rank=return_stab_rank,
            imsize=imsize,
            n_iter=n_iter,
        )
        res_2 = _compute_grouped_lip(
            layer.weight_2.to(device),
            stride=(layer.stride[0], layer.stride[1]),
            padding_mode=layer.padding_mode,
            return_stab_rank=return_stab_rank,
            imsize=imsize,
            n_iter=n_iter,
        )

        # Combine the results from res_1 and res_2
        if return_stab_rank:
            # Each is a tuple: (lip_factor, stab_rank)
            combined = (res_1[0] * res_2[0], 0.5 * res_1[1] + 0.5 * res_2[1])
        else:
            combined = res_1 * res_2

        return _maybe_detach(combined)
    else:
        # The layer has a single weight
        # If groups>1, _compute_grouped_lip will handle the reshape/apply logic
        res = _compute_grouped_lip(
            layer.weight.to(device),
            stride=(layer.stride[0], layer.stride[1]),
            padding_mode=layer.padding_mode,
            return_stab_rank=return_stab_rank,
            imsize=imsize,
            n_iter=n_iter,
        )
        return _maybe_detach(res)


def _compute_lip_subkernel(
    kernel, s1, s2, padding_mode, return_stab_rank=True, imsize=None, n_iter=4
):
    ci, co, kw, kh = kernel.shape
    if kw == s1 and kh == s2:
        sigma = _rko_lip_cste(kernel)
    elif padding_mode == "circular":
        sigma = compute_delattre2023(kernel, n=imsize, n_iter=n_iter)
    elif padding_mode == "zeros":
        sigma = compute_delattre2024(kernel, n_iter=n_iter)
    else:
        raise ValueError(f"Padding mode {padding_mode} not supported.")
    if return_stab_rank:
        fro_norm = torch.sum(torch.square(kernel))
        stab_rank = fro_norm / (sigma**2 * min(ci, co * (s1 * s2)))
        return sigma, stab_rank
    return sigma


def _rko_lip_cste(weight):
    oc, ic, kh, kw = weight.shape
    return torch.linalg.norm(
        weight.reshape(oc, ic * kw * kh),
        ord=2,
    )
