import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
    def __init__(self, group_in, group_out, dim=1):
        """ChannelShuffle layer. this layer is used to mix the groups of the input tensor.
        Read it this way: ChannelShuffle take groups of size group_in (size / group_out groups in total)
        and mix them to groups of size group_out (size / group_in groups in total).
        """
        super(ChannelShuffle, self).__init__()
        assert dim == 1, "ChannelShuffle layer only supports dim=1 for the moment"
        self.dim = dim
        self.group_in = group_in
        self.group_out = group_out

    def forward(self, x):
        assert (
            x.size(self.dim) == self.group_in * self.group_out
        ), "input tensor size must be equal to group_in * group_out"
        return torch.concat(
            [x[:, i :: self.group_out] for i in range(self.group_out)], dim=1
        ).contiguous()

    def extra_repr(self):
        return f"group_in={self.group_in}, group_out={self.group_out}"
