import torch
import torch.nn as nn


class GroupMix(nn.Module):
    def __init__(self, group_in, group_out, dim=1):
        """GroupMix layer. this layer is used to mix the groups of the input tensor.
        Read it this way: groupmix take groups of size group_in (size / group_out groups in total)
        and mix them to groups of size group_out (size / group_in groups in total).
        """
        super(GroupMix, self).__init__()
        assert dim == 1, "GroupMix layer only supports dim=1 for the moment"
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


if __name__ == "__main__":
    x = torch.randn(2, 6)
    # takes two groups of size 3, and return 3 groups of size 2
    gm = GroupMix(2, 3)
    print(f"in: {x}")
    y = gm(x)
    print(f"out: {y}")
    x2 = torch.randn(2, 6, 32, 32)
    gm = GroupMix(2, 3)
    y2 = gm(x2)
    print(f"in shape: {x2.shape}, out shape: {y2.shape}")
    gp = GroupMix(3, 2)
    x2b = gp(y2)
    assert torch.allclose(x2, x2b), "GroupMix is not invertible"
    print("GroupMix is invertible")
