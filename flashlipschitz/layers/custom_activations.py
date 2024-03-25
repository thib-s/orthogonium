import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, z):
        return torch.abs(z)


class MaxMin(nn.Module):
    def __init__(self, axis=1):
        self.axis = axis
        super(MaxMin, self).__init__()

    def forward(self, z):
        a, b = z.split(z.shape[self.axis] // 2, self.axis)
        c, d = torch.max(a, b), torch.min(a, b)
        return torch.cat([c, d], dim=self.axis)


class HouseHolder(nn.Module):
    def __init__(self, channels):
        super(HouseHolder, self).__init__()
        assert (channels % 2) == 0
        eff_channels = channels // 2

        self.theta = nn.Parameter(
            0.5 * np.pi * torch.ones(1, eff_channels, 1, 1).cuda(), requires_grad=True
        )

    def forward(self, z, axis=1):
        theta = self.theta
        x, y = z.split(z.shape[axis] // 2, axis)

        selector = (x * torch.sin(0.5 * theta)) - (y * torch.cos(0.5 * theta))

        a_2 = x * torch.cos(theta) + y * torch.sin(theta)
        b_2 = x * torch.sin(theta) - y * torch.cos(theta)

        a = x * (selector <= 0) + a_2 * (selector > 0)
        b = y * (selector <= 0) + b_2 * (selector > 0)
        return torch.cat([a, b], dim=axis)


class HouseHolder_Order_2(nn.Module):
    def __init__(self, channels):
        super(HouseHolder_Order_2, self).__init__()
        assert (channels % 2) == 0
        self.num_groups = channels // 2

        self.theta0 = nn.Parameter(
            (np.pi * torch.rand(self.num_groups)).cuda(), requires_grad=True
        )
        self.theta1 = nn.Parameter(
            (np.pi * torch.rand(self.num_groups)).cuda(), requires_grad=True
        )
        self.theta2 = nn.Parameter(
            (np.pi * torch.rand(self.num_groups)).cuda(), requires_grad=True
        )

    def forward(self, z, axis=1):
        theta0 = torch.clamp(self.theta0.view(1, -1, 1, 1), 0.0, 2 * np.pi)

        x, y = z.split(z.shape[axis] // 2, axis)
        z_theta = (torch.atan2(y, x) - (0.5 * theta0)) % (2 * np.pi)

        theta1 = torch.clamp(self.theta1.view(1, -1, 1, 1), 0.0, 2 * np.pi)
        theta2 = torch.clamp(self.theta2.view(1, -1, 1, 1), 0.0, 2 * np.pi)
        theta3 = 2 * np.pi - theta1
        theta4 = 2 * np.pi - theta2

        ang1 = 0.5 * (theta1)
        ang2 = 0.5 * (theta1 + theta2)
        ang3 = 0.5 * (theta1 + theta2 + theta3)
        ang4 = 0.5 * (theta1 + theta2 + theta3 + theta4)

        select1 = torch.logical_and(z_theta >= 0, z_theta < ang1)
        select2 = torch.logical_and(z_theta >= ang1, z_theta < ang2)
        select3 = torch.logical_and(z_theta >= ang2, z_theta < ang3)
        select4 = torch.logical_and(z_theta >= ang3, z_theta < ang4)

        a1 = x
        b1 = y

        a2 = x * torch.cos(theta0 + theta1) + y * torch.sin(theta0 + theta1)
        b2 = x * torch.sin(theta0 + theta1) - y * torch.cos(theta0 + theta1)

        a3 = x * torch.cos(theta2) + y * torch.sin(theta2)
        b3 = -x * torch.sin(theta2) + y * torch.cos(theta2)

        a4 = x * torch.cos(theta0) + y * torch.sin(theta0)
        b4 = x * torch.sin(theta0) - y * torch.cos(theta0)

        a = (a1 * select1) + (a2 * select2) + (a3 * select3) + (a4 * select4)
        b = (b1 * select1) + (b2 * select2) + (b3 * select3) + (b4 * select4)

        z = torch.cat([a, b], dim=axis)
        return z
