import argparse
import math
import time

import numpy as np
import schedulefree
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from tqdm import tqdm

from flashlipschitz.classparam import ClassParam
from flashlipschitz.layers import MaxMin
from flashlipschitz.layers import OrthoConv2d
from flashlipschitz.layers import OrthoLinear
from flashlipschitz.layers import ScaledAvgPool2d
from flashlipschitz.layers import SOC
from flashlipschitz.layers import UnitNormLinear
from flashlipschitz.layers.custom_activations import Abs
from flashlipschitz.layers.custom_activations import HouseHolder
from flashlipschitz.layers.custom_activations import HouseHolder_Order_2
from flashlipschitz.losses import Cosine_VRA_Loss
from flashlipschitz.losses import LossXent
from flashlipschitz.losses import SoftHKRMulticlassLoss
from flashlipschitz.models_factory import PatchBasedCNN
from flashlipschitz.models_factory import PatchBasedExapandedCNN

# from deel import torchlip as tl  ## copy pasted code from the lib to reduce dependencies

#### run this to reach 82% on cifar10 in less than 10 minutes on a single GPU
#### python train_convmixer.py --epochs=25 --batch-size=1024 --lr-max=1e-4 --ra-n=0 --ra-m=0 --wd=0. --scale=1.0 --jitter=0 --reprob=0 --conv-ks=5 --hdim=128 --gamma=0.1 --amp-enabled
#### python train_convmixer.py --epochs=50 --batch-size=1024 --lr-max=1e-4 --ra-n=2 --ra-m=10 --wd=0. --scale=1.0 --jitter=0 --reprob=0 --conv-ks=5 --hdim=128 --gamma=0.1 --amp-enabled --expand-factor=4

# increase gamma for better robustness (max 1.0)

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="ConvMixer")

parser.add_argument("--batch-size", default=512, type=int)
parser.add_argument("--scale", default=0.75, type=float)
parser.add_argument("--reprob", default=0.25, type=float)
parser.add_argument("--ra-m", default=8, type=int)
parser.add_argument("--ra-n", default=1, type=int)
parser.add_argument("--jitter", default=0.1, type=float)

parser.add_argument("--hdim", default=256, type=int)
parser.add_argument("--depth", default=8, type=int)
parser.add_argument("--stages", default=2, type=int)
parser.add_argument("--psize", default=2, type=int)
parser.add_argument("--conv-ks", default=5, type=int)
parser.add_argument("--expand-factor", default=2, type=int)
parser.add_argument("--bjorck-nbp-iters", default=0, type=int)
parser.add_argument("--bjorck-bp-iters", default=10, type=int)

parser.add_argument("--wd", default=1e-5, type=float)
parser.add_argument("--clip-norm", action="store_true")
parser.add_argument("--amp-enabled", action="store_true")
parser.add_argument("--epochs", default=25, type=int)
parser.add_argument("--lr-max", default=0.0005, type=float)
parser.add_argument("--workers", default=2, type=int)
parser.add_argument("--gamma", default=0.0, type=float)

args = parser.parse_args()


def VRA(output, class_indices, L=1.0, eps=36 / 255, return_certs=False):
    batch_size = output.shape[0]
    batch_indices = torch.arange(batch_size)

    onehot = torch.zeros_like(output).cuda()
    onehot[torch.arange(output.shape[0]), class_indices] = 1.0
    output_trunc = output - onehot * 1e6

    output_class_indices = output[batch_indices, class_indices]
    output_nextmax = torch.max(output_trunc, dim=1)[0]
    output_diff = output_class_indices - output_nextmax
    certs = output_diff / (math.sqrt(2) * L)
    # vra is percentage of certs > eps
    vra = (certs > eps).float()
    if return_certs:
        return certs
    return vra


# def BasicCNN(dim, depth, kernel_size=5, patch_size=2, expand_factor=2, n_classes=10):
#     return nn.Sequential(
#         FlashBCOP(
#             in_channels=3,
#             out_channels=dim,
#             kernel_size=patch_size,
#             stride=patch_size,
#             padding=0,
#             padding_mode="zeros",
#             bias=False,
#             pi_iters=3,
#             # exp_niter=5,
#             bjorck_bp_iters=args.bjorck_bp_iters,
#             bjorck_nbp_iters=args.bjorck_nbp_iters,
#         ),
#         # MaxMin(),
#         *[
#             # Residual(
#             nn.Sequential(
#                 FlashBCOP(
#                     in_channels=dim,
#                     out_channels=expand_factor * dim,
#                     kernel_size=kernel_size,
#                     padding="same",
#                     padding_mode="circular",
#                     bias=False,
#                     pi_iters=3,
#                     # exp_niter=5,
#                     bjorck_bp_iters=args.bjorck_bp_iters,
#                     bjorck_nbp_iters=args.bjorck_nbp_iters,
#                 ),
#                 # Oddly MaxMin works better than HouseHolder
#                 # also as these activations are pairwise
#                 # doubling the number of channels drastically improves
#                 # performances
#                 MaxMin(),
#                 # FlashBCOP(
#                 #     in_channels=expand_factor * dim,
#                 #     out_channels=dim,
#                 #     kernel_size=kernel_size,
#                 #     padding="same",
#                 #     padding_mode="circular",
#                 #     bias=False,
#                 #     pi_iters=3,
#                 #     # exp_niter=5,
#                 #     bjorck_bp_iters=args.bjorck_bp_iters,
#                 #     bjorck_nbp_iters=args.bjorck_nbp_iters,
#                 # ),
#                 # once we got back to dim don't add MaxMin
#             )
#             # )
#             for i in range(depth)
#         ],
#         ## I tried two kinds of pooling
#         ## L2NormPooling compute the norm of each channel
#         # tl.ScaledL2NormPool2d(
#         #     (32 // patch_size, 32 // patch_size),
#         #     None,
#         #     # k_coef_lip=1 / (32 // patch_size) ** 2,
#         # ),
#         ## scaledAvgPool2d is AvgPool2d but with a sqrt(w*h)
#         ## factor, as it would be 1/sqrt(w,h) lip otherwise
#         # nn.AvgPool2d(
#         #     32 // patch_size,
#         #     None,
#         #     divisor_override=32 // patch_size,
#         # ),
#         # nn.LPPool2d(2, kernel_size=32 // patch_size, stride=32 // patch_size),
#         nn.Flatten(),
#         UnitNormLinear(
#             dim * (32 // patch_size) ** 2,
#             n_classes,
#             # k_coef_lip=1 / (dim / n_classes),
#             bias=False,
#         ),
#     )


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
        transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        transforms.RandomErasing(p=args.reprob),
    ]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=train_transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
)


model = PatchBasedExapandedCNN(
    (3, 32, 32),
    n_classes=10,
    dim=args.hdim,
    depth=args.depth,
    groups=1,
    skip=True,
    conv=ClassParam(
        OrthoConv2d,
        bias=True,
        padding="same",
        padding_mode="circular",
        # padding_mode="zeros",
        # bjorck_params=BjorckParams(
        #     power_it_niter=3,
        #     eps=1e-6,
        #     bjorck_iters=15,
        #     beta=0.5,
        #     contiguous_optimization=False,
        # ),
    ),
    patch_size=args.psize,
    kernel_size=args.conv_ks,
    norm=None,
    expand_factor=args.expand_factor,
    pool=ClassParam(torch.nn.LPPool2d, norm_type=2, kernel_size=32 // args.psize),
    # pool=ClassParam(
    #     torch.nn.AvgPool2d,
    #     kernel_size=32 // args.psize,
    #     divisor_override=32 // args.psize,
    # ),
)

model = model.cuda()
summary(model, (args.batch_size, 3, 32, 32))
# model.compile()  ## TODO: make the modules compilable !!!!

# lr_schedule = lambda t: np.interp(
#     [t],
#     # [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs],
#     [0, 5, 15, args.epochs],
#     [1e-6, args.lr_max, args.lr_max / 20.0, 0],
# )[0]
# gamma_schedule = lambda t: np.interp(
#     [t],
#     # [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs],
#     [0, args.epochs],
#     [1e-5, args.gamma],
# )[0]
# use cosine lr schedule
# lr_schedule = (
#     lambda t: 1e-7 + args.lr_max * (1 + math.cos(math.pi * t / args.epochs)) / 2
# )


# opt = optim.AdamW(
#     model.parameters(), lr=args.lr_max, weight_decay=args.wd, betas=(0.9, 0.99)
# )
opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr_max)
std = torch.tensor(cifar10_std).cuda()
L = 1 / torch.max(std)
# criterion = Cosine_VRA_Loss(
#     args.gamma, L=L
# )  # here, I used the cosine (only accuracy, no robustness)

# criterion = SoftHKRMulticlassLoss(
#     alpha=0.75, min_margin=2*36 / 255, alpha_mean=0.99, temperature=2
# )
criterion = LossXent(10, offset=2 * 36 / 255, temperature=0.125)
# criterion = nn.CrossEntropyLoss()
if args.amp_enabled:
    scaler = torch.cuda.amp.GradScaler()


for epoch in range(args.epochs):
    start = time.time()
    train_loss, train_acc, n, train_vra = 0, 0, 0, 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for i, (X, y) in pbar:
        model.train()
        # opt.train()
        X, y = X.cuda(), y.cuda()

        # lr = lr_schedule(epoch + (i + 1) / len(trainloader))
        lr = args.lr_max
        # gamma = gamma_schedule(epoch + (i + 1) / len(trainloader))
        gamma = args.gamma
        opt.param_groups[0].update(lr=lr)

        opt.zero_grad()
        if args.amp_enabled:
            with torch.cuda.amp.autocast():
                output = model(X)
                certs = VRA(output, y, L=L, eps=36 / 255, return_certs=True)
                loss = criterion(output, y)
                # loss = criterion(output, nn.functional.one_hot(y, 10))
                # loss = (
                #     -(1.0 - gamma) * criterion(output, nn.functional.one_hot(y, 10))
                #     - gamma * torch.clamp(certs, min=0.0, max=36 / 255)
                # ).mean()

            scaler.scale(loss).backward()
            if args.clip_norm:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            output = model(X)
            certs = VRA(output, y, L=L, eps=36 / 255, return_certs=True)
            loss = criterion(output, y)
            # loss = (
            #     -(1.0 - gamma) * criterion(output, nn.functional.one_hot(y, 10))
            #     - gamma * torch.clamp(certs, min=0.0, max=36 / 255)
            # ).mean()
            loss.backward()
            if args.clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_vra += VRA(output, y, L=L, eps=36 / 255).sum().item()
        n += y.size(0)

    model.eval()
    # opt.eval()
    test_acc, test_vra, m = 0, 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()
            if args.amp_enabled:
                with torch.cuda.amp.autocast():
                    output = model(X)
            else:
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_vra += VRA(output, y, L=L, eps=36 / 255).sum().item()
            m += y.size(0)
    print(
        f"[{args.name}] Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Train VRA: {train_vra/n:.4f}, Test Acc: {test_acc/m:.4f}, Test VRA: {test_vra/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}, gamma: {gamma:.2f}"
    )
print("#" * 40)
print("training finished, computing singular values of each layer")


def print_sv(layer):
    if hasattr(layer, "singular_values"):
        sv_min, sv_max, stable_rank = layer.singular_values()
        print(
            f"{layer.__class__.__name__}:min={sv_min:.4f}:max={sv_max:.4f}:stab rank ratio={stable_rank:.4f}"
        )


model.apply(print_sv)

# print("saving model")
# # torch.save(
# #     model.state_dict(), f"{args.name}_acc_{test_acc/m:.2f}_vra_{test_vra/m:.2f}.pth"
# # )
# # save args
# with open(f"{args.name}_args.txt", "w") as f:
#     f.write(str(args))
