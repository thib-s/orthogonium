import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from flashlipschitz.layers.custom_activations import Abs
from flashlipschitz.layers.custom_activations import HouseHolder
from flashlipschitz.layers.custom_activations import HouseHolder_Order_2
from flashlipschitz.layers.custom_activations import MaxMin
from flashlipschitz.layers.fast_block_ortho_conv import BCOP
from flashlipschitz.layers.fast_block_ortho_conv import OrthoLinear
from flashlipschitz.layers.fast_block_ortho_conv import ScaledAvgPool2d

# from deel import torchlip as tl  ## copy pasted code from the lib to reduce dependencies

#### run this to reach 84% on cifar10 in less than 10 minutes on a single GPU
#### python train_convmixer.py --epochs=25 --batch-size=512 --lr-max=5e-4 --ra-n=0 --ra-m=0 --wd=0. --scale=1.0 --jitter=0 --reprob=0 --conv-ks=5 --hdim=128 --amp-enabled

#### run this to reach 86% and 64% VRA (each epoch is waaaaay much longer)
#### python train_convmixer.py --epochs=200 --batch-size=512 --lr-max=5e-4 --ra-n=2 --ra-m=12 --wd=0. --scale=1.0 --jitter=0 --reprob=0 --conv-ks=5 --hdim=512 --gamma=10.0 --amp-enabled

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


def BasicCNN(dim, depth, kernel_size=5, patch_size=2, expand_factor=2, n_classes=10):
    return nn.Sequential(
        BCOP(
            in_channels=3,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
            bias=False,
            pi_iters=3,
            bjorck_bp_iters=args.bjorck_bp_iters,
            bjorck_nbp_iters=args.bjorck_nbp_iters,
        ),
        # MaxMin(),
        *[
            nn.Sequential(
                BCOP(
                    in_channels=dim,
                    out_channels=expand_factor * dim,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=False,
                    pi_iters=3,
                    bjorck_bp_iters=args.bjorck_bp_iters,
                    bjorck_nbp_iters=args.bjorck_nbp_iters,
                    override_min_channels=expand_factor * dim,
                ),
                # Oddly MaxMin works better than HouseHolder
                # also as these activations are pairwise
                # doubling the number of channels drastically improves
                # performances
                MaxMin(),
                BCOP(
                    in_channels=expand_factor * dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=False,
                    pi_iters=3,
                    bjorck_bp_iters=args.bjorck_bp_iters,
                    bjorck_nbp_iters=args.bjorck_nbp_iters,
                    override_min_channels=expand_factor * dim,
                ),
                # once we got back to dim don't add MaxMin
            )
            for i in range(depth)
        ],
        ## I tried two kinds of pooling
        ## L2NormPooling compute the norm of each channel
        # tl.ScaledL2NormPool2d(
        #     (32 // patch_size, 32 // patch_size),
        #     None,
        #     # k_coef_lip=1 / (32 // patch_size) ** 2,
        # ),
        ## scaledAvgPool2d is AvgPool2d but with a sqrt(w*h)
        ## factor, as it would be 1/sqrt(w,h) lip otherwise
        ScaledAvgPool2d(
            (32 // patch_size, 32 // patch_size),
            None,
            # k_coef_lip=1 / (32 // patch_size) ** 2,
        ),
        nn.Flatten(),
        OrthoLinear(
            dim,
            n_classes,
            # k_coef_lip=1 / (dim / n_classes),
            bias=False,
        ),
    )


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


model = BasicCNN(
    args.hdim,
    args.depth,
    patch_size=args.psize,
    kernel_size=args.conv_ks,
    n_classes=10,
    expand_factor=args.expand_factor,
)

model = nn.DataParallel(model).cuda()
# model.compile()  ## TODO: make the modules compilable !!!!
summary(model, (3, 32, 32))

lr_schedule = lambda t: np.interp(
    [t],
    # [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs],
    [0, 5, 15, args.epochs],
    [0, args.lr_max, args.lr_max / 20.0, 0],
)[0]

opt = optim.AdamW(
    model.parameters(), lr=args.lr_max, weight_decay=args.wd, betas=(0.9, 0.99)
)
criterion = (
    nn.CosineSimilarity()
)  # here, I used the cosine (only accuracy, no robustness)
# criterion = nn.CrossEntropyLoss()
if args.amp_enabled:
    scaler = torch.cuda.amp.GradScaler()


for epoch in range(args.epochs):
    start = time.time()
    train_loss, train_acc, n, train_vra = 0, 0, 0, 0
    for i, (X, y) in enumerate(trainloader):
        model.train()
        X, y = X.cuda(), y.cuda()

        lr = lr_schedule(epoch + (i + 1) / len(trainloader))
        opt.param_groups[0].update(lr=lr)

        opt.zero_grad()
        if args.amp_enabled:
            with torch.cuda.amp.autocast():
                output = model(X)
                certs = VRA(output, y, L=1.0, eps=36 / 255, return_certs=True)
                # loss = criterion(output, y)
                loss = (
                    -criterion(output, nn.functional.one_hot(y, 10))
                    - args.gamma * torch.clamp(certs, min=0.0, max=36 / 255)
                ).mean()

            scaler.scale(loss).backward()
            if args.clip_norm:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            output = model(X)
            certs = VRA(output, y, L=1.0, eps=36 / 255, return_certs=True)
            # loss = criterion(output, y)
            loss = (
                -criterion(output, nn.functional.one_hot(y, 10))
                - args.gamma * torch.clamp(certs, min=0.0, max=36 / 255)
            ).mean()
            loss.backward()
            if args.clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_vra += VRA(output, y, L=1.0, eps=36 / 255).sum().item()
        n += y.size(0)

    model.eval()
    test_acc, test_vra, m = 0, 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_vra += VRA(output, y, L=1.0, eps=36 / 255).sum().item()
            m += y.size(0)
    print(
        f"[{args.name}] Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Test Acc: {test_acc/m:.4f}, Test VRA: {test_vra/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}"
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
