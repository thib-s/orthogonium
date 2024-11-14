from __future__ import print_function

import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchinfo import summary

from orthogonium.layers import MaxMin
from orthogonium.layers import OrthoConv2d
from orthogonium.layers.conv.AOC.bcop_x_rko_conv import BcopRkoConv2d
from orthogonium.layers.conv.AOC.bcop_x_rko_conv import BcopRkoConvTranspose2d

# from orthogonium.layers import LayerCentering

# from orthogonium.layers import OrthoLinear
# from orthogonium.layers import ScaledAvgPool2d
# from orthogonium.layers import SOC
# from orthogonium.layers import UnitNormLinear
# from orthogonium.layers.custom_activations import Abs
# from orthogonium.layers.custom_activations import HouseHolder
# from orthogonium.layers.custom_activations import HouseHolder_Order_2

cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")

bs = 512
# set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# loading the dataset
dataset = dset.ImageFolder(
    # dataset = dset.CIFAR10(
    # root="/local_data/imagenet_cache/ILSVRC/Data/CLS-LOC/train/",
    root="/datasets/shared_datasets/imagenette/imagenette2-160/train/",
    # root="/mnt/deel/datasets/shared_datasets/imagenet/ILSVRC/Data/CLS-LOC/train/",
    # root="./data",
    # split="unlabeled",
    # download=True,
    # )
    transform=transforms.Compose(
        [
            # transforms.Resize(64),
            transforms.Resize(64),
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        ]
    ),
)
nc = 3

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=bs,
    shuffle=True,
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
)

# checking the availability of cuda devices
device = "cuda" if torch.cuda.is_available() else "cpu"

# number of gpu's available
ngpu = 1
# input noise dimension
nz = 128
# number of generator filters
ngf = 64
# number of discriminator filters
ndf = 64

ks = 5


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    # if classname.find("Conv") != -1:
    #     m.weight.data.normal_(0.0, 0.02)
    # elif classname.find("BatchNorm") != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.add_module("fn", fn)

    def forward(self, x):
        # split x
        # x1, x2 = x.chunk(2, dim=1)
        # apply function
        out = self.fn(x)
        # concat and return
        # return torch.cat([x1, out], dim=1)
        return (x + out) * 0.5


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            BcopRkoConvTranspose2d(
                nz,
                ngf * 16,
                4,
                2,
                padding=(1, 1),
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 16),
            # nn.ReLU(True),
            Residual(
                nn.Sequential(
                    BcopRkoConvTranspose2d(
                        ngf * 16,
                        ngf * 16,
                        ks,
                        1,
                        padding=(ks // 2, ks // 2),
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ngf * 16),
                    nn.ReLU(True),
                )
            ),
            # state size. (ngf*8) x 4 x 4
            BcopRkoConvTranspose2d(
                ngf * 16,
                ngf * 8,
                4,
                2,
                padding=(1, 1),
                output_padding=0,
                bias=False,
            ),
            Residual(
                nn.Sequential(
                    BcopRkoConvTranspose2d(
                        ngf * 8,
                        ngf * 8,
                        ks,
                        1,
                        padding=(ks // 2, ks // 2),
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    BcopRkoConvTranspose2d(
                        ngf * 8,
                        ngf * 8,
                        ks,
                        1,
                        padding=(ks // 2, ks // 2),
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                )
            ),
            # state size. (ngf*8) x  x 4
            BcopRkoConvTranspose2d(
                ngf * 8,
                ngf * 4,
                4,
                2,
                padding=(1, 1),
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            Residual(
                nn.Sequential(
                    BcopRkoConvTranspose2d(
                        ngf * 4,
                        ngf * 4,
                        ks,
                        1,
                        padding=(ks // 2, ks // 2),
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    BcopRkoConvTranspose2d(
                        ngf * 4,
                        ngf * 4,
                        ks,
                        1,
                        padding=(ks // 2, ks // 2),
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                )
            ),
            # state size. (ngf*4) x 8 x 8
            BcopRkoConvTranspose2d(
                ngf * 4,
                ngf * 2,
                4,
                2,
                padding=(1, 1),
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(True),
            Residual(
                nn.Sequential(
                    BcopRkoConvTranspose2d(
                        ngf * 2,
                        ngf * 2,
                        ks,
                        1,
                        padding=(ks // 2, ks // 2),
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    BcopRkoConvTranspose2d(
                        ngf * 2,
                        ngf * 2,
                        ks,
                        1,
                        padding=(ks // 2, ks // 2),
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                )
            ),
            # state size. (ngf*2) x 16 x 16
            BcopRkoConvTranspose2d(
                ngf * 2,
                ngf,
                4,
                2,
                padding=(1, 1),
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            Residual(
                nn.Sequential(
                    BcopRkoConvTranspose2d(
                        ngf,
                        ngf,
                        ks,
                        1,
                        padding=(ks // 2, ks // 2),
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    BcopRkoConvTranspose2d(
                        ngf,
                        ngf,
                        ks,
                        1,
                        padding=(ks // 2, ks // 2),
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                )
            ),
            # state size. (ngf) x 32 x 32
            BcopRkoConvTranspose2d(
                ngf,
                ngf,
                4,
                2,
                padding=(1, 1),
                output_padding=0,
                bias=False,
            ),
            Residual(
                nn.Sequential(
                    BcopRkoConvTranspose2d(
                        ngf,
                        ngf,
                        ks,
                        1,
                        padding=(ks // 2, ks // 2),
                        output_padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                )
            ),
            BcopRkoConvTranspose2d(
                ngf, nc, ks, 1, padding=(ks // 2, ks // 2), output_padding=0, bias=False
            ),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
# load weights to test the model
# netG.load_state_dict(torch.load('weights/netG_epoch_24.pth'))
summary(netG, (1024, nz, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            BcopRkoConv2d(
                3,
                ndf,
                kernel_size=ks,
                stride=1,
                padding=(ks // 2, ks // 2),
                padding_mode="circular",
                bias=True,
            ),
            # LayerCentering(),
            MaxMin(),
            BcopRkoConv2d(
                ndf,
                ndf * 2,
                kernel_size=ks,
                stride=2,
                padding=(ks // 2, ks // 2),
                padding_mode="circular",
                bias=True,
            ),
            # LayerCentering(),
            MaxMin(),
            BcopRkoConv2d(
                ndf * 2,
                ndf * 2,
                kernel_size=ks,
                stride=1,
                padding=(ks // 2, ks // 2),
                padding_mode="circular",
                bias=True,
            ),
            # LayerCentering(),
            MaxMin(),
            # state size. (ndf) x 32 x 32
            BcopRkoConv2d(
                ndf * 2,
                ndf * 4,
                kernel_size=ks,
                stride=2,
                padding=(ks // 2, ks // 2),
                padding_mode="circular",
                bias=True,
            ),
            # LayerCentering(),
            # MaxMin(),
            BcopRkoConv2d(
                ndf * 4,
                ndf * 4,
                kernel_size=ks,
                stride=1,
                padding=(ks // 2, ks // 2),
                padding_mode="circular",
                bias=False,
            ),
            # LayerCentering(),
            MaxMin(),
            # state size. (ndf*2) x 16 x 16
            BcopRkoConv2d(
                ndf * 4,
                ndf * 8,
                kernel_size=ks,
                stride=2,
                padding=(ks // 2, ks // 2),
                padding_mode="circular",
                bias=True,
            ),
            # LayerCentering(),
            MaxMin(),
            BcopRkoConv2d(
                ndf * 8,
                ndf * 8,
                kernel_size=ks,
                stride=1,
                padding=(ks // 2, ks // 2),
                padding_mode="circular",
                bias=True,
            ),
            # LayerCentering(),
            MaxMin(),
            # state size. (ndf*4) x 8 x 8
            BcopRkoConv2d(
                ndf * 8,
                ndf * 16,
                kernel_size=ks,
                stride=2,
                padding=(ks // 2, ks // 2),
                padding_mode="circular",
                bias=True,
            ),
            # LayerCentering(),
            MaxMin(),
            BcopRkoConv2d(
                ndf * 16,
                ndf * 16,
                kernel_size=ks,
                stride=1,
                padding=(ks // 2, ks // 2),
                padding_mode="circular",
                bias=True,
            ),
            # LayerCentering(),
            MaxMin(),
            # state size. (ndf*8) x 4 x 4
            OrthoConv2d(
                ndf * 16,
                1,
                kernel_size=4,
                stride=4,
                padding=(0, 0),
                # padding_mode="circular",
                bias=False,
            ),
            nn.Flatten(),
            # UnitNormLinear(4 * 4, 1),
            # nn.Sigmoid(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
        # return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
# load weights to test the model
# netD.load_state_dict(torch.load('weights/netD_epoch_24.pth'))
# print(netD)
summary(netD, (1024, 3, 64, 64))


# criterion = nn.BCELoss()
# use KR criterion
def criterion(output, label):
    kr = torch.mean(output * label) - torch.mean(output * (1 - label))
    hinge = torch.mean(torch.nn.functional.relu(0.1 + output * (label * 2 - 1)))
    return 0.5 * kr + 0.5 * hinge
    # return hkr_loss(output, label, alpha=0, min_margin=0, true_values=(1, 0))


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
# optimizerD = schedulefree.AdamWScheduleFree(
#     netD.parameters(), lr=0.0005, weight_decay=0.0
# )
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
# optimizerG = schedulefree.AdamWScheduleFree(
#     netG.parameters(), lr=0.0005, weight_decay=0.0
# )

fixed_noise = torch.randn(128, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

niter = 100
g_loss = []
d_loss = []
for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full(
            (batch_size,), real_label, dtype=real_cpu.dtype, device=device
        )

        output = netD(
            real_cpu
        )  # + torch.normal(0, 1e-3, size=real_cpu.size()).to(device))
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(
            fake.detach()  # + torch.normal(0, 1e-3, size=fake.size()).to(device)
        )
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()

        fake = netG(noise)
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
        optimizerG.step()
        print(
            "[%d/%d][%d/%d] Loss_G: %.4f Loss_D: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
            % (
                epoch,
                niter,
                i,
                len(dataloader),
                errG.item(),
                errD.item(),
                D_x,
                D_G_z1,
                D_G_z2,
            )
        )

        # save the output
        if i % 100 == 1:
            print("saving the output")
            vutils.save_image(real_cpu[:128], "output/real_samples.png", normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(
                fake.detach(),
                "output/fake_samples_epoch_%03d.png" % (epoch),
                normalize=True,
            )

    # Check pointing for every epoch
    # torch.save(netG.state_dict(), "weights/netG_epoch_%d.pth" % (epoch))
    # torch.save(netD.state_dict(), "weights/netD_epoch_%d.pth" % (epoch))
