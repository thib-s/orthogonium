import os

import pytorch_lightning
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageNet
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from flashlipschitz.layers.custom_activations import MaxMin
from flashlipschitz.layers.conv.fast_block_ortho_conv import FlashBCOP
from flashlipschitz.layers.normalization import LayerCentering
from flashlipschitz.layers.pooling import ScaledAvgPool2d
from flashlipschitz.layers.conv.rko_conv import OrthoLinear
from flashlipschitz.layers.conv.rko_conv import UnitNormLinear
from flashlipschitz.losses import Cosine_VRA_Loss


this_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(this_directory, os.pardir))

MAX_EPOCHS = 25


class ImagenetDataModule(pytorch_lightning.LightningDataModule):
    # Dataset configuration
    _DATA_PATH = os.path.join(
        parent_directory, f"/local_data/imagenet_cache/ILSVRC/Data/CLS-LOC/"
    )
    _BATCH_SIZE = 512
    _NUM_WORKERS = 16  # Number of parallel processes fetching data
    _PREPROCESSING_PARAMS = {
        "img_mean": (0.41757566, 0.26098573, 0.25888634),
        "img_std": (0.21938758, 0.1983, 0.19342837),
        "crop_size": 224,
        "horizontal_flip_prob": 0.5,
        # "randaug_params": {"magnitude": 6, "num_layers": 2, "prob": 0.5},
        "random_resized_crop_params": {
            "scale": (0.08, 1.0),
            "ratio": (3.0 / 4.0, 4.0 / 3.0),
        },
    }

    def train_dataloader(self):
        # Define the transformations
        transform = Compose(
            [
                Resize(256),
                RandomResizedCrop(
                    self._PREPROCESSING_PARAMS["crop_size"],
                    **self._PREPROCESSING_PARAMS["random_resized_crop_params"],
                ),
                RandomHorizontalFlip(
                    self._PREPROCESSING_PARAMS["horizontal_flip_prob"]
                ),
                ToTensor(),
                Normalize(
                    mean=self._PREPROCESSING_PARAMS["img_mean"],
                    std=self._PREPROCESSING_PARAMS["img_std"],
                ),
            ]
        )

        # Load the dataset
        train_dataset = ImageFolder(
            self._DATA_PATH + "train",
            transform=transform,
        )

        return DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            # prefetch_factor=2,
            shuffle=True,
        )

    def val_dataloader(self):
        # Define the transformations
        transform = Compose(
            [
                Resize(256),
                CenterCrop(self._PREPROCESSING_PARAMS["crop_size"]),
                ToTensor(),
                Normalize(
                    mean=self._PREPROCESSING_PARAMS["img_mean"],
                    std=self._PREPROCESSING_PARAMS["img_std"],
                ),
            ]
        )

        # Load the dataset
        val_dataset = ImageFolder(
            self._DATA_PATH + "val",
            transform=transform,
        )

        return DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )


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


def BasicCNN(img_size, dim, depth, kernel_size, patch_size, expand_factor, n_classes):
    return nn.Sequential(
        FlashBCOP(
            in_channels=3,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
            bias=False,
            # pi_iters=3,
            # exp_niter=5,
            bjorck_bp_iters=10,
            bjorck_nbp_iters=0,
        ),
        # MaxMin(),
        *[
            Residual(
                nn.Sequential(
                    FlashBCOP(
                        in_channels=dim,
                        out_channels=expand_factor * dim,
                        kernel_size=kernel_size,
                        padding="same",
                        padding_mode="zeros",
                        bias=False,
                        pi_iters=3,
                        # exp_niter=5,
                        bjorck_bp_iters=10,
                        bjorck_nbp_iters=0,
                    ),
                    # Oddly MaxMin works better than HouseHolder
                    # also as these activations are pairwise
                    # doubling the number of channels drastically improves
                    # performances
                    LayerCentering(),
                    MaxMin(),
                    FlashBCOP(
                        in_channels=expand_factor * dim,
                        out_channels=dim,
                        kernel_size=kernel_size,
                        padding="same",
                        padding_mode="zeros",
                        bias=False,
                        pi_iters=3,
                        # exp_niter=5,
                        bjorck_bp_iters=10,
                        bjorck_nbp_iters=0,
                    ),
                    # once we got back to dim don't add MaxMin
                )
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
        LayerCentering(),
        nn.AvgPool2d(
            img_size // patch_size,
            None,
            divisor_override=img_size // patch_size,
        ),
        nn.Flatten(),
        UnitNormLinear(
            dim,
            n_classes,
            # k_coef_lip=1 / (dim / n_classes),
            bias=False,
        ),
    )


class ClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.model = BasicCNN(
            img_size=224,
            dim=512,
            depth=8,
            kernel_size=5,
            patch_size=16,
            expand_factor=2,
            n_classes=num_classes,
        )
        self.criteria = Cosine_VRA_Loss(gamma=0.0, L=2 / 0.225, eps=36 / 255)
        # self.automatic_optimization = False
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.preds_buffer = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, label = batch
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(img)
        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = self.criteria(y_hat, label)
        # label = label.argmax(dim=-1)
        self.train_acc(y_hat, label)
        # Log the train loss to Tensorboard
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        y_hat = self.model(img)
        loss = self.criteria(y_hat, label)
        # label = label.argmax(dim=-1)
        self.val_acc(y_hat, label)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            self.val_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        # return torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=1e-5)
        optimizer = torch.optim.NAdam(self.parameters(), lr=1e-5, weight_decay=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=MAX_EPOCHS // 3, gamma=0.2
                ),
                # "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                #     optimizer, T_max=MAX_EPOCHS, eta_min=1e-8
                # ),
                "interval": "epoch",
            },
        }


def train():
    classification_module = ClassificationLightningModule(num_classes=1000)
    # classification_module = torch.compile(classification_module)
    data_module = ImagenetDataModule()
    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",
        devices=2,  # GPUs per node
        num_nodes=4,  # Number of nodes
        strategy="ddp",  # Distributed strategy
        precision="bf16-mixed",
        max_epochs=MAX_EPOCHS,
        enable_model_summary=True,
        # logger=[pytorch_lightning.loggers.TensorBoardLogger("logs/")],
    )
    trainer.fit(classification_module, data_module)
    # save the model
    torch.save(classification_module.model.state_dict(), "single_stage.pth")


if __name__ == "__main__":
    train()
