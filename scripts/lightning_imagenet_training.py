import os

import pytorch_lightning
import schedulefree
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageNet
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandAugment
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from flashlipschitz.losses import Cosine_VRA_Loss
from flashlipschitz.losses import CosineLoss
from flashlipschitz.models_factory import SplitConcatNet
from flashlipschitz.models_factory import SplitConcatNetConfigs

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")


this_directory = os.path.abspath(os.path.dirname(__file__))
parent_directory = os.path.abspath(os.path.join(this_directory, os.pardir))

MAX_EPOCHS = 90


class ImagenetDataModule(pytorch_lightning.LightningDataModule):
    # Dataset configuration
    _DATA_PATH = os.path.join(
        parent_directory, f"/local_data/imagenet_cache/ILSVRC/Data/CLS-LOC/"
    )
    _BATCH_SIZE = 128
    _NUM_WORKERS = 16  # Number of parallel processes fetching data
    _PREPROCESSING_PARAMS = {
        "img_mean": (0.41757566, 0.26098573, 0.25888634),
        "img_std": (0.21938758, 0.1983, 0.19342837),
        "crop_size": 224,
        "horizontal_flip_prob": 0.5,
        "randaug_params": {"magnitude": 7, "num_ops": 2},
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
                # RandAugment(**self._PREPROCESSING_PARAMS["randaug_params"]),
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
            prefetch_factor=2,
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


class ClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        # self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = SplitConcatNet(
            img_shape=(3, 224, 224),
            n_classes=num_classes,
            **SplitConcatNetConfigs["L"],
        )
        # self.criteria = Cosine_VRA_Loss(gamma=0.1, L=2 / 0.225, eps=36 / 255)
        self.criteria = CosineLoss()
        # self.automatic_optimization = False
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.opt.train()
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
            "loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "accuracy",
            self.train_acc,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # self.model.train()
        # self.opt.eval()
        # with torch.no_grad():
        #     for batch in itertools.islice(train_loader, 50):
        #         self.model(batch)
        self.model.eval()
        self.opt.eval()
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
            "val_accuracy",
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
        # optimizer = torch.optim.NAdam(self.parameters(), lr=1e-2, weight_decay=0)
        optimizer = schedulefree.AdamWScheduleFree(
            self.parameters(), lr=5e-3, weight_decay=1e-6
        )
        self.opt = optimizer
        return optimizer
        # return {
        #     "optimizer": optimizer,
        #     # "lr_scheduler": {
        #     #     "scheduler": torch.optim.lr_scheduler.StepLR(
        #     #         optimizer, step_size=MAX_EPOCHS // 3, gamma=0.2
        #     #     ),
        #     #     # "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
        #     #     #     optimizer, T_max=MAX_EPOCHS, eta_min=1e-8
        #     #     # ),
        #     #     "interval": "epoch",
        #     # },
        # }


def train():
    classification_module = ClassificationLightningModule(num_classes=1000)
    # classification_module = torch.compile(classification_module)
    data_module = ImagenetDataModule()
    wandb_logger = WandbLogger(project="lipschitz-imagenet", log_model=True)
    # wandb_logger.watch(classification_module)
    # wandb_logger.watch(classification_module, log="all")
    # wandb_logger.watch(classification_module, log="all", log_freq=1000)
    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",
        devices=2,  # GPUs per node
        num_nodes=1,  # Number of nodes
        # num_nodes=3,  # Number of nodes
        strategy="ddp",  # Distributed strategy
        precision="bf16-mixed",
        max_epochs=MAX_EPOCHS,
        enable_model_summary=True,
        # accumulate_grad_batches=2,
        # logger=[pytorch_lightning.loggers.TensorBoardLogger("logs/")],
        logger=[wandb_logger],
        default_root_dir="~/checkpoints/",
    )
    trainer.fit(classification_module, data_module)
    # save the model
    # torch.save(classification_module.model.state_dict(), "single_stage.pth")


if __name__ == "__main__":
    train()
