import gc
import os

import pandas as pd
import pytorch_lightning
import torch
from batch_times import evaluate_all_model_time_statistics
from memory_usage import get_model_memory
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from orthogonium.classparam import ClassParam
from orthogonium.layers import OrthoConv2d as BCOP_new
from orthogonium.layers.legacy.block_ortho_conv import BCOP as BCOP_old
from orthogonium.layers.legacy.cayley_ortho_conv import Cayley
from orthogonium.layers.legacy.skew_ortho_conv import SOC
from orthogonium.models_factory import LipResNet

# from orthogonium.layers.conv.reparametrizers import BjorckParams

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

layers = [
    ("BCOP_new", BCOP_new),
    ("BCOP_old", BCOP_old),
    (
        "SOC",
        lambda in_channels, out_channels, kernel_size=3, stride=1, padding=None, padding_mode="zeros", bias=True: SOC(
            in_channels, out_channels, kernel_size, stride, padding, bias
        ),
    ),
    (
        "Cayley",
        lambda in_channels, out_channels, kernel_size=3, stride=1, padding=None, padding_mode="zeros", bias=True: Cayley(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
    ),
    ("Conv2D", Conv2d),
]


class ImagenetDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, batch_size=512):
        super().__init__()
        self._BATCH_SIZE = batch_size

    # Dataset configuration
    _DATA_PATH = os.path.join(f"/local_data/imagenet_cache/ILSVRC/Data/CLS-LOC/")
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


res = []

for batch_size in [32, 64, 128, 256, 512]:
    data_module = ImagenetDataModule(batch_size=batch_size)
    # data_module.prepare_data()
    # data_module.setup()
    for layer_name, layer_cls in layers:
        try:
            config = {}
            config["conv"] = ClassParam(
                layer_cls,
                bias=False,
                padding="same",
                padding_mode="zeros",
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()
            gc.collect()
            torch.cuda.synchronize()
            res_1 = get_model_memory(
                lambda: LipResNet(
                    img_shape=(3, 224, 224),
                    n_classes=1000,
                    **config,
                ),
                test_loader=data_module.val_dataloader(),
                train_loader=data_module.train_dataloader(),
                logging=print,
            )

            conv_layer = LipResNet(
                img_shape=(3, 224, 224),
                n_classes=1000,
                **config,
            )
            conv_layer.to(device)
            # summary(conv_layer, (1, 3, 224, 224))
            conv_layer.train()
            res_1.update(
                evaluate_all_model_time_statistics(
                    conv_layer,
                    train_loader=data_module.train_dataloader(),
                    test_loader=data_module.val_dataloader(),
                    nrof_batches=31,
                    log=print,
                )
            )
            print(f"{layer_name}")
            print("\n".join([f"{k}: {v}" for k, v in res_1.items()]))
            res_1["conv_type"] = layer_name
            res_1["batch_size"] = batch_size
            res.append(res_1)

            torch.cuda.empty_cache()
            # clear memory
            del conv_layer
            del res_1
        except RuntimeError as e:
            print(f"Out of memory for {layer_name} and batch size 512")
            # remove the layer from the list
            layers = [l for l in layers if l[0] != layer_name]
            torch.cuda.empty_cache()

pd.DataFrame.from_records(res).to_csv("result_bs.csv")
