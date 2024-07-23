import gc
import os
import time
from timeit import timeit

import pandas as pd
import pytorch_lightning
import torch
from batch_times import evaluate_all_model_time_statistics
from memory_usage import get_model_memory
from torch.nn import Conv2d
from torch.profiler import profile
from torch.profiler import ProfilerActivity
from torch.profiler import record_function
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchinfo import summary
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

from flashlipschitz.classparam import ClassParam
from flashlipschitz.layers import OrthoConv2d
from flashlipschitz.layers import OrthoConv2d as BCOP_new
from flashlipschitz.layers.block_ortho_conv import BCOP as BCOP_old
from flashlipschitz.layers.conv.rko_conv import UnitNormLinear
from flashlipschitz.layers.custom_activations import MaxMin
from flashlipschitz.models_factory import LipResNet
from flashlipschitz.models_factory import Residual
from flashlipschitz.models_factory import SplitConcatNet
from flashlipschitz.models_factory import SplitConcatNetConfigs

# from flashlipschitz.layers.conv.reparametrizers import BjorckParams

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

layers = [
    ("BCOP_new", BCOP_new),
    ("BCOP_old", BCOP_old),
    ("Conv2D", Conv2d),
]


class ImagenetDataModule(pytorch_lightning.LightningDataModule):
    # Dataset configuration
    _DATA_PATH = os.path.join(f"/local_data/imagenet_cache/ILSVRC/Data/CLS-LOC/")
    _BATCH_SIZE = 256
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


data_module = ImagenetDataModule()


# SplitConcatNet(
#     img_shape=(3, 224, 224),
#     n_classes=1000,
#     **SplitConcatNetConfigs["M3"],
# )
res = []

for layer_name, layer_cls in layers:
    # config = SplitConcatNetConfigs["M3"].copy()
    config = {}
    config["conv"] = ClassParam(
        layer_cls,
        bias=False,
        padding="same",
        padding_mode="zeros",
    )
    # config = dict(
    #     skip=ClassParam(
    #         Residual,
    #         init_val=1.0,
    #     ),
    #     conv=ClassParam(
    #         layer_cls,
    #         bias=False,
    #         padding="same",
    #         padding_mode="zeros",
    #         # bjorck_params=BjorckParams(
    #         #     power_it_niter=3,
    #         #     eps=1e-6,
    #         #     bjorck_iters=10,
    #         #     beta=0.5,
    #         #     contiguous_optimization=False,
    #         # ),
    #     ),
    #     act=ClassParam(MaxMin),
    #     lin=ClassParam(UnitNormLinear, bias=False),
    #     norm=None,  # ClassParam(BatchCentering2D),
    #     # pool=ClassParam(nn.LPPool2d, norm_type=2),
    # )
    # dataloader that generate the random data, and random target
    # reset all memory
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
            nrof_batches=100,
            log=print,
        )
    )
    print(f"{layer_name}")
    print("\n".join([f"{k}: {v}" for k, v in res_1.items()]))
    torch.cuda.empty_cache()
    res_1["conv_type"] = layer_name
    res.append(res_1)
    # clear memory
    del conv_layer
    del res_1

pd.DataFrame.from_records(res).to_csv("result.csv")
