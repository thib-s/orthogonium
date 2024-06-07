import gc
import time
from timeit import timeit

import pandas as pd
import torch
from batch_times import evaluate_all_model_time_statistics
from memory_usage import get_model_memory
from torch.nn import Conv2d
from torch.profiler import profile
from torch.profiler import ProfilerActivity
from torch.profiler import record_function
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from flashlipschitz.layers import OrthoConv2d as BCOP_new
from flashlipschitz.layers.block_ortho_conv import BCOP as BCOP_old
from flashlipschitz.layers.conv.reparametrizers import BjorckParams


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

layers = [("BCOP_new", BCOP_new), ("BCOP_old", BCOP_old), ("Conv2D", Conv2d)]


class RandomDataset(Dataset):
    def __init__(self, data_shape, target_shape, num_samples):
        self.data_shape = data_shape
        self.target_shape = target_shape
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.randn(self.data_shape)
        target = torch.randn(self.target_shape)
        return data, target


def sim_learning(
    conv_layer,
    input_shape,
    niter,
):
    torch.profiler._utils._init_for_cuda_graphs()
    time.sleep(1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for i in range(niter):
            conv_layer.zero_grad()
            # Forward pass
            output = conv_layer(
                torch.randn(*input_shape, requires_grad=False).to(device)
            )
            # Perform a backward pass to compute gradients
            output.backward(torch.randn_like(output).to(device))
            # with torch.no_grad():
            #     conv_layer.conv_matrices += 0.05 * conv_layer.conv_matrices.grad
            # conv_layer.conv_matrices.detach().cpu().numpy()
    key_averages = prof.key_averages()
    # Sum up the self CUDA memory usage
    total_memory_usage = sum([item.self_cuda_memory_usage for item in key_averages])
    # Sum up the CPU and CUDA run times
    total_cpu_time = sum([item.cpu_time_total for item in key_averages])
    total_cuda_time = (
        sum([item.cuda_time_total for item in key_averages])
        if torch.cuda.is_available()
        else 0
    )

    # Convert memory usage to MB
    total_memory_usage_mb = total_memory_usage / (1024**2)

    # Convert times to milliseconds
    total_cpu_time_ms = total_cpu_time / 1000
    total_cuda_time_ms = total_cuda_time / 1000
    return {
        "CPU_time_ms": total_cpu_time_ms,
        "CUDA_time_ms": total_cuda_time_ms,
        "CUDA_mem_mb": total_memory_usage_mb,
    }


niter = 10
padding = None
res = []

for kernel_size in [3, 5, 7, 11]:
    for stride in [1, 2]:
        for input_channels in [2, 32, 64, 128, 256]:
            for out_channels in [2, 32, 64, 128, 256]:
                for layer_name, layer_cls in layers:
                    # dataloader that generate the random data, and random target
                    random_data_loader = DataLoader(
                        RandomDataset(
                            (input_channels, 32, 32),
                            (out_channels, 32 // stride, 32 // stride),
                            256,
                        ),
                        batch_size=256,
                    )
                    input_shape = [128, input_channels, 32, 32]
                    # reset all memory
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_max_memory_allocated()
                    gc.collect()
                    torch.cuda.synchronize()
                    res_1 = get_model_memory(
                        lambda: layer_cls(
                            input_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding_mode="circular",
                            padding="same" if stride == 1 else (kernel_size - 1) // 2,
                            bias=False,
                        ),
                        test_loader=random_data_loader,
                        train_loader=random_data_loader,
                        logging=print,
                    )

                    conv_layer = layer_cls(
                        input_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding_mode="circular",
                        padding="same" if stride == 1 else "valid",
                        bias=False,
                    )
                    conv_layer.to(device)
                    conv_layer.train()
                    res_1.update(
                        evaluate_all_model_time_statistics(
                            conv_layer,
                            train_loader=random_data_loader,
                            test_loader=random_data_loader,
                            nrof_batches=25,
                            log=print,
                        )
                    )
                    # res_1 = sim_learning(
                    #     conv_layer,
                    #     input_shape,
                    #     niter,
                    # )
                    metadata = {
                        "method": layer_name,
                        "input_channels": input_channels,
                        "out_channels": out_channels,
                        "stride": stride,
                        "kernel_size": kernel_size,
                    }
                    res_1.update(
                        metadata,
                    )
                    res.append(res_1)
                    # print(",".join([f"{k}:{v}" for k, v in res_1.items()]))
                    # clear cuda cache
                    torch.cuda.empty_cache()
                    # clear memory
                    del conv_layer
                    del random_data_loader
                    print(f"Done: {metadata}")
                    del res_1
                    del metadata
pd.DataFrame(res).to_csv("bench_bcop.csv")
