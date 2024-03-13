from timeit import timeit
import time
import torch
from torch.nn import Conv2d
from flashlipschitz.layers.block_ortho_conv import BCOP as BCOP_old
from flashlipschitz.layers.fast_block_ortho_conv import BCOP as BCOP_new
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

layers = [("BCOP_new", BCOP_new), ("BCOP_old", BCOP_old), ("Conv2D", Conv2d)]


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

for kernel_size in [3, 5]:
    for stride in [1, 2]:
        for input_channels in [64, 128, 256]:
            for out_channels in [64, 128, 256]:
                for layer_name, layer_cls in layers:
                    input_shape = [128, input_channels, 32, 32]
                    conv_layer = layer_cls(
                        input_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding="same" if stride == 1 else "valid",
                        bias=False,
                    )
                    conv_layer.to(device)
                    conv_layer.train()
                    res_1 = sim_learning(
                        conv_layer,
                        input_shape,
                        niter,
                    )
                    res_1.update(
                        {
                            "method": layer_name,
                            "input_channels": input_channels,
                            "out_channels": out_channels,
                            "stride": stride,
                            "kernel_size": kernel_size,
                        }
                    )
                    res.append(res_1)
                    print(",".join([f"{k}:{v}" for k, v in res_1.items()]))
                    # clear cuda cache
                    torch.cuda.empty_cache()
                    # clear memory
                    del conv_layer
pd.DataFrame(res).to_csv("bench_bcop.csv")
