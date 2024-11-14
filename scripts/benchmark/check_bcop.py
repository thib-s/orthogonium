import torch
from torch.profiler import profile
from torch.profiler import ProfilerActivity

from orthogonium.layers import OrthoConv2d as BCOP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for in_channels in [128, 256]:
    torch.profiler._utils._init_for_cuda_graphs()
    # Define your Conv2D layer with appropriate parameters
    # in_channels = 128  # Number of input channels
    out_channels = 128  # Number of output channels (filters)
    kernel_size = 3  # Size of the convolutional kernel
    stride = 2  # Stride for the convolution
    padding = "circular"  # Padding for the convolution

    conv_layer = BCOP(
        in_channels, out_channels, kernel_size, stride, padding, bias=False
    )
    conv_layer.to(device)
    # conv_layer.compile()
    # print(conv_layer)
    conv_layer.train()
    # Generate random input data
    batch_size = 128  # You can change this to your desired batch size
    input_height = 32  # Height of the input image
    input_width = 32  # Width of the input image
    random_input = torch.randn(batch_size, in_channels, input_height, input_width).to(
        device
    )
    output = conv_layer(random_input)
    output.backward(torch.randn_like(output).to(device))

    random_input = torch.randn(batch_size, in_channels, input_height, input_width).to(
        device
    )
    # Forward pass
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        output = conv_layer(random_input)
        # Perform a backward pass to compute gradients
        output.backward(torch.randn_like(output).to(device))
    # # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=2))
    # # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    # # prof.export_chrome_trace("trace_bcop_new.json")
    # print(prof.key_averages().total_average())
    # print(prof)
    print(
        # f"cpu_time:{prof.key_averages().self_cpu_time_total}",
        # f"cpu_time:{prof.key_averages().total_average().cpu_time_total}",
        f"cpu_time:{prof.key_averages().total_average().self_cpu_time_total}",
        # f"gpu_time:{prof.key_averages().self_cuda_time_total}",
        f"gpu_time:{prof.key_averages().total_average().self_cuda_time_total}",
        # f"gpu_time:{prof.key_averages().total_average().cuda_time_total}",
        f"cuda_memory:{prof.key_averages().total_average().self_cuda_memory_usage}",
        f"cuda_memory:{prof.key_averages().total_average().cuda_memory_usage}",
    )
    total_memory_usage = sum(
        [item.self_cuda_memory_usage for item in prof.key_averages()]
    )

    print(f"Total CUDA memory used: {total_memory_usage} bytes")

# with torch.no_grad():
#     conv_layer.weight += 0.05 * conv_layer.weight.grad

random_input = torch.randn(batch_size, in_channels, input_height, input_width).to(
    device
)

# Forward pass
output = conv_layer(random_input)
# print(conv_layer.weight.max())
# recheck singular values
sv_min, sv_max, stable_rank = conv_layer.singular_values()
print(f"min sv: {sv_min:.4f}, max sv: {sv_max:.4f} stable rank: {stable_rank:.4f}")

##
## compare with old implem
##


# conv_layer = BCOP_old(
#     in_channels, out_channels, kernel_size, stride, padding, bjorck_iters=5, bias=False
# )
# conv_layer.to(device)
# conv_layer.compile()
# print(conv_layer)
# # torch.nn.init.xavier_uniform_(conv_layer.weight)
# conv_layer.train()
# # Generate random input data
# batch_size = 128  # You can change this to your desired batch size
# input_channels = in_channels
# input_height = 32  # Height of the input image
# input_width = 32  # Width of the input image

# random_input = torch.randn(batch_size, input_channels, input_height, input_width).to(
#     device
# )
# output = conv_layer(random_input)
# output.backward(torch.randn_like(output).to(device))

# random_input = torch.randn(batch_size, input_channels, input_height, input_width).to(
#     device
# )
# # Forward pass
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     profile_memory=True,
#     record_shapes=True,
# ) as prof:
#     output = conv_layer(random_input)
#     # Perform a backward pass to compute gradients
#     output.backward(torch.randn_like(output).to(device))
# # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
