# Orthogonium: Improved implementations of orthogonal layers

This library aims to centralize, standardize and improve methods to 
build orthogonal layers, with a focus on convolutional layers . We noticed that a layer's implementation play a
significant role in the final performance : a more efficient implementation 
allows larger networks and more training steps within the same compute 
budget. So our implementation differs from original papers in order to 
be faster, to consume less memory or more flexible.

## Our flagship: AOC

AOC is a method that allows to build orthogonal convolutions with 
an explicit kernel, that support all features like stride, conv transposed,
grouped convolutions and dilation. This approach is highly scalable, and can
be applied to problems like Imagenet-1K. 

### Install the library:

The library will soon be available on pip, in the meanwhile, you can clone the repository and run the following command 
to install it locally:
```
pip install -e .
```

### Use the layer:

```python
from orthogonium.layers import AdaptiveOrthoConv2d
from orthogonium.layers.linear.reparametrizers import DEFAULT_ORTHO_PARAMS

# use OrthoConv2d with the same params as torch.nn.Conv2d

conv = AdaptiveOrthoConv2d(
  kernel_size=3,
  in_channels=256,
  out_channels=256,
  stride=2,
  groups=16,
  bias=bias,
  padding=(kernel_size // 2, kernel_size // 2),
  padding_mode="circular",
  ortho_params=DEFAULT_ORTHO_PARAMS
)
```

# Model Zoo

Stay tuned, a model zoo will be available soon !

# Ongoing developments

Layers:
- SOC:
  - remove channels padding to handle ci != co efficiently
  - enable groups
  - enable support for native stride, transposition and dilation
- AOL:
  - torch implementation of AOL
- Sandwish:
  - import code
  - plug AOC into Sandwish conv

ZOO:
- models from the paper
