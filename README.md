<div align="center">
    <img src="assets/banner.png" width="75%" alt="Orthogonium" align="center" />
</div>
<br>


<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.9 ... 3.12-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/Pytorch-2.0 ... 2.5-00008b">
    </a>
    <a href="https://github.com/TODO/orthogonium/actions/workflows/python-lints.yml">
        <img alt="PyLint" src="https://github.com/TODO/orthogonium/actions/workflows/python-lints.yml/badge.svg">
    </a>
    <a href="https://github.com/TODO/orthogonium/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/TODO/orthogonium/actions/workflows/python-tests.yml/badge.svg">
    </a>
    <a href="https://github.com/TODO/orthogonium/actions/workflows/python-publish.yml">
        <img alt="Pypi" src="https://github.com/TODO/orthogonium/actions/workflows/python-publish.yml/badge.svg">
    </a>
    <a href="https://pepy.tech/project/orthogonium">
        <img alt="Pepy" src="https://static.pepy.tech/badge/orthogonium">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>
<br>

# Orthogonium: Improved implementations of orthogonal layers

This library aims to centralize, standardize and improve methods to 
build orthogonal layers, with a focus on convolutional layers . We noticed that a layer's implementation play a
significant role in the final performance : a more efficient implementation 
allows larger networks and more training steps within the same compute 
budget. So our implementation differs from original papers in order to 
be faster, to consume less memory or be more flexible.

## What is included in this library ?

| Layer name   | Description                                                                                                                        | Orthogonal ? | Usage                                                                                                                              | Status         |
|--------------|------------------------------------------------------------------------------------------------------------------------------------|--------------|------------------------------------------------------------------------------------------------------------------------------------|----------------|
| AOC          | The most scalable method to build orthogonal convolution. Allows control of kernel size, stride, groups dilation and convtranspose | Orthogonal   | A flexible method for complex architectures. Preserve orthogonality and works on large scale images.                               | done           |
| Adaptive-SOC | SOC modified to be: i) faster and memory efficient ii) handle stride, groups, dilation & convtranspose                             | Orthogonal   | Good for depthwise convolutions and cases where control over the kernel size is not required                                       | in progress    |
| SLL          | The original SLL layer, which is already quite efficient.                                                                          | 1-Lipschitz  | Well suited for residual blocks, it also contains ReLU activations.                                                                | done           |
| SLL-AOC      | SLL-AOC is to the downsampling block what SLL is to the residual block (see ResNet paper)                                          | 1-Lipschitz  | Allows to construct a "strided" residual block than can change the number of channels. It adds a convolution in the residual path. | done           |
| Sandwish-AOC | Sandwish convolutions that uses AOC to replace the FFT. Allowing it to scale to large images.                                      | 1-Lipschitz  |                                                                                                                                    | pending        |
| Adaptive-ECO | ECO modified to i) handle stride, groups & convtranspose                                                                           | Orthogonal   |                                                                                                                                    | (low priority) |

### directory structure

```
orthogonium
├── layers
│   ├── conv
│   │   ├── AOC
│   │   │   ├── ortho_conv.py # contains AdaptiveOrthoConv2d layer
│   │   ├── AdaptiveSOC
│   │   │   ├── ortho_conv.py # contains AdaptiveSOCConv2d layer (untested)
│   │   ├── SLL
│   │   │   ├── sll_layer.py # contains SDPBasedLipschitzConv, SDPBasedLipschitzDense, SDPBasedLipschitzAOCConv
│   ├── legacy
│   │   ├── original code of BCOP, SOC, Cayley etc.
│   ├── linear
│   │   ├── ortho_linear.py # contains OrthoLinear layer (can be used with BB, QR and Exp parametrization)
│   ├── normalization.py # contains Batch centering and Layer centering
│   ├── custom_activations.py # contains custom activations for 1 lipschitz networks
│   ├── channel_shuffle.py # contains channel shuffle layer  
├── model_factory.py # factory function to construct various models for the zoo
├── losses # loss functions, VRA estimation
```

### AOC:

AOC is a method that allows to build orthogonal convolutions with 
an explicit kernel, that support all features like stride, conv transposed,
grouped convolutions and dilation (and all compositions of these parameters). This approach is highly scalable, and can
be applied to problems like Imagenet-1K.

### Adaptive-SOC:

Adaptive-SOC blend the approach of AOC and SOC. It differs from SOC in the way that it is more memory efficient and 
sometimes faster. It also allows to handle stride, groups, dilation and transposed convolutions. However, it does not allow to 
control the kernel size explicitly as the resulting kernel size is larger than the requested kernel size. 
It is due to the computation to the exponential of a kernel that increases the kernel size at each iteration.

Its development is still in progress, so extra testing is still require to ensure exact orthogonality.

### SLL:

SLL is a method that allows to construct small residual blocks with ReLU activations. We kept most to the original 
implementation, and added `SDPBasedLipschitzAOCConv` that construct a down-sampling residual block by fusing SLL with 
$AOC.

### more layers are coming soon !

## Install the library:

The library will soon be available on pip, in the meanwhile, you can clone the repository and run the following command 
to install it locally:
```
pip install -e .
```

### Use the layer:

```python
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
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

## Model Zoo

Stay tuned, a model zoo will be available soon !



# Disclaimer

Given the great quality of the original implementations, orthogonium do not focus on reproducing exactly the results of
the original papers, but rather on providing a more efficient implementation. Some degradations in the final provable 
accuracy may be observed when reproducing the results of the original papers, we consider this acceptable is the gain 
in terms of scalability is worth it. This library aims to provide more scalable and versatile implementations for people who seek to use orthogonal layers 
in a larger scale setting.

# Contributing

This library is still in a very early stage, so expect some bugs and missing features. Also, before the version 1.0.0,
the API may change and no backward compatibility will be ensured, this will allow a rapid integration of new features.
In order to prioritize the development, we will focus on the most used layers and models. If you have a specific need,
please open an issue, and we will try to address it as soon as possible.

Also, if you have a model that you would like to share, please open a PR with the model and the training script. We will
be happy to include it in the zoo.

If you want to contribute, please open a PR with the new feature or bug fix. We will review it as soon as possible.

## Ongoing developments

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
