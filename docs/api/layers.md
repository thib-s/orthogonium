The most scalable method to build orthogonal convolution. Allows control of kernel size, 
stride, groups dilation and transposed convolutions.

The classes `AdaptiveOrthoConv2d` and `AdaptiveOrthoConv2d` are not classes,
 but factory function to selecte bewteen 3 different parametrizations, as depicted
in the following figure:

<img src="/assets/flowchart_v4.png" alt="drawing" style="width:300px;"/>

::: orthogonium.layers.conv.AOC.ortho_conv
    rendering:
        show_root_toc_entry: True
    selection:
        inherited_members: True


::: orthogonium.layers.conv.AOC.fast_block_ortho_conv
    rendering:
        show_root_toc_entry: true
    selection:
        inherited_members: true

::: orthogonium.layers.conv.AOC.rko_conv
    rendering:
        show_root_toc_entry: true
    selection:
        inherited_members: true

