Spatial Operations
==================

TorchFlint offers advanced tensor manipulations including high-performance folding/unfolding and masked convolutions.

Patch-based Operations
----------------------

Using the high-performance patch functions:

.. code-block:: python

    from torchflint import patchwork
    import torch

    tensor = torch.randn(32, 3, 32, 32)

    # Extract patches
    patches = patchwork.unfold_space(tensor, kernel_size=3, stride=2, padding=1)

    # Reconstruct
    cumulative_output = patchwork.fold_stack(patches, stride=2)
    average_output = patchwork.fold_space(patches, stride=2)

Convolution (Standard & Masked)
-------------------------------

Perform convolutions, optionally applying a mask to handle sparse data or boundary validity.

.. code-block:: python

    from torchflint import patchwork

    tensor = torch.randn(1, 3, 32, 32)
    weight = torch.randn(16, 3, 3, 3) # Standard weight shape: [Out, In, K, K]

    # Masked Convolution
    # Apply convolution only where input_mask is valid (1)
    input_mask = torch.randn(1, 1, 32, 32) > 0.5
    masked_output = patchwork.masked_conv(
        tensor, weight, 
        stride=2, padding=1, 
        input_mask=input_mask
    )

Pooling
-------

Masked pooling allows accurate statistical reduction by ignoring invalid regions instead of padding with zeros or infinity.

.. code-block:: python

    # Masked Max Pooling
    mask = torch.ones_like(tensor)
    mask[..., 0:5, 0:5] = 0 # Invalidate top-left corner
    
    masked_output = patchwork.masked_max_pool(
        tensor, kernel_size=2, stride=2, 
        mask=mask.bool()
    )