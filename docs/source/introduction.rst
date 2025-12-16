Why TorchFlint?
===============

**TorchFlint** provides granular control over high-dimensional data processing, bridging the gap between high-level neural modules and low-level tensor operations.

Key Features
------------

1. Transparent Buffer Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simplifies the registration of non-parameter states in ``nn.Module`` using a Pythonic assignment syntax. It wraps a tensor so it can be assigned directly inside a module, automatically handling device movement and ``state_dict`` saving without using ``torch.nn.Module.register_buffer``.

2. Patchwork & High Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A core module for handling N-dimensional patches. Notably, **torchflint.patchwork.fold_stack is faster than the official torch.nn.functional.fold** in many scenarios.

3. Functional Convolution & Pooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Exposes intermediate steps of convolution and pooling, supporting masked operations for sparse data or boundary validity.
