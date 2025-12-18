# TorchFlint: Advanced Tensor Operations and State Management for PyTorch

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/torchflint/badge/?version=latest)](https://torchflint.readthedocs.io/)
[![Python Version](https://img.shields.io/badge/Python-3.7%2B-green)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-1.6%2B-orange)](https://pytorch.org/)

**TorchFlint** is a collection of advanced utility functions designed for tensor manipulation within the [PyTorch](https://pytorch.org/) ecosystem. It provides granular control over high-dimensional data processing, bridging the gap between high-level neural modules and low-level tensor operations.

## ✨ Key Features

### 1. Transparent Buffer Management

Simplifies the registration of non-parameter states in `nn.Module` using a Pythonic assignment syntax.

* **`torchflint.buffer(tensor, persistent)`**: Wraps a tensor so it can be assigned directly inside a module. It automatically handles device movement and `state_dict` saving without using `torch.nn.Module.register_buffer`.
* **Containers**: Includes `BufferObject`, `BufferList` and `BufferDict` for organized state management.

### 2. Patchwork

A core module for handling N-dimensional patches, serving as the foundation for convolution and pooling operations.

* **High Performance**: Includes `torchflint.patchwork.unfold_space`, `torchflint.patchwork.fold_space`, and `torchflint.patchwork.fold_stack`. Notably, the speed of **`torchflint.patchwork.fold_stack` is close to the official `torch.nn.functional.fold` (only support images less than 2D)** in many scenarios.
* **N-Dimensional Support**: Works seamlessly on 1D, 2D, and 3D+ spatial data.
* **Functional Convolution & Pooling**: Built upon the `torchflint.patchwork` module, many functions expose the intermediate steps of convolution and pooling, like `torchflint.patchwork.conv_patches`, `torchflint.patchwork.masked_conv`, `torchflint.patchwork.pool`, `torchflint.patchwork.max_pool`, and `torchflint.patchwork.masked_avg_pool`. However, although many of them works well in some tests, but limited (masked version was only tested for their forward process), they will be tested more rigorously in the future.

### 3. Convenient Tensor Utilities

A suite of tools in `torchflint.functional`.

## 🛠️ Installation

```bash
pip install torchflint
```

*Requirement: Python 3.7+ and PyTorch >= 1.6.0.*

## 📚 Documentation

Complete API documentation:  
https://torchflint.readthedocs.io/

## 🚀 Usage Examples

### Using Buffers

Register buffers via simple assignment:

```python
import torch
import torchflint


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Registers a persistent buffer named 'buffer'
        self.buffer = torchflint.buffer(torch.zeros(10), persistent=True)


model = MyModule()
```

### Patch-based Operations

Using the high-performance patch functions:

```python
from torchflint import patchwork


tensor = torch.randn(32, 3, 32, 32)


# Extract patches
patches = patchwork.unfold_space(tensor, kernel_size=3, stride=2, padding=1)


# Reconstruct
cumulative_output = patchwork.fold_stack(patches, stride=2)
average_output = patchwork.fold_space(patches, stride=2)
```

### Convolution (Standard & Masked)

Perform convolutions, optionally applying a mask to handle sparse data or boundary validity.

```python
from torchflint import patchwork


tensor = torch.randn(1, 3, 32, 32)
weight = torch.randn(16, 3, 3, 3) # Standard weight shape: [Out, In, K, K]


# Standard Convolution
output = patchwork.conv(tensor, weight, stride=2, padding=1)


# Masked Convolution
# Apply convolution only where input_mask is valid (1)
input_mask = torch.randn(1, 1, 32, 32) > 0.5
masked_output = patchwork.masked_conv(
    tensor, weight, 
    stride=2, padding=1, 
    input_mask=input_mask
)
```

### Pooling (Standard & Masked)

Apply pooling operations. Masked pooling allows accurate statistical reduction by ignoring invalid regions instead of padding with zeros or infinity.

```python
from torchflint import patchwork


tensor = torch.randn(1, 3, 32, 32)


# Standard Max Pooling
output = patchwork.max_pool(tensor, kernel_size=2, stride=2)


# Masked Max Pooling
# Regions marked as 0 in the mask are ignored during max calculation
# (Useful for non-rectangular images or padding handling)
mask = torch.ones_like(tensor)
mask[..., 0:5, 0:5] = 0 # Invalidate top-left corner
masked_output = patchwork.masked_max_pool(
    tensor, kernel_size=2, stride=2, 
    mask=mask.bool()
)
```

## 📄 License

This project is licensed under the MIT License.

## Citation
```bibtex
@misc{torchflint,
  author = {Juanhua Zhang},
  title = {TorchFlint: Advanced Tensor Operations and State Management for PyTorch},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Caewinix/torchflint}}
}
```