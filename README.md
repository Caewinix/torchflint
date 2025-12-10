# TorchFlint

## Overview

This Python module provides a collection of utility functions designed for advanced tensor manipulation using PyTorch. It includes functions for applying operations along specific dimensions, mapping values to new ranges, and generating linearly spaced tensors, among others.

## Patchwork

A new module that handles patches, for instance, `unfold`, `fold`, `unfold_space`, `fold_space`, and `fold_stack`. These functions were tested and work well and fast (for instance, `fold_stack`, as a substitute for the PyTorch official `torch.nn.functional.fold`, is faster than that).

Moreover, module `convolution` and `pool` have been developed from that, many of them works well by some tests, but limited (masked version was only tested for their forward process), they will be tested more rigorously in the future.

## Functions

### `buffer(tensor, persistent)`
Used in the `nn.Module`, for registering a buffer in an assignment form.

### `map_range(tensor, interval, dim, dtype, scalar_default, eps)`
Maps tensor values to a specified range.

### `map_ranges(tensor, intervals, dim=None, dtype, scalar_default, eps)`
Maps tensor values to multiple specified ranges.

### `invert(tensor)`
Inverts the values in the tensor across its dimensions.

### `nn.Buffer(tensor, persistent)`
The class that used in the `buffer(tensor, persistent)`.

## Usage

These functions are intended for use with PyTorch tensors in deep learning and numerical computation contexts. Each function provides additional control over tensor operations, particularly in high-dimensional data manipulation and preprocessing.