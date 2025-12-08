# PyTorch Extension

## Overview

This Python module provides a collection of utility functions designed for advanced tensor manipulation using PyTorch. It includes functions for applying operations along specific dimensions, mapping values to new ranges, and generating linearly spaced tensors, among others.

## Patchwork

A new module that handles patches, for instance, `unfold`, `fold`, `unfold_space`, `fold_space`, and `fold_stack`. These functions were tested and work well and fast.

Moreover, module `convolution` and `pool` have been developed from that, many of them works well by some tests, but limited (only forward part), they will be tested more rigorously in the future.

## Functions

### `buffer(tensor, persistent)`
Used in the `nn.Module`, for registering a buffer in an assignment form.

### `apply_from_dim(func, tensor, dim, otypes)`
Applies a given function to a specified dimension of a tensor.

### `map_range(tensor, interval, dim, dtype, scalar_default, eps)`
Maps tensor values to a specified range.

### `map_ranges(tensor, intervals, dim=None, dtype, scalar_default, eps)`
Maps tensor values to multiple specified ranges.

### `invert(tensor)`
Inverts the values in the tensor across its dimensions.

### `nn.refine_model(model)`
Extracts the underlying model from a DataParallel wrapper, if present.

### `nn.Buffer(tensor, persistent)`
The class that used in the `buffer(tensor, persistent)`.

## Usage

These functions are intended for use with PyTorch tensors in deep learning and numerical computation contexts. Each function provides additional control over tensor operations, particularly in high-dimensional data manipulation and preprocessing.