# PyTorch Extension

## Overview

This Python module provides a collection of utility functions designed for advanced tensor manipulation using PyTorch. It includes functions for applying operations along specific dimensions, mapping values to new ranges, and generating linearly spaced tensors, among others.

## Functions

### `buffer(tensor, persistent)`
Used in the `nn.Module`, for registering a buffer in an assignment form.

### `apply_from_dim(func, tensor, dim, otypes)`
Applies a given function to a specified dimension of a tensor.

### `min_dims(tensor, dims, keepdim, out)`
Computes the minimum values over specified dimensions.

### `max_dims(tensor, dims, keepdim, out)`
Computes the maximum values over specified dimensions.

### `map_range(tensor, interval, dim, dtype, scalar_default, eps)`
Maps tensor values to a specified range.

### `map_ranges(tensor, intervals, dim=None, dtype, scalar_default, eps)`
Maps tensor values to multiple specified ranges.

### `gamma(input, out)`
Calculates the gamma function for each element in the tensor.

### `gamma_div(left, right, out)`
Calculates the division of gamma functions for corresponding elements of two tensors.

### `recur_lgamma(n, base)`
Calculates the recursive logarithm of the gamma function.

### `arith_gamma_prod(arith_term, arith_base, ratio_base)`
Calculates the product of terms using the arithmetic series and gamma function.

### `linspace(start, stop, num, dtype)`
Generates linearly spaced values between `start` and `stop`, supporting `Tensor` as input.

### `linspace_at(index, start, stop, num, dtype)`
Generates linearly spaced values at specific indices.

### `invert(tensor)`
Inverts the values in the tensor across its dimensions.

### `nn.refine_model(model)`
Extracts the underlying model from a DataParallel wrapper, if present.

### `nn.Buffer(tensor, persistent)`
The class that used in the `buffer(tensor, persistent)`.

## Usage

These functions are intended for use with PyTorch tensors in deep learning and numerical computation contexts. Each function provides additional control over tensor operations, particularly in high-dimensional data manipulation and preprocessing.