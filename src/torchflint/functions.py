from typing import Callable, Iterable, Sequence, Optional, Union, Tuple
from enum import Enum, auto
import torch
from torch.types import _dtype
import numpy as np
from array_like_generic import *
from .nn import Buffer
from .private import _min_max_dims


def apply_from_dim(function: Callable, input: torch.Tensor, dim = 0, otypes: Iterable[_dtype] = None):
    '''
    Select a dim of an `torch.Tensor` and apply a function.
    '''
    slices = (slice(None, None, None),) * dim
    if len(otypes) > 1:
        return tuple(torch.as_tensor(item, dtype=otypes[i], device=input.device) for i, item in enumerate(zip(*[function(input[slices + (i,)]) for i in range(input.shape[dim])])))
    elif len(otypes) == 1:
        return torch.as_tensor([function(input[slices + (i,)]) for i in range(input.shape[dim])], otypes[0], device=input.device)
    else:
        [function(input[slices + (i,)]) for i in range(input.shape[dim])]


def min(input: torch.Tensor, dim: Sequence[int], keepdim=False, *, out: Union[torch.Tensor, Tuple[torch.Tensor, ...], Sequence[torch.Tensor], None] = None):
    return _min_max_dims(torch.min, input, dim, keepdim, out=out)


def max(input: torch.Tensor, dim: Sequence[int], keepdim=False, *, out: Union[torch.Tensor, Tuple[torch.Tensor, ...], Sequence[torch.Tensor], None] = None):
    return _min_max_dims(torch.max, input, dim, keepdim, out=out)


def min_dims(tensor: torch.Tensor, dims: Sequence[int], keepdim=False, *, out: Union[torch.Tensor, Tuple[torch.Tensor, ...], Sequence[torch.Tensor], None] = None):
    return min(tensor, dims, keepdim, out=out)


def max_dims(tensor: torch.Tensor, dims: Sequence[int], keepdim=False, *, out: Union[torch.Tensor, Tuple[torch.Tensor, ...], Sequence[torch.Tensor], None] = None):
    return max(tensor, dims, keepdim, out=out)


def map_range(
    input: torch.Tensor,
    interval: Sequence[int] = (0, 1),
    dim: Union[int, Sequence[int], None] = None,
    dtype: torch.dtype = None,
    scalar_default: ScalarDefault = ScalarDefault.max,
    eps: float = 1e-6
):
    min_value: torch.Tensor = min_dims(input, dims=dim, keepdim=True)
    max_value: torch.Tensor = max_dims(input, dims=dim, keepdim=True)
    max_min_difference = max_value - min_value
    max_min_equal_mask = max_min_difference == 0
    max_min_difference.masked_fill_(max_min_equal_mask, 1)
    input = (input - min_value).masked_fill(max_min_equal_mask, torch.tensor(scalar_default_value(scalar_default, eps)).to(input.dtype))
    return (input / max_min_difference * (interval[1] - interval[0]) + interval[0]).to(dtype)


def map_ranges(
    input: torch.Tensor,
    intervals: Sequence[Sequence[int]] = [(0, 1)],
    dim: Union[int, Sequence[int], None] = None,
    dtype: torch.dtype = None,
    scalar_default: ScalarDefault = ScalarDefault.max,
    eps: float = 1e-6
):
    min_value: torch.Tensor = min_dims(input, dims=dim, keepdim=True)
    max_value: torch.Tensor = max_dims(input, dims=dim, keepdim=True)
    max_min_difference = max_value - min_value
    max_min_equal_mask = max_min_difference == 0
    max_min_difference.masked_fill_(max_min_equal_mask, 1)
    normed = ((input - min_value) / max_min_difference).masked_fill_(max_min_equal_mask, torch.tensor(scalar_default_value(scalar_default, eps)).to(input.dtype))
    def generator():
        for interval in intervals:
            yield (normed * (interval[1] - interval[0]) + interval[0]).to(dtype)
    return tuple(*generator())


def gamma(input: torch.Tensor, *, out: torch.Tensor | None = None):
    return torch.exp(torch.lgamma(input, out=out), out=out)


def gamma_div(left: torch.Tensor, right: torch.Tensor, *, out: torch.Tensor | None = None):
    return torch.exp(torch.lgamma(left, out=out) - torch.lgamma(right, out=out), out=out)


def recur_lgamma(n: torch.Tensor, base: torch.Tensor):
    out = torch.lgamma(base + n) - torch.lgamma(base)
    gamma_sign = torch.empty_like(base, dtype=torch.int64)
    gamma_sign[base > 0] = 1
    negative_one_exponent = torch.ceil(-base).to(device=base.device, dtype=int)
    negative_one_exponent[negative_one_exponent > n] = torch.broadcast_to(n, negative_one_exponent.shape)[negative_one_exponent > n]
    negative_base_mask = base < 0
    even_mod_exponent = negative_one_exponent % 2
    gamma_sign[negative_base_mask & (even_mod_exponent == 0)] = 1
    gamma_sign[negative_base_mask & (even_mod_exponent != 0)] = -1
    return out, gamma_sign


def arith_gamma_prod(arith_term: torch.Tensor, arith_base: torch.Tensor, ratio_base: torch.Tensor):
    arith_lgamma, gamma_sign = recur_lgamma(arith_term, arith_base)
    ratio_base_sign = torch.sign(ratio_base)
    ratio_base_sign[(ratio_base_sign == -1) & (arith_term % 2 == 0)] = 1
    return torch.exp(arith_term * torch.log(torch.abs(ratio_base)) + arith_lgamma) * gamma_sign * ratio_base_sign


def linspace(start, stop, num, dtype: Optional[_dtype] = None):
    start = torch.as_tensor(start)
    stop = torch.as_tensor(stop)
    num = torch.as_tensor(num)
    common_difference = torch.as_tensor((stop - start) / (num - 1).to(dtype=dtype))
    index = torch.arange(num).to(common_difference.device)
    return start + common_difference * index.view([*index.shape] + [1] * len(common_difference.shape))


def linspace_at(index, start, stop, num, dtype: Optional[_dtype] = None):
    start = torch.as_tensor(start)
    stop = torch.as_tensor(stop)
    num = torch.as_tensor(num)
    common_difference = torch.as_tensor((stop - start) / (num - 1).to(dtype=dtype))
    index = torch.as_tensor(index)
    return start + common_difference * index.view([*index.shape] + [1] * len(common_difference.shape))


def linspace_cover(index, start, stop, num, dtype: Optional[_dtype] = None):
    start = torch.as_tensor(start)
    stop = torch.as_tensor(stop)
    num = torch.as_tensor(num)
    common_difference = (stop - start) / (num - 1).to(dtype=dtype)
    index = torch.as_tensor(index)
    return start + common_difference * index.view([*index.shape] + [1] * (len(common_difference.shape) - len(index.shape)))


def linspace_cumprod_at(index, start, stop, num, dtype: Optional[_dtype] = None):
    start = torch.as_tensor(start)
    stop = torch.as_tensor(stop)
    num = torch.as_tensor(num)
    common_difference = (stop - start) / (num - 1).to(dtype=dtype)
    index = torch.as_tensor(index)
    result_index_prefix = (slice(None),) * len(index.shape)
    n = index + 1
    n_shape = n.shape
    result = torch.zeros((*n.shape, *common_difference.shape), device=common_difference.device, dtype=dtype)
    zero_common_difference_mask = common_difference == 0
    left_required_shape, right_required_shape = broadcast_required_shape(start[zero_common_difference_mask].shape, n_shape)
    n = n.view(right_required_shape)
    result[combine_mask_index(result_index_prefix, zero_common_difference_mask)] = torch.pow(start[zero_common_difference_mask].view(left_required_shape), n)
    sequence_with_zero_mask = torch.zeros_like(zero_common_difference_mask, device=common_difference.device, dtype=bool)
    nonzero_common_difference_mask = common_difference != 0
    sequence_with_zero_mask[nonzero_common_difference_mask] = ((start[nonzero_common_difference_mask] % common_difference[nonzero_common_difference_mask] == 0)
                                                               & (torch.sign(start[nonzero_common_difference_mask]) != torch.sign(common_difference[nonzero_common_difference_mask])))
    del nonzero_common_difference_mask
    result[combine_mask_index(result_index_prefix, sequence_with_zero_mask)] = 0
    first_divided_mask = torch.logical_not(sequence_with_zero_mask | zero_common_difference_mask)
    del zero_common_difference_mask
    del sequence_with_zero_mask
    first_divided = torch.full(common_difference.shape, torch.nan, device=common_difference.device, dtype=torch.float64)
    first_divided[first_divided_mask] = start[first_divided_mask] / common_difference[first_divided_mask].to(torch.float64)
    neg_int_arith_base_mask = ((torch.abs(torch.round(first_divided) - first_divided) < torch.pow(10, torch.floor(torch.log10(first_divided)) + 9))
                               & (torch.sign(first_divided) != 1))
    non_neg_int_arith_base_mask = torch.logical_not(neg_int_arith_base_mask) & first_divided_mask
    neg_int_arith_base_mask &= first_divided_mask
    if torch.any(non_neg_int_arith_base_mask == True):
        left_required_shape, right_required_shape = broadcast_required_shape(common_difference[non_neg_int_arith_base_mask].shape, n_shape)
        n = n.view(right_required_shape)
        result[combine_mask_index(result_index_prefix, non_neg_int_arith_base_mask)] = arith_gamma_prod(n, first_divided[non_neg_int_arith_base_mask].view(left_required_shape), common_difference[non_neg_int_arith_base_mask].view(left_required_shape)).to(result.dtype)
    del non_neg_int_arith_base_mask
    if torch.any(neg_int_arith_base_mask == True):
        left_required_shape, right_required_shape = broadcast_required_shape(common_difference[neg_int_arith_base_mask].shape, n_shape)
        n = n.view(right_required_shape)
        result[combine_mask_index(result_index_prefix, neg_int_arith_base_mask)] = arith_gamma_prod(n, -first_divided[neg_int_arith_base_mask].view(left_required_shape) - n + 1, -common_difference[neg_int_arith_base_mask].view(left_required_shape)).to(result.dtype)
    return result


def linspace_cumprod_cover(index, start, stop, num, dtype: Optional[_dtype] = None):
    start = torch.as_tensor(start)
    stop = torch.as_tensor(stop)
    num = torch.as_tensor(num)
    common_difference = (stop - start) / (num - 1).to(dtype=dtype)
    index = torch.as_tensor(index)
    n = index + 1
    n_shape = n.shape
    result = torch.zeros(common_difference.shape, device=common_difference.device, dtype=dtype)
    zero_common_difference_mask = common_difference == 0
    n_required_shape = broadcast_large_dims(common_difference.shape, n_shape)
    n = n.view(n_required_shape)
    result[zero_common_difference_mask] = start[zero_common_difference_mask].to(result.dtype)
    result = torch.pow(result, n)
    sequence_with_zero_mask = torch.zeros_like(zero_common_difference_mask, device=common_difference.device, dtype=bool)
    nonzero_common_difference_mask = common_difference != 0
    sequence_with_zero_mask[nonzero_common_difference_mask] = ((start[nonzero_common_difference_mask] % common_difference[nonzero_common_difference_mask] == 0)
                                                               & (torch.sign(start[nonzero_common_difference_mask]) != torch.sign(common_difference[nonzero_common_difference_mask])))
    del nonzero_common_difference_mask
    result[sequence_with_zero_mask] = 0
    first_divided_mask = torch.logical_not(sequence_with_zero_mask | zero_common_difference_mask)
    del zero_common_difference_mask
    del sequence_with_zero_mask
    first_divided = torch.full(common_difference.shape, torch.nan, device=common_difference.device, dtype=torch.float64)
    first_divided[first_divided_mask] = start[first_divided_mask] / common_difference[first_divided_mask].to(torch.float64)
    neg_int_arith_base_mask = ((torch.abs(torch.round(first_divided) - first_divided) < torch.pow(10, torch.floor(torch.log10(first_divided)) + 9))
                               & (torch.sign(first_divided) != 1))
    non_neg_int_arith_base_mask = torch.logical_not(neg_int_arith_base_mask) & first_divided_mask
    neg_int_arith_base_mask &= first_divided_mask
    if torch.any(non_neg_int_arith_base_mask == True):
        result[non_neg_int_arith_base_mask] = arith_gamma_prod(n, first_divided, common_difference)[non_neg_int_arith_base_mask].to(result.dtype)
    del non_neg_int_arith_base_mask
    if torch.any(neg_int_arith_base_mask == True):
        result[neg_int_arith_base_mask] = arith_gamma_prod(n, -first_divided - n + 1, -common_difference)[neg_int_arith_base_mask].to(result.dtype)
    return result


def invert(input: torch.Tensor):
    shape_length = len(input.shape)
    if shape_length > 2:
        dims = (shape_length - 2, shape_length - 1)
    else:
        dims = (shape_length - 1,)
    min_values = min_dims(input, dims=dims, keepdim=True)
    max_values = max_dims(input, dims=dims, keepdim=True)
    return max_values - input + min_values


def buffer(tensor: Optional[torch.Tensor], persistent: bool = True) -> torch.Tensor:
    return Buffer(tensor, persistent)


def advanced_indexing(shape, indices, dim):
    from .private import _limited_dim_indexing
    len_shape = len(shape)
    indices_shape = indices.shape
    left_args = _limited_dim_indexing(shape[:dim], len_shape)
    right_args = _limited_dim_indexing(shape[dim + 1:], len_shape, dim + 1)
    return *left_args, indices.view((*indices_shape,) + (1,) * (len_shape - len(indices_shape))), *right_args


class DimsGrowthDirection(Enum):
    leading = auto()
    trailing = auto()


def grow_dims(input: torch.Tensor, expected_dims_num: int, direction: DimsGrowthDirection = DimsGrowthDirection.leading):
    tensor_shape = input.shape
    remaining_dims_num = expected_dims_num - len(tensor_shape)
    if direction == DimsGrowthDirection.leading:
        return input.view((*((1,) * remaining_dims_num), *tensor_shape))
    else:
        return input.view((*tensor_shape, *((1,) * remaining_dims_num)))


def shift(input: torch.Tensor, shift: Sequence[int], fill_value = 0):
    empty = torch.full_like(input, fill_value=fill_value)
    shift_slices, container_slices = [np.empty((len(shift),), dtype=object) for _ in range(2)]
    for i, current_shift in enumerate(shift):
        if current_shift > 0:
            shift_slices[i] = slice(current_shift, None, None)
            container_slices[i] = slice(None, -current_shift, None)
        elif current_shift < 0:
            shift_slices[i] = slice(None, current_shift, None)
            container_slices[i] = slice(-current_shift, None, None)
        else:
            shift_slices[i] = slice(None, None, None)
            container_slices[i] = slice(None, None, None)
    shift_slices = (..., *shift_slices)
    container_slices = (..., *container_slices)
    empty[container_slices] = input[shift_slices]
    return empty

# def view_gather(input: torch.Tensor, dim: int, index: torch.Tensor):
#     return input[advanced_indexing(input.shape, index, dim)]

# def map_range(tensor: torch.Tensor, range: Sequence[int] = (0, 1), dim = None, dtype=None, scalar_default = ScalarDefault.max):
#     assert range[0] < range[1]
#     min_value = min_dims(tensor, dims=dim, keepdim=True)
#     max_value = max_dims(tensor, dims=dim, keepdim=True)
#     result = torch.empty(tensor.shape, dtype=dtype, device=tensor.device)
#     unrangable_index = torch.where(max_value == min_value)[0]
#     if len(unrangable_index) > 0:
#         result[unrangable_index] = scalar_default_value(scalar_default, range)
#     rangable_index = torch.where(max_value != min_value)[0]
#     if len(rangable_index) > 0:
#         rangable_min = min_value[rangable_index]
#         result[rangable_index] = ((tensor[rangable_index] - rangable_min) / (max_value[rangable_index] - rangable_min) * (range[1] - range[0]) + range[0]).to(result.dtype)
#     return result

# def map_ranges(tensor: torch.Tensor, ranges: Sequence[int] = [(0, 1)], dim = None, dtype=torch.float32, scalar_default = ScalarDefault.max):
#     min_value = min_dims(tensor, dims=dim, keepdim=True)
#     max_value = max_dims(tensor, dims=dim, keepdim=True)
#     unrangable_index = torch.where(max_value == min_value)[0]
#     rangable_index = torch.where(max_value != min_value)[0]
#     results = np.empty(len(ranges), dtype=object)
#     rangable_min = min_value[rangable_index]
#     normed = ((tensor[rangable_index] - rangable_min) / (max_value[rangable_index] - rangable_min)).to(dtype=dtype)
#     del rangable_min
#     for i, range in enumerate(ranges):
#         assert range[0] < range[1]
#         results[i] = torch.empty(tensor.shape, dtype=dtype, device=tensor.device)
#         if len(unrangable_index) > 0:
#             results[i][unrangable_index] = scalar_default_value(scalar_default, range)
#         if len(rangable_index) > 0:
#             results[i][rangable_index] = (normed * (range[1] - range[0]) + range[0]).to(dtype=dtype)
#     return results

# torch.apply_from_dim = apply_from_dim
# torch.map_range = map_range
# torch.map_ranges = map_ranges
# torch.linspace_at = linspace_at
# torch.linspace_cover = linspace_cover
# torch.linspace_cumprod_at = linspace_cumprod_at
# torch.linspace_cumprod_cover = linspace_cumprod_cover
# torch.invert = invert
# torch.buffer = buffer