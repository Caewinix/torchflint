from typing import Callable, Iterable, Sequence, Optional, Union, Tuple
from functools import reduce
from itertools import chain
import torch
from array_like_generic import combine_mask_index, broadcast_required_shape, broadcast_large_dims
from .nn import Buffer


def apply_from_dim(function: Callable, input: torch.Tensor, dim = 0, otypes: Iterable[torch.dtype] = None) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
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


min = torch.amin


max = torch.amax


def scalar_default_value(scalar_default, eps=1e-6):
    if scalar_default == 'max':
        return 1 - eps
    elif scalar_default == 'min':
        return eps
    else:
        return 0.5


def map_range(
    input: torch.Tensor,
    interval: Sequence[int] = (0, 1),
    dim: Union[int, Sequence[int], None] = None,
    dtype: torch.dtype = None,
    scalar_default: Union[str, None] = 'max',
    eps: float = 1e-6
) -> torch.Tensor:
    min_value: torch.Tensor = torch.amin(input, dim=dim, keepdim=True)
    max_value: torch.Tensor = torch.amax(input, dim=dim, keepdim=True)
    max_min_difference = max_value - min_value
    max_min_equal_mask = max_min_difference == 0
    max_min_difference.masked_fill_(max_min_equal_mask, 1)
    input = input - min_value
    if not (scalar_default is None or scalar_default == 'none'):
        input.masked_fill_(max_min_equal_mask, torch.tensor(scalar_default_value(scalar_default, eps)).to(input.dtype))
    return (input / max_min_difference * (interval[1] - interval[0]) + interval[0]).to(dtype)


def map_ranges(
    input: torch.Tensor,
    intervals: Sequence[Sequence[int]] = [(0, 1)],
    dim: Union[int, Sequence[int], None] = None,
    dtype: torch.dtype = None,
    scalar_default: Union[str, None] = 'max',
    eps: float = 1e-6
) -> Tuple[torch.Tensor, ...]:
    min_value: torch.Tensor = torch.amin(input, dim=dim, keepdim=True)
    max_value: torch.Tensor = torch.amax(input, dim=dim, keepdim=True)
    max_min_difference = max_value - min_value
    max_min_equal_mask = max_min_difference == 0
    max_min_difference.masked_fill_(max_min_equal_mask, 1)
    input = input - min_value
    if not (scalar_default is None or scalar_default == 'none'):
        input.masked_fill_(max_min_equal_mask, torch.tensor(scalar_default_value(scalar_default, eps)).to(input.dtype))
    normed = input / max_min_difference
    def generator():
        for interval in intervals:
            yield (normed * (interval[1] - interval[0]) + interval[0]).to(dtype)
    return tuple(*generator())


def gamma(input: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    return torch.exp(torch.lgamma(input, out=out), out=out)


def gamma_div(left: torch.Tensor, right: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
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


def arith_gamma_prod(arith_term: torch.Tensor, arith_base: torch.Tensor, ratio_base: torch.Tensor) -> torch.Tensor:
    arith_lgamma, gamma_sign = recur_lgamma(arith_term, arith_base)
    ratio_base_sign = torch.sign(ratio_base)
    ratio_base_sign[(ratio_base_sign == -1) & (arith_term % 2 == 0)] = 1
    return torch.exp(arith_term * torch.log(torch.abs(ratio_base)) + arith_lgamma) * gamma_sign * ratio_base_sign


def linspace(start, stop, num, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    start = torch.as_tensor(start)
    stop = torch.as_tensor(stop)
    num = torch.as_tensor(num)
    common_difference = torch.as_tensor((stop - start) / (num - 1).to(dtype=dtype))
    index = torch.arange(num).to(common_difference.device)
    return start + common_difference * index.view([*index.shape] + [1] * len(common_difference.shape))


def linspace_at(index, start, stop, num, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    start = torch.as_tensor(start)
    stop = torch.as_tensor(stop)
    num = torch.as_tensor(num)
    common_difference = torch.as_tensor((stop - start) / (num - 1).to(dtype=dtype))
    index = torch.as_tensor(index)
    return start + common_difference * index.view([*index.shape] + [1] * len(common_difference.shape))


def linspace_cover(index, start, stop, num, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    start = torch.as_tensor(start)
    stop = torch.as_tensor(stop)
    num = torch.as_tensor(num)
    common_difference = (stop - start) / (num - 1).to(dtype=dtype)
    index = torch.as_tensor(index)
    return start + common_difference * index.view([*index.shape] + [1] * (len(common_difference.shape) - len(index.shape)))


def linspace_cumprod_at(index, start, stop, num, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
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


def linspace_cumprod_cover(index, start, stop, num, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
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


def invert(input: torch.Tensor) -> torch.Tensor:
    shape_length = len(input.shape)
    if shape_length > 2:
        dims = (shape_length - 2, shape_length - 1)
    else:
        dims = (shape_length - 1,)
    min_values = torch.amin(input, dim=dims, keepdim=True)
    max_values = torch.amax(input, dim=dims, keepdim=True)
    return max_values - input + min_values


def buffer(tensor: Optional[torch.Tensor], persistent: bool = True) -> torch.Tensor:
    return Buffer(tensor, persistent)


def advanced_indexing(shape, indices, dim):
    len_shape = len(shape)
    indices_shape = indices.shape
    left_args = _limited_dim_indexing(shape[:dim], len_shape)
    right_args = _limited_dim_indexing(shape[dim + 1:], len_shape, dim + 1)
    return *left_args, indices.view((*indices_shape,) + (1,) * (len_shape - len(indices_shape))), *right_args


def grow(input: torch.Tensor, ndim: int, direction: str = 'leading') -> torch.Tensor:
    tensor_shape = input.shape
    remaining_dims_num = ndim - len(tensor_shape)
    if direction == 'leading':
        return input.view((*((1,) * remaining_dims_num), *tensor_shape))
    elif direction == 'trailing':
        return input.view((*tensor_shape, *((1,) * remaining_dims_num)))
    else:
        return input


def is_int(dtype) -> bool:
    try:
        torch.iinfo(dtype)
        return True
    except TypeError:
        return False


def is_float(dtype) -> bool:
    try:
        torch.finfo(dtype)
        return True
    except TypeError:
        return False


def _shift_arguments(input: torch.Tensor, shift: torch.Tensor):
    batch_size, num_channels, *spatial_shape = input.shape
    length_ndim = len(spatial_shape)
    expected_shifts_size = torch.Size((batch_size, num_channels, length_ndim))
    
    if is_float(shift.dtype):
        shift = shift.round().int()
    if shift.shape != expected_shifts_size:
        shift = shift.broadcast_to(expected_shifts_size)
    del expected_shifts_size
    
    device = input.device

    aranged = torch.arange(max(input.shape), device=device)
    batch_size_indices, num_channels_indices, *axes = tuple(aranged[:each] for each in input.shape)
    grid = torch.stack(torch.meshgrid(*axes, indexing='ij'))
    del aranged, axes
    grid = grid.unsqueeze(1).unsqueeze(1).expand(-1, batch_size, num_channels, *spatial_shape)

    expanded_shift = shift.permute(2, 0, 1).reshape(length_ndim, batch_size, num_channels, *([1] * length_ndim))
    source_indices = grid - expanded_shift
    del expanded_shift, shift

    length = torch.as_tensor(spatial_shape, device=device).reshape(length_ndim, *([1] * (2 + length_ndim)))
    valid = ((source_indices >= 0) & (source_indices < length)).all(dim=0)
    del length

    batch_stride, channels_stride, *length_stride = input.stride()
    length_stride = torch.as_tensor(length_stride, device=device).reshape(length_ndim, *([1] * (2 + length_ndim)))

    destination_inplane = (grid * length_stride).sum(dim=0)
    source_inplane = (source_indices * length_stride).sum(dim=0)
    del grid, length_stride, source_indices

    batch_size_indices = batch_size_indices.view(batch_size, 1, *([1] * length_ndim)).expand(batch_size, num_channels, *spatial_shape)
    num_channels_indices = num_channels_indices.view(1, num_channels, *([1] * length_ndim)).expand(batch_size, num_channels, *spatial_shape)
    batch_channels_offset = batch_size_indices * batch_stride + num_channels_indices * channels_stride
    del batch_stride, channels_stride, batch_size_indices, num_channels_indices

    source_flat = (batch_channels_offset + source_inplane).reshape(-1)
    destination_flat = (batch_channels_offset + destination_inplane).reshape(-1)
    del destination_inplane, source_inplane, batch_channels_offset
    
    mask = valid.reshape(-1)
    return destination_flat, source_flat, mask


def shift_(input: torch.Tensor, shift: torch.Tensor, fill_value: int = 0) -> torch.Tensor:
    destination_flat, source_flat, mask = _shift_arguments(input, shift)
    input.view(-1)[destination_flat[mask]] = input.view(-1)[source_flat[mask]]
    input.view(-1)[destination_flat[~mask]] = fill_value
    return input


def shift(input: torch.Tensor, shift: torch.Tensor, fill_value: int = 0) -> torch.Tensor:
    out = torch.full_like(input, fill_value)
    destination_flat, source_flat, mask = _shift_arguments(input, shift)
    out.view(-1)[destination_flat[mask]] = input.view(-1)[source_flat[mask]]
    return out


def promote_types(*types) -> torch.dtype:
    return reduce(torch.promote_types, types)


def _tailing_dim(ndim: int, dim: Sequence[int]):
    inv_dims = [0] * ndim
    def generate_dims():
        nonlocal inv_dims
        for permute_back_dim, permuted_dim in enumerate(chain((d for d in range(ndim) if d not in dim), dim)):
            inv_dims[permuted_dim] = permute_back_dim
            yield permuted_dim
    tailing_dims = type(dim)(generate_dims())
    return tailing_dims, inv_dims


def _limited_dim_indexing(using_shape, len_shape, start: int = 0):
    return (torch.arange(dim_size).view((1,) * (i + start) + (dim_size,) + (1,) * (len_shape - (i + start) - 1)) for i, dim_size in enumerate(using_shape))


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