from typing import overload, Union, Sequence, Tuple, Callable
from functools import partial
from itertools import product
import math
from numbers import Number
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from ..utils import compilable
from ..utils.devices import multidevices


@overload
def unfold(input: Tensor, dimension: int, size: int, step: int) -> Tensor: ...
unfold = Tensor.unfold


def fold_roll(input: Tensor, dimension: int, step: int) -> Tensor:
    size, output_shape = _fold_parameters(input, dimension, step)
    return _fold_roll_overlap_implementation[input.device.type](input, dimension, size, step, output_shape)


def fold(input: Tensor, dimension: int, step: int) -> Tensor:
    device = input.device
    size, output_shape = _fold_parameters(input, dimension, step)
    implementation = _fold_roll_overlap_implementation[device.type]
    ones = implementation(
        torch.ones((1,) * input.ndim, device=device, dtype=input.dtype).expand(input.shape), dimension, size, step, output_shape
    )
    output: Tensor = implementation(input, dimension, size, step, output_shape)
    if output.requires_grad:
        return output / ones
    else:
        return output.div_(ones)


def unfold_space(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1, # cannot be recovered through `fold_space`
    ceil_mode: bool = False, # cannot be recovered through `fold_space`
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    if isinstance(default_space, Number):
        functional_pad = partial(F.pad, mode='constant', value=default_space)
    else:
        functional_pad = partial(F.pad, mode=default_space)
    
    spatial_ndim = input.ndim - 2
    
    kernel_size = _spatialize_tuple(kernel_size, spatial_ndim)
    stride = _spatialize_tuple(stride, spatial_ndim)
    
    while padding != 0:
        if isinstance(padding, int):
            padding = (padding,) * (spatial_ndim * 2)
        elif any(value != 0 for value in padding):
            padding = [value for value in padding[::-1] for _ in range(2)]
        else:
            break
        input = functional_pad(input, padding)
        break
    
    if ceil_mode:
        spatial_shape = input.shape[2:]
        ceil_pad = (math.ceil((length - size) / step) * step + size - length for length, size, step in zip(reversed(spatial_shape), reversed(kernel_size), reversed(stride)))
        if any(pad != 0 for pad in ceil_pad):
            input = functional_pad(input, [pad_value for value in ceil_pad for pad_value in (0, value)])
    
    if dilation == 1:
        iterator = zip(kernel_size, stride)
        single_unfold = unfold
    else:
        dilation = _spatialize_tuple(dilation, spatial_ndim)
        iterator = zip(kernel_size, stride, dilation)
        single_unfold = _dilatedly_unfold
    
    output = input
    for dimension, unfold_args in enumerate(iterator, 2):
        output = single_unfold(output, dimension, *unfold_args)
    return output


def fold_space(
    input: Tensor,
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1
) -> Tensor:
    return _fold_space(input, stride, padding, dilation, folder=_fold_space_overlap)


def fold_stack(
    input: Tensor,
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1
) -> Tensor:
    return _fold_space(input, stride, padding, dilation, folder=_fold_stack_overlap)


def patches_column(input: Tensor) -> Tensor:
    """
    Args:
        input (Tensor): [B, C, out_L_0, out_L_1, ..., out_L_n, k_L_0, k_L_1, ..., k_L_n]
    
    Returns:
        output (Tensor): [B, C * k_L_0 * k_L_1 * ... * k_L_n, out_L_0 * out_L_1 * ... * out_L_n]
    
    """
    double_spatial_ndim = input.ndim - 2
    if double_spatial_ndim % 2 != 0:
        raise ValueError(f"'input' is expected to be a tensor that was completely spatially unfolded")
    spatial_ndim = double_spatial_ndim // 2
    
    output = input.permute(0, 1, *range(2 + spatial_ndim, input.ndim), *range(2, 2 + spatial_ndim))
    output_shape = output.shape
    return output.reshape(output_shape[0], -1, math.prod(output_shape[-spatial_ndim:]))


def keep_kernel_dim(input: Tensor) -> Tensor:
    spatial_ndim = input.ndim - 2
    return input.view(*input.shape, *((1,) * spatial_ndim))


def expand_kernel_dim(input: Tensor, kernel_size: Sequence[int]) -> Tensor:
    spatial_ndim = input.ndim - 2
    output = input.view(*input.shape, *((1,) * spatial_ndim))
    if spatial_ndim != len(kernel_size):
        raise ValueError(f"the length of 'kernel_size' should be equal to the number of spatial dimensions of 'input'")
    output = output.expand(*((-1,) * input.ndim), *kernel_size)
    return output


def fold2d_roll(input: Tensor, stride: Tuple[int, int]) -> Tensor:
    device = input.device
    
    input_shape = input.shape
    batch_size, num_channels = input_shape[:2]
    other_shape = torch.as_tensor(input_shape[2:], device=device)
    kernel_size = other_shape[-2:]
    output_size = (other_shape[:2] - 1) * torch.as_tensor(stride, device=device) + kernel_size
    functional_fold_input = input.permute(0, 1, 4, 5, 2, 3).reshape(batch_size, num_channels * torch.prod(kernel_size).item(), -1)
    
    output = F.fold(
        input=functional_fold_input,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride
    )
    
    return output


def _fold_parameters(input: Tensor, dimension: int, step: int) -> Tuple[int, Sequence[int]]:
    output_shape = list(input.shape[:-1])
    num_patches = output_shape[dimension]
    size = input.size(-1)
    output_shape[dimension] = (num_patches - 1) * step + size
    return size, output_shape


class _CheckerboardFoldRoll(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        input: Tensor,
        dimension: int,
        size: int,
        step: int,
        output_shape: Sequence[int]
    ) -> Tensor:
        ctx.dimension = dimension
        ctx.size = size
        ctx.step = step
        
        # Checkerboard Algorithm
        # Sometimes faster than `F.fold` when `step` > 1
        output = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
        patch = output.unfold(dimension, size, step)

        # Checkerboard Partitioning
        # Calculate security step size, since as long as one block is taken every step, the extracted blocks will not physically overlap with each other.
        # `step = ceil(K / S)`
        # Generate all offset combinations (Offsets)
        offset_range = range(math.ceil(size / step))
        
        # Pre-defined full slice: `(slice(None), slice(None))` -> Batch, Channel
        base_slices = (slice(None),) * dimension
        for offset in offset_range:
            # Construct the slice object for the current iteration.
            full_slice = (*base_slices, slice(offset, None, step))
            
            # Due to the large step size, any two elements in that point to physical memory addresses that do not overlap.
            # Therefore, it can be safely do Inplace Add, which is extremely efficient.
            patch[full_slice].add_(input[full_slice])

        return output
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor):
        dimension = ctx.dimension
        size = ctx.size
        step = ctx.step
        
        if grad_output is None:
            return None, None, None, None, None
        
        grad_input = grad_output.unfold(dimension, size, step)
        return grad_input, None, None, None, None
_fold_roll_overlap_implementation = multidevices(_CheckerboardFoldRoll.apply)


# The `indices` may occupy a large amount of memory, but it may be eliminated by `torch.compile` when this is supported.
@compilable.compile(fullgraph=True)
def _fold_roll_overlap_indexing(input: Tensor, dimension: int, size: int, step: int, output_shape: Sequence[int]) -> Tensor:
    device = input.device
    
    output: Tensor = torch.zeros(output_shape, device=device, dtype=input.dtype)
    indices: Tensor = torch.arange(output.numel(), device=device).view(output_shape).unfold(dimension, size, step)
    output.view(-1).put_(indices, input, accumulate=True)
    return output
for device in compilable.compilable_gpus(): _fold_roll_overlap_implementation.register_device(device, _fold_roll_overlap_indexing)


def _fold_roll_overlap_cpu(input: Tensor, dimension: int, size: int, step: int, output_shape: Sequence[int]) -> Tensor:
    output: Tensor = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
    patch = output.unfold(dimension, size, step)
    patch.add_(input)
    return output
_fold_roll_overlap_implementation.register_device('cpu', _fold_roll_overlap_cpu)


def _fold_space(
    input: Tensor,
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    folder: Callable[[Tensor, Sequence[int]], Tensor]
) -> Tensor:
    double_spatial_ndim = input.ndim - 2
    if double_spatial_ndim % 2 != 0:
        raise ValueError(f"'input' is expected to be a tensor that was completely spatially unfolded")
    spatial_ndim = double_spatial_ndim // 2
    
    while dilation != 1:
        if isinstance(dilation, int):
            dilation = _spatialize_tuple(dilation, spatial_ndim)
        elif not any(value != 1 for value in dilation):
            break
        input = _dilate_patch(input, dilation, spatial_ndim)
        break
    
    stride = _spatialize_tuple(stride, spatial_ndim)
    
    output = folder(input, stride)
    
    if padding != 0:
        padding = _spatialize_tuple(padding, spatial_ndim)
        output = output[(..., *(slice(pad, -pad) for pad in padding))]
    
    return output


def _dilatedly_unfold(input: Tensor, dimension: int, size: int, step: int, dilation: int) -> Tensor:
    if dilation == 1:
        return unfold(input, dimension, size, step)
    effective_size = (size - 1) * dilation + 1
    return input.unfold(dimension, effective_size, step)[..., ::dilation]


def _dilate_patch(input: Tensor, dilation: Sequence[int], spatial_ndim: int) -> Tensor:
    shape = input.shape
    num_patches = shape[2:-spatial_ndim]
    patch_size =  shape[-spatial_ndim:]
    effective_patch_size = [(each_patch_size - 1) * each_dilation + 1 for each_patch_size, each_dilation in zip(patch_size, dilation)]
    def generate_index():
        dilation_length = len(dilation)
        for i, (size, expansion) in enumerate(zip(num_patches, dilation)):
            yield torch.arange(size, device=input.device).mul_(expansion)[(slice(None), *((None,) * (dilation_length - i - 1)))]
    expanded = input.new_zeros(*shape[:-spatial_ndim], *effective_patch_size)
    expanded[(..., *generate_index())] = input
    return expanded


def _spatialize_tuple(parameter: Union[int, Sequence[int]], spatial_ndim: int) -> tuple[int, ...]:
    if isinstance(parameter, int):
        parameter = (parameter,) * spatial_ndim
    else:
        parameter_length = len(parameter)
        if parameter_length < spatial_ndim:
            parameter = (*((1,) * spatial_ndim - parameter_length), *parameter)
    return parameter


class _CheckerboardFoldStack(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        input: Tensor,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        output_shape: Sequence[int]
    ) -> Tensor:
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        
        # Checkerboard Algorithm
        # Sometimes faster than `F.fold` when `stride` > 1
        output = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
        patch = output
        for dimension, (size, step) in enumerate(zip(kernel_size, stride), 2):
            patch = patch.unfold(dimension, size, step)

        # Checkerboard Partitioning
        # Calculate security step size, since as long as one block is taken every step, the extracted blocks will not physically overlap with each other.
        # `step = ceil(K / S)`
        steps = [math.ceil(size / step) for size, step in zip(kernel_size, stride)]
        
        # Generate all offset combinations (Offsets)
        offset_ranges = (range(step) for step in steps)
        
        # Pre-defined full slice: `(slice(None), slice(None))` -> Batch, Channel
        base_slices = (slice(None),) * 2
        for offsets in product(*offset_ranges):
            # Construct the slice object for the current iteration.
            full_slice = (*base_slices, *(slice(offset, None, step) for offset, step in zip(offsets, steps)))
            
            # Due to the large step size, any two elements in that point to physical memory addresses that do not overlap.
            # Therefore, it can be safely do Inplace Add, which is extremely efficient.
            patch[full_slice].add_(input[full_slice])

        return output
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor):
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        
        if grad_output is None:
            return None, None, None, None
        
        grad_input = grad_output
        for dimension, (size, step) in enumerate(zip(kernel_size, stride), 2):
            grad_input = grad_input.unfold(dimension, size, step)
        return grad_input, None, None, None
_fold_stack_overlap_implementation = multidevices(_CheckerboardFoldStack.apply)


# The `indices` may occupy a large amount of memory, but it may be eliminated by `torch.compile` when this is supported.
@compilable.compile(fullgraph=True)
def _fold_stack_overlap_indexing(input: Tensor, kernel_size: Sequence[int], stride: Sequence[int], output_shape: Sequence[int]) -> Tensor:
    device = input.device
    
    output: Tensor = torch.zeros(output_shape, device=device, dtype=input.dtype)
    indices: Tensor = torch.arange(output.numel(), device=device).view(output_shape)
    for dimension, (size, step) in enumerate(zip(kernel_size, stride), 2):
        indices = indices.unfold(dimension, size, step)
    output.view(-1).put_(indices, input, accumulate=True)
    return output
for device in compilable.compilable_gpus(): _fold_stack_overlap_implementation.register_device(device, _fold_stack_overlap_indexing)


def _fold_stack_overlap_cpu(input: Tensor, kernel_size: Sequence[int], stride: Sequence[int], output_shape: Sequence[int]) -> Tensor:
    output: Tensor = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
    patch = output
    for dimension, (size, step) in enumerate(zip(kernel_size, stride), 2):
        patch = patch.unfold(dimension, size, step)
    patch.add_(input)
    return output
_fold_stack_overlap_implementation.register_device('cpu', _fold_stack_overlap_cpu)


def _fold_space_overlap_parameters(input: Tensor, stride: Sequence[int]) -> Tuple[Sequence[int], Sequence[int]]:
    """
    Args:
        input (Tensor): [B, C, out_L_0, out_L_1, ..., out_L_n, k_L_0, k_L_1, ..., k_L_n]
    
    """
    double_spatial_ndim = input.ndim - 2
    if double_spatial_ndim % 2 != 0:
        raise ValueError(f"'input' is expected to be a tensor that was completely spatially unfolded")
    spatial_ndim = double_spatial_ndim // 2

    batch_size, num_channels, *other_shape = input.shape
    kernel_size = other_shape[-spatial_ndim:]
    output_shape = (batch_size, num_channels, *(
        (num_patches - 1) * step + size
        for num_patches, size, step in zip(other_shape[:-spatial_ndim], kernel_size, stride)
    ))
    return kernel_size, output_shape


def _fold_stack_overlap(input: Tensor, stride: Sequence[int]) -> Tensor:
    """
    Args:
        input (Tensor): [B, C, out_L_0, out_L_1, ..., out_L_n, k_L_0, k_L_1, ..., k_L_n]
    
    """
    kernel_size, output_shape = _fold_space_overlap_parameters(input, stride)
    return _fold_stack_overlap_implementation[input.device.type](input, kernel_size, stride, output_shape)


def _fold_space_overlap(input: Tensor, stride: Sequence[int]) -> Tensor:
    device = input.device
    spatial_shape = input.shape[2:]
    kernel_size, output_shape = _fold_space_overlap_parameters(input, stride)
    implementation = _fold_stack_overlap_implementation[device.type]
    ones: Tensor = implementation(
        torch.ones((1,) * (2 + len(spatial_shape)), device=device, dtype=input.dtype).expand(1, 1, *spatial_shape), kernel_size, stride, output_shape
    )
    output: Tensor = implementation(input, kernel_size, stride, output_shape)
    if output.requires_grad:
        return output / ones
    else:
        return output.div_(ones)