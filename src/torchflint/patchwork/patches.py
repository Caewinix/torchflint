from typing import Union, Sequence, Callable, Optional
import operator
import math
from numbers import Number
from functools import partial, reduce
from itertools import product, accumulate, chain
from cachetools import LRUCache
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
import torch.fx
from ..utils import compilable
from ..utils.devices import multidevices


def unfold(input: Tensor, dimension: int, size: int, step: int) -> Tensor:
    """
    Returns a view of the original tensor which contains all slices of size :attr:`size` from
    :attr:`input` tensor in the dimension :attr:`dimension`.

    Step between two slices is given by :attr:`step`.

    If `sizedim` is the size of dimension :attr:`dimension` for :attr:`input`, the size of
    dimension :attr:`dimension` in the returned tensor will be
    `(sizedim - size) / step + 1`.

    An additional dimension of size :attr:`size` is appended in the returned tensor.

    Args:
        input (Tensor): the tensor that is expected to be unfolded
        dimension (int): dimension in which unfolding happens
        size (int): the size of each slice that is unfolded
        step (int): the step between each slice
    
    Returns:
        output (Tensor): an unfolded view of the original tensor
    """
    return input.unfold(dimension, size, step)


def fold(input: Tensor, dimension: int, step: int) -> Tensor:
    """
    Returns a folded tensor by averaging slices along the specified dimension. This operation
    is one of the inverse of the :func:`unfold` operation, reconstructing the original tensor
    shape from its unfolded slices.

    The input tensor is assumed to contain an additional dimension of size. This additional
    dimension will be merged back into :attr:`dimension` using the given :attr:`step`.
    
    Args:
        input (Tensor): the tensor that has been unfolded at the given
            :attr:`dimension` using the given :attr:`step`
        dimension (int): the dimension along which folding happens,and
            the input tensor must have an additional dimension at last
        step (int): the step between each slice

    Returns:
        output (Tensor): a folded tensor that reconstructs the original shape prior to unfolding,
            the overlaps will be averaged
    """
    return _Fold.apply(input, dimension, step)


def fold_roll(input: Tensor, dimension: int, step: int) -> Tensor:
    """
    Returns a folded tensor by merging slices along the specified dimension. This operation
    is one of the inverse of the :func:`unfold` operation, reconstructing the original tensor
    shape from its unfolded slices.

    The input tensor is assumed to contain an additional dimension of size. This additional
    dimension will be merged back into :attr:`dimension` using the given :attr:`step`.
    
    Args:
        input (Tensor): the tensor that has been unfolded at the given
            :attr:`dimension` using the given :attr:`step`
        dimension (int): the dimension along which folding happens, and
            the input tensor must have an additional dimension at last
        step (int): the step between each slice

    Returns:
        output (Tensor): a folded tensor that reconstructs the original shape prior to unfolding,
            the overlaps will be added up
    """
    return _fold_roll_overlap_implementation[input.device.type](input, dimension, step)


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
    """
    Returns a view or a copy of the original tensor where all spatial dimensions are unfolded into patches according
    to the specified parameters. This operation extracts patches (local sliding blocks) from spatial dimensions.
    
    The input tensor should have the format where the first two dimensions are batch size and the number of channels,
    followed by all spatial dimensions. All spatial dimensions are unfolded independently, and additional dimensions
    of :attr:`kernel_size` will be appended for each spatial dimension in the returned tensor.
    
    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        kernel_size (int or ints): the size of the patches
        stride (int or ints): the stride of the patches in the input spatial dimensions
        padding (int or ints): implicit padding to be added on both sides of input. Default: 0
        dilation (int or ints): a parameter that controls the stride of elements within the neighborhood.
            Default: 1
        ceil_mode (bool): If `True`, will use ceil instead of floor to compute the output shape.
            This ensures that every element in the input tensor is covered by a sliding window. Default: False
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the tensor containing all extracted patches with shape
            `(batch_size, num_channels, num_patches_0, num_patches_1, ..., num_patches_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
    """
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
        single_unfold = Tensor.unfold
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
    output_padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Reconstructs the original spatial dimensions by folding patches, with averaging the overlapping
    regions if they exist, where the input tensor should be as the same layout as the output of :func:`unfold_space`,
    since this operation is one of the inverse of :func:`unfold_space`, merging the spatial patches
    back into the original spatial layout.
    
    This function folds the patches back into spatial dimensions, averaging values in
    overlapping regions according to the specified stride.
    
    Args:
        input (Tensor): input tensor of shape
            `(batch_size, num_channels, num_patches_0, num_patches_1, ..., num_patches_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        stride (int or ints): the stride of the patches in the input spatial dimensions
        padding (int or ints): implicit padding to be removed on both sides of output. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        dilation (int or ints): a parameter that controls the stride of elements within the neighborhood.
            Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the reconstructed tensor with shape
            `(batch_size, num_channels, length_0, length_1, ..., length_n)`
    """
    return _fold_space(input, stride, padding, output_padding, dilation, default_space=default_space, folder=_FoldSpace.apply)


def fold_stack(
    input: Tensor,
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Reconstructs the original spatial dimensions by folding the patches, with aggregating the overlapping
    regions if they exist, where the input tensor should be as the same layout as the output of :func:`unfold_space`,
    since this operation is one of the inverse of :func:`unfold_space`, merging the spatial patches
    back into the original spatial layout.
    
    This function folds the patches back into spatial dimensions, aggregating values in
    overlapping regions according to the specified stride.
    
    Args:
        input (Tensor): input tensor of shape
            `(batch_size, num_channels, num_patches_0, num_patches_1, ..., num_patches_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        stride (int or ints): the stride of the patches in the input spatial dimensions
        padding (int or ints): implicit padding to be removed on both sides of output. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        dilation (int or ints): a parameter that controls the stride of elements within the neighborhood.
            Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the reconstructed tensor with shape
            `(batch_size, num_channels, length_0, length_1, ..., length_n)`
    """
    return _fold_space(input, stride, padding, output_padding, dilation, default_space=default_space, folder=_fold_stack_overlap)


def patches_column(input: Tensor) -> Tensor:
    """
    Reshapes the unfolded patches into the column format, where the input tensor should be as 
    the same layout as the output of :func:`unfold_space`.
    
    This function flattens the kernel dimensions into the channel dimension, and flattens
    all spatial patch dimensions into a single dimension, producing a standard matrix
    layout commonly used in linear layers and convolution implementations.
    
    Args:
        input (Tensor): input tensor containing unfolded patches, expected to have shape
            `(batch_size, num_channels, num_patches_0, num_patches_1, ..., num_patches_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
    
    Returns:
        output (Tensor): the reshaped tensor in column format with shape
            `(batch_size, num_channels * kernel_size_0 * kernel_size_1 * ... * kernel_size_n, num_patches_0 * num_patches_1 * ... * num_patches_n)`
    """
    double_spatial_ndim = input.ndim - 2
    if double_spatial_ndim % 2 != 0:
        raise ValueError(f"'input' is expected to be a tensor that was completely spatially unfolded")
    spatial_ndim = double_spatial_ndim // 2
    
    output = input.permute(0, 1, *range(2 + spatial_ndim, input.ndim), *range(2, 2 + spatial_ndim))
    output_shape = output.shape
    return output.reshape(output_shape[0], -1, math.prod(output_shape[-spatial_ndim:]))


def kernel_keepdim(input: Tensor) -> Tensor:
    """
    Maintains kernel dimensions structure of the input tensor by appending singleton dimensions to
    the end of the tensor, ensuring compatibility with patch operations that expect explicit kernel dimensions.
    
    This function adds singleton dimensions (size `1`) at the end of the tensor, matching the
    number of spatial dimensions. This is useful when preparing tensors for operations that
    expect the `(..., kernel_size_0, kernel_size_1, ..., kernel_size_n)` format, but the
    kernel dimensions have been flattened or reduced.
    
    Args:
        input (Tensor): input tensor of shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n)`
    
    Returns:
        output (Tensor): the dimensions kept tensor with shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n, 1 (0), 1 (1), ..., 1 (n))`
    """
    spatial_ndim = input.ndim - 2
    return input.view(*input.shape, *((1,) * spatial_ndim))


def kernel_expand(input: Tensor, kernel_size: Sequence[int]) -> Tensor:
    """
    Expands the input tensor by appending and expanding kernel dimensions, creating explicit
    kernel dimensions with specified sizes. This operation transforms tensors into the format
    expected by patch-based operations.
    
    The function first appends singleton dimensions to match the number of spatial dimensions,
    then expands these singleton dimensions to the specified kernel sizes. This is useful for
    converting tensors that lack explicit kernel dimensions into the standard patch format
    `(..., kernel_size_0, kernel_size_1, ..., kernel_size_n)`.
    
    Args:
        input (Tensor): input tensor of shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n)`
        kernel_size (ints): sizes of the kernel dimensions to create. The length must
            equal the number of spatial dimensions in the input tensor.
    
    Returns:
        output (Tensor): the expanded tensor with shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
    """
    spatial_ndim = input.ndim - 2
    output = input.view(*input.shape, *((1,) * spatial_ndim))
    if spatial_ndim != len(kernel_size):
        raise ValueError(f"the length of 'kernel_size' should be equal to the number of spatial dimensions of 'input'")
    output = output.expand(*((-1,) * input.ndim), *kernel_size)
    return output


class _FoldRoll(Function):
    @classmethod
    def initialize(cls, ctx: FunctionCtx, input: Tensor, dimension: int, step: int) -> Tensor:
        ctx.dimension = dimension
        ctx.step = step
        output_shape = list(input.shape[:-1])
        num_patches = output_shape[dimension]
        size = input.size(-1)
        ctx.size = size
        output_shape[dimension] = (num_patches - 1) * step + size
        return output_shape
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Optional[Tensor]):
        if grad_output is None:
            return None, None, None
        
        dimension = ctx.dimension
        size = ctx.size
        step = ctx.step
        
        grad_input = grad_output.unfold(dimension, size, step)
        return grad_input, None, None


class _CheckerboardFoldRoll(_FoldRoll):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, dimension: int, step: int) -> Tensor:
        output_shape = __class__.initialize(ctx, input, dimension, step)
        size = ctx.size
        
        # Checkerboard Algorithm
        # Sometimes faster than `F.fold` when `step` > 1
        output: Tensor = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
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
_fold_roll_overlap_implementation = multidevices(_CheckerboardFoldRoll.apply)


def _create_term(
    graph: torch.fx.Graph,
    shape_node: torch.fx.Node,
    device_node: torch.fx.Node,
    index: int,
    factor: int,
    input_ndim: int
) -> torch.fx.Node:
    dim_size = graph.call_function(operator.getitem, (shape_node, index))
    view_shape = [1] * input_ndim
    view_shape[index] = dim_size
    arange = graph.call_function(torch.arange, (dim_size,), {"device" : device_node})
    view = graph.call_method("view", (arange, view_shape))
    return graph.call_function(operator.mul, (view, factor))


# The `indices` may occupy a large amount of memory, but it may be eliminated by `torch.compile` when this is supported.
def _fold_roll_overlap_indexing_graph(dimension: int, step: int, input_ndim: int) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    
    input_node = graph.placeholder("input")
    shape_node = graph.call_function(getattr, (input_node, "shape"))
    shape_nodes = [graph.call_function(operator.getitem, (shape_node, i)) for i in range(input_ndim)]
    
    num_patches_node = shape_nodes[dimension]
    kernel_size_node = shape_nodes[-1]
    
    # Output Spatial Size: (L - 1) * step + K
    output_spatial_node = graph.call_function(operator.add, (
        graph.call_function(operator.mul, (
            graph.call_function(operator.sub, (num_patches_node, 1)), 
            step
        )),
        kernel_size_node
    ))
    
    output_shape_nodes = shape_nodes[:-1]
    output_shape_nodes[dimension] = output_spatial_node
    output_stride = list(accumulate(output_shape_nodes[:0:-1], lambda left, right: graph.call_function(operator.mul, (left, right)), initial=1))[::-1]
    
    device_node = graph.call_function(getattr, (input_node, "device"))
    dtype_node = graph.call_function(getattr, (input_node, "dtype"))
    
    output_node = graph.call_function(torch.zeros, (output_shape_nodes,), {"device": device_node, "dtype": dtype_node})
    
    # Calculate linear factor
    indices_factor = (
        *output_stride[:dimension],
        graph.call_function(operator.mul, (output_stride[dimension], step)),
        *output_stride[dimension+1:],
        output_stride[dimension]
    )
    
    indices_node = reduce(
        lambda left, right: graph.call_function(operator.add, (left, right)),
        map(partial(_create_term, graph, shape_node, device_node, input_ndim=input_ndim), range(input_ndim), indices_factor)
    )
    
    graph.call_method("put_", (
        graph.call_method("flatten", (output_node,)),
        indices_node,
        input_node
    ), {"accumulate": True})
    
    graph.output(output_node)
    return torch.fx.GraphModule({}, graph)


class _IndexingFoldRoll(_FoldRoll):
    _cache = LRUCache(maxsize=compilable.recompile_limit())
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, dimension: int, step: int) -> Tensor:
        input_ndim = input.ndim
        
        key = (dimension, step, input_ndim)
        cache = __class__._cache
        try:
            compiled_function = cache[key]
        except:
            compiled_function = compilable.compile(_fold_roll_overlap_indexing_graph(dimension, step, input_ndim), fullgraph=True)
            cache[key] = compiled_function
        output: Tensor = compiled_function(input)
        
        ctx.dimension = dimension
        ctx.size = input.size(-1)
        ctx.step = step
        
        return output
for device in compilable.compilable_gpus(): _fold_roll_overlap_implementation.register_device(device, _IndexingFoldRoll.apply)


class _CpuFoldRoll(_FoldRoll):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, dimension: int, step: int) -> Tensor:
        output_shape = __class__.initialize(ctx, input, dimension, step)
        
        output: Tensor = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
        patch = output.unfold(dimension, ctx.size, step)
        patch.add_(input)
        return output
_fold_roll_overlap_implementation.register_device('cpu', _CpuFoldRoll.apply)


class _Fold(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, dimension: int, step: int) -> Tensor:
        device = input.device
        input_ndim = input.ndim

        implementation = _fold_roll_overlap_implementation[device.type]
        expansion = [1] * input_ndim
        ones = torch.ones(expansion, device=device, dtype=input.dtype)
        expansion[dimension] = input.size(dimension)
        expansion[-1] = input.size(-1)
        ones = ones.expand(expansion)
        count: Tensor = implementation(
            ones, dimension, step
        )
        count.clamp_min_(1)
        output: Tensor = implementation(input, dimension, step)
        output.div_(count)
        
        ctx.save_for_backward(count)
        ctx.dimension = dimension
        ctx.size = input.size(-1)
        ctx.step = step
        
        return output
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Optional[Tensor]):
        if grad_output is None:
            return None, None, None
        
        count: Tensor = ctx.saved_tensors[0]
        dimension = ctx.dimension
        size = ctx.size
        step = ctx.step
        
        grad_input = (grad_output / count).unfold(dimension, size, step)
        return grad_input, None, None


def _fold_space(
    input: Tensor,
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0,
    folder: Callable[[Tensor, Sequence[int]], Tensor]
) -> Tensor:
    double_spatial_ndim = input.ndim - 2
    if double_spatial_ndim % 2 != 0:
        raise ValueError(f"'input' is expected to be a tensor that was completely spatially unfolded")
    spatial_ndim = double_spatial_ndim // 2
    
    stride = _spatialize_tuple(stride, spatial_ndim)
    
    while output_padding != 0:
        if isinstance(output_padding, int):
            output_slices = (slice(None), slice(None), *(slice(output_padding - step) for step in stride))
            output_padding = list(chain((0 for _ in range(spatial_ndim * 2)), (pad_value for _ in range(spatial_ndim) for pad_value in (0, 1))))
        elif any(value != 0 for value in output_padding):
            output_slices = (slice(None), slice(None), *(slice(None) if pad == 0 else slice(pad - step) for pad, step in zip(output_padding, stride)))
            output_padding = list(chain((0 for _ in range(spatial_ndim * 2)), (pad_value for value in reversed(output_padding) for pad_value in (0, 1 if value > 0 else 0))))
        else:
            output_slices = None
            break
        if isinstance(default_space, Number):
            functional_pad = partial(F.pad, mode='constant', value=default_space)
        else:
            functional_pad = partial(F.pad, mode=default_space)
        input = functional_pad(input, output_padding)
        break
    else:
        output_slices = None
    
    while dilation != 1:
        if isinstance(dilation, int):
            dilation = _spatialize_tuple(dilation, spatial_ndim)
        elif not any(value != 1 for value in dilation):
            break
        input = _dilate_patch(input, dilation, spatial_ndim)
        break
    
    output = folder(input, stride)
    
    if padding != 0:
        padding = _spatialize_tuple(padding, spatial_ndim)
        output = output[(..., *(slice(pad, -pad) for pad in padding))]
    
    if output_slices is not None:
        output = output[output_slices]
    
    return output


def _dilatedly_unfold(input: Tensor, dimension: int, size: int, step: int, dilation: int) -> Tensor:
    if dilation == 1:
        return input.unfold(dimension, size, step)
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


class _FoldStack(Function):
    @classmethod
    def initialize(cls, ctx: FunctionCtx, input: Tensor, stride: Sequence[int]):
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
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        return output_shape
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Optional[Tensor]):
        if grad_output is None:
            return None, None
        
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        
        grad_input = grad_output
        for dimension, (size, step) in enumerate(zip(kernel_size, stride), 2):
            grad_input = grad_input.unfold(dimension, size, step)
        return grad_input, None


class _CheckerboardFoldStack(_FoldStack):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, stride: Sequence[int]) -> Tensor:
        output_shape = __class__.initialize(ctx, input, stride)
        kernel_size = ctx.kernel_size
        
        # Checkerboard Algorithm
        # Sometimes faster than `F.fold` when `stride` > 1
        output: Tensor = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
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
_fold_stack_overlap_implementation = multidevices(_CheckerboardFoldStack.apply)


# The `indices` may occupy a large amount of memory, but it may be eliminated by `torch.compile` when this is supported.
def _fold_stack_overlap_indexing_graph(stride: Sequence[int], input_ndim: int) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    
    input_node = graph.placeholder("input")
    shape_node = graph.call_function(getattr, (input_node, "shape"))
    shape_nodes = [graph.call_function(operator.getitem, (shape_node, i)) for i in range(input_ndim)]
    
    spatial_ndim = (input_ndim - 2) // 2
    watershed_dimension = 2 + spatial_ndim
    
    num_patches_nodes = shape_nodes[2:watershed_dimension]
    kernel_size_nodes = shape_nodes[watershed_dimension:]

    output_spatial_nodes = [
        graph.call_function(operator.add, (graph.call_function(operator.mul, (graph.call_function(operator.sub, (num_patches_node, 1)), step)), size_node))
        for num_patches_node, size_node, step in zip(num_patches_nodes, kernel_size_nodes, stride)
    ]
    output_shape = shape_nodes[:2] + output_spatial_nodes
    output_stride = list(accumulate(output_shape[:0:-1], lambda left, right: graph.call_function(operator.mul, (left, right)), initial=1))[::-1]
    
    device_node = graph.call_function(getattr, (input_node, "device"))
    dtype_node = graph.call_function(getattr, (input_node, "dtype"))
    
    output_node = graph.call_function(torch.zeros, (output_shape,), {"device": device_node, "dtype": dtype_node})
    
    # Calculate linear factor
    indices_factor = (
        *output_stride[:2], 
        *(graph.call_function(operator.mul, (output_step, step)) for output_step, step in zip(output_stride[2:], stride)), 
        *output_stride[2:]
    )
    indices_node = reduce(
        lambda left, right: graph.call_function(operator.add, (left, right)),
        map(partial(_create_term, graph, shape_node, device_node, input_ndim=input_ndim), range(input_ndim), indices_factor)
    )
    
    graph.call_method("put_", (
        graph.call_method("flatten", (output_node,)),
        indices_node,
        input_node
    ), {"accumulate": True})
    
    graph.output(output_node)
    return torch.fx.GraphModule({}, graph)


class _IndexingFoldStack(_FoldStack):
    _cache = LRUCache(maxsize=compilable.recompile_limit())
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, stride: Sequence[int]) -> Tensor:
        input_ndim = input.ndim
        
        key = (*stride, input_ndim)
        cache = __class__._cache
        try:
            compiled_function = cache[key]
        except:
            compiled_function = compilable.compile(_fold_stack_overlap_indexing_graph(stride, input_ndim), fullgraph=True)
            cache[key] = compiled_function
        output: Tensor = compiled_function(input)
        
        ctx.kernel_size = input.shape[output.ndim - input_ndim:]
        ctx.stride = stride

        return output
for device in compilable.compilable_gpus(): _fold_stack_overlap_implementation.register_device(device, _IndexingFoldStack.apply)


class _CpuFoldStack(_FoldStack):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, stride: Sequence[int]) -> Tensor:
        output_shape = __class__.initialize(ctx, input, stride)

        output: Tensor = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
        patch = output
        for dimension, (size, step) in enumerate(zip(ctx.kernel_size, stride), 2):
            patch = patch.unfold(dimension, size, step)
        patch.add_(input)
        return output
_fold_stack_overlap_implementation.register_device('cpu', _CpuFoldStack.apply)


def _fold_stack_overlap(input: Tensor, stride: Sequence[int]) -> Tensor:
    return _fold_stack_overlap_implementation[input.device.type](input, stride)


class _FoldSpace(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, stride: Sequence[int]) -> Tensor:
        device = input.device
        spatial_shape_with_kernel_size = input.shape[2:]
        implementation = _fold_stack_overlap_implementation[device.type]
        count: Tensor = implementation(
            torch.ones(
                (1,) * (2 + len(spatial_shape_with_kernel_size)),
                device=device,
                dtype=input.dtype
            ).expand(1, 1, *spatial_shape_with_kernel_size),
            stride
        )
        count.clamp_min_(1)
        
        output: Tensor = implementation(input, stride)
        output.div_(count)
        
        ctx.save_for_backward(count)
        ctx.kernel_size = input.shape[output.ndim - input.ndim:]
        ctx.stride = stride
        
        return output
    
    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Optional[Tensor]):
        if grad_output is None:
            return None, None
        
        count, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        
        grad_input: Tensor = grad_output / count
        for dimension, (size, step) in enumerate(zip(kernel_size, stride), 2):
            grad_input = grad_input.unfold(dimension, size, step)
        return grad_input, None