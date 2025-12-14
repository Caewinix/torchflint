from typing import Callable, Sequence, Optional, Union
from functools import reduce
from itertools import chain
from numbers import Number
import torch
from torch import Tensor
from .nn import Buffer


def buffer(tensor: Optional[Tensor], persistent: bool = True) -> Buffer:
    """
    Creates a buffer wrapper for a tensor, enabling transparent registration as a module buffer.
    
    This function wraps a tensor in a :class:`Buffer` container that can be automatically
    recognized and registered as a buffer in PyTorch modules.
    The buffer can be made persistent (saved in state dict) or non-persistent as needed.
    
    Args:
        tensor (Tensor, optional): the tensor to wrap as a buffer. If `None`, creates
            an empty buffer
        persistent (bool): whether the buffer should be persistent (saved in state dict).
            Default: True
    
    Returns:
        output (Buffer): a buffer-wrapped tensor that can be registered to modules
    
    Examples:
        >>> import torch
        >>> import torchflint
        >>>
        >>> class MyModule(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.my_buffer = torchflint.buffer(torch.randn(3, 4), persistent=True)
        >>>
        >>> my_module = MyModule()
        >>> my_module.my_buffer.shape
        torch.Size([3, 4])
    """
    return Buffer(tensor, persistent)


def promote_types(*dtypes: torch.dtype) -> torch.dtype:
    """
    Promotes multiple data types to a common type that can represent all data types.
    
    This function finds a common dtype that can safely represent values from all
    input dtypes, following PyTorch's type promotion rules. The promotion follows
    the same semantics as :func:`torch.promote_types` but for multiple arguments.
    
    Args:
        *dtypes (torch.dtype): variable number of dtypes to promote
    
    Returns:
        output (torch.dtype): the promoted dtype that can represent all inputs
    
    Examples:
        >>> promote_types(torch.int32, torch.float32)
        torch.float32
        >>> promote_types(torch.int8, torch.int16, torch.int32)
        torch.int32
    """
    return reduce(torch.promote_types, dtypes)


def map_range(
    input: Tensor,
    interval: Sequence[int] = (0, 1),
    dim: Union[int, Sequence[int], None] = None,
    dtype: torch.dtype = None,
    scalar_default: Optional[str] = 'max',
    eps: float = 1.e-6
) -> Tensor:
    """
    Linearly maps input values to a specified output range for specific dimension(s).
    
    This function computes the minimum and maximum values along specified dimension(s)
    and linearly transforms the input to the target interval. For constant inputs
    (where minimum == maximum), a default value is assigned based on the :attr:`scalar_default`
    parameter.
    
    Args:
        input (Tensor): input tensor to normalize
        interval (pair of ints): target interval for the output values.
            Default: (0, 1)
        dim (int or ints, optional): dimension(s) along which to compute
            min and max. If `None`, uses all dimensions
        dtype (torch.dtype, optional): desired data type for output tensor
        scalar_default (str, optional): default value for constant inputs.
            Options: 'max' (1 - :attr:`eps`), 'min' (:attr:`eps`), 'none' (0.5), or None (keep as 0).
            Default: 'max'
        eps (float): small epsilon value used in default calculations.
            Default: 1.e-6
    
    Returns:
        output (Tensor): normalized tensor with values in the specified interval
    """
    min_value: Tensor = amin(input, dim=dim, keepdim=True)
    max_value: Tensor = amax(input, dim=dim, keepdim=True)
    max_min_difference = max_value - min_value
    max_min_equal_mask = max_min_difference == 0
    max_min_difference.masked_fill_(max_min_equal_mask, 1)
    input = input - min_value
    if not (scalar_default is None or scalar_default == 'none'):
        input.masked_fill_(max_min_equal_mask, torch.tensor(_scalar_default_value(scalar_default, eps)).to(input.dtype))
    return (input / max_min_difference * (interval[1] - interval[0]) + interval[0]).to(dtype)


def map_ranges(
    input: Tensor,
    intervals: Sequence[Sequence[int]] = [(0, 1)],
    dim: Union[int, Sequence[int], None] = None,
    dtype: Optional[torch.dtype] = None,
    scalar_default: Optional[str] = 'max',
    eps: float = 1e-6
) -> Tensor:
    """
    Maps input values to multiple output ranges simultaneously, broadcasting results.
    
    This function applies :func:`map_range` to the same input for multiple target
    intervals, producing a batched output where the last dimension contains the
    results for each interval. This is useful for generating multiple normalized
    versions of the same data in a single operation.
    
    Args:
        input (Tensor): input tensor to normalize
        intervals (many pairs of ints): list of target intervals for the output values.
            Default: [(0, 1)]
        dim (int or ints, optional): dimension(s) along which to compute
            min and max. If `None`, uses all dimensions
        dtype (torch.dtype, optional): desired data type for output tensor
        scalar_default (str, optional): default value for constant inputs.
            Options: 'max' (1 - :attr:`eps`), 'min' (:attr:`eps`), 'none' (0.5), or None (keep as 0).
            Default: 'max'
        eps (float): small epsilon value used in default calculations.
            Default: 1.e-6
    
    Returns:
        output (Tensor): batched normalized tensor where the first dimension corresponds
            to different intervals
    """
    intervals_length = len(intervals)
    min_value: Tensor = amin(input, dim=dim, keepdim=True)
    max_value: Tensor = amax(input, dim=dim, keepdim=True)
    max_min_difference = max_value - min_value
    max_min_equal_mask = max_min_difference == 0
    max_min_difference.masked_fill_(max_min_equal_mask, 1)
    input = input - min_value
    if not (scalar_default is None or scalar_default == 'none'):
        input.masked_fill_(max_min_equal_mask, torch.tensor(_scalar_default_value(scalar_default, eps)).to(input.dtype))
    normed = (input / max_min_difference).to(dtype=dtype)[..., None].expand(*((-1,) * input.ndim), intervals_length)
    intervals: Tensor = torch.tensor(intervals, device=normed.device, dtype=normed.dtype)
    return (normed * intervals.diff(dim=1)[..., 0] + intervals[..., 0]).permute(input.ndim, *range(input.ndim))


try:
    amin = torch.amin
    amax = torch.amax
except AttributeError:
    def amin(
        input: Tensor,
        dim: Union[int, Sequence[int], None] = (),
        keepdim: bool = False,
        *,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        """
        amin(input, dim, keepdim=False, *, out=None) -> Tensor

        Returns the minimum value of each slice of the :attr:`input` tensor in the given
        dimension(s) :attr:`dim`.

        .. note::
            The difference between ``max``/``min`` and ``amax``/``amin`` is:
                - ``amax``/``amin`` supports reducing on multiple dimensions,
                - ``amax``/``amin`` does not return indices.

            Both ``amax``/``amin`` evenly distribute gradients between equal values
            when there are multiple input elements with the same minimum or maximum value.

            For ``max``/``min``:
                - If reduce over all dimensions(no dim specified), gradients evenly distribute between equally ``max``/``min`` values.
                - If reduce over one specified axis, only propagate to the indexed element.


        If :attr:`keepdim` is ``True``, the output tensor is of the same size
        as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
        Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
        output tensor having 1 (or ``len(dim)``) fewer dimension(s).


        Args:
            input (Tensor): the input tensor.

            dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
                If ``None``, all dimensions are reduced.


            keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.


        Keyword args:
        out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[ 0.6451, -0.4866,  0.2987, -1.3312],
                    [-0.5744,  1.2980,  1.8397, -0.2713],
                    [ 0.9128,  0.9214, -1.7268, -0.2995],
                    [ 0.9023,  0.4853,  0.9075, -1.6165]])
            >>> torchflint.amin(a, 1)
            tensor([-1.3312, -0.5744, -1.7268, -1.6165])
        """
        return _a_min_max(torch.min, input, dim, keepdim, out=out)


    def amax(
        input: Tensor,
        dim: Union[int, Sequence[int], None] = (),
        keepdim: bool = False,
        *,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        """
        amax(input, dim, keepdim=False, *, out=None) -> Tensor

        Returns the maximum value of each slice of the :attr:`input` tensor in the given
        dimension(s) :attr:`dim`.

        .. note::
            The difference between ``max``/``min`` and ``amax``/``amin`` is:
                - ``amax``/``amin`` supports reducing on multiple dimensions,
                - ``amax``/``amin`` does not return indices.

            Both ``amax``/``amin`` evenly distribute gradients between equal values
            when there are multiple input elements with the same minimum or maximum value.

            For ``max``/``min``:
                - If reduce over all dimensions(no dim specified), gradients evenly distribute between equally ``max``/``min`` values.
                - If reduce over one specified axis, only propagate to the indexed element.


        If :attr:`keepdim` is ``True``, the output tensor is of the same size
        as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
        Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
        output tensor having 1 (or ``len(dim)``) fewer dimension(s).


        Args:
            input (Tensor): the input tensor.

            dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
                If ``None``, all dimensions are reduced.


            keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.


        Keyword args:
        out (Tensor, optional): the output tensor.

        Example::

            >>> a = torch.randn(4, 4)
            >>> a
            tensor([[ 0.8177,  1.4878, -0.2491,  0.9130],
                    [-0.7158,  1.1775,  2.0992,  0.4817],
                    [-0.0053,  0.0164, -1.3738, -0.0507],
                    [ 1.9700,  1.1106, -1.0318, -1.0816]])
            >>> torchflint.amax(a, 1)
            tensor([1.4878, 2.0992, 0.0164, 1.9700])
        """
        return _a_min_max(torch.max, input, dim, keepdim, out=out)
    

    def _a_min_max(
        min_max_func,
        input: torch.Tensor,
        dim: Sequence[int],
        keepdim=False,
        *,
        out: Optional[Tensor] = None
    ):
        if dim is None:
            return min_max_func(input)
        elif isinstance(dim, int):
            dim = (dim,)
        ndim = input.ndim
        dim_length = len(dim)
        middle_index = ndim - dim_length
        tailing_dim, inv_dim = _tailing_dim(ndim, dim)
        input_for_func = input.permute(tailing_dim).flatten(middle_index)
        if out is None:
            out = min_max_func(input_for_func, dim=middle_index)[0]
        else:
            _ = torch.empty_like(out, device=out.device)
            min_max_func(input_for_func, dim=middle_index, out=(out, _))
        if keepdim:
            out = out.view((*out.shape,) + (1,) * len(dim))
            out = out.permute(inv_dim)
        return out


    def _tailing_dim(ndim: int, dim: Sequence[int]):
        inv_dim = [0] * ndim
        def generate_dims():
            nonlocal inv_dim
            for permute_back_dim, permuted_dim in enumerate(chain((d for d in range(ndim) if d not in dim), dim)):
                inv_dim[permuted_dim] = permute_back_dim
                yield permuted_dim
        tailing_dim = type(dim)(generate_dims())
        return tailing_dim, inv_dim


min = amin
max = amax
imin = torch.min
imax = torch.max


def is_int(dtype: torch.dtype) -> bool:
    """
    Checks if a dtype represents integer data.
    
    Args:
        dtype (torch.dtype): the dtype to check
    
    Returns:
        output (bool): `True` if the dtype is an integer type, `False` otherwise
    
    Examples:
        >>> is_int(torch.int32)
        True
        >>> is_int(torch.float32)
        False
    """
    try:
        torch.iinfo(dtype)
        return True
    except TypeError:
        return False


def is_float(dtype: torch.dtype) -> bool:
    """
    Checks if a dtype represents floating-point data.
    
    Args:
        dtype (torch.dtype): the dtype to check
    
    Returns:
        output (bool): `True` if the dtype is a floating-point type, `False` otherwise
    
    Examples:
        >>> is_float(torch.float32)
        True
        >>> is_float(torch.int32)
        False
    """
    return dtype.is_floating_point


def invert(input: Tensor, dim: Union[int, Sequence[int], None] = None) -> Tensor:
    """
    Inverts tensor values within their local range along specified dimensions.
    
    This function computes the minimum and maximum values along the specified
    dimension(s) and maps each value to its opposite position within the local
    range: `output = max - input + min`. The operation preserves the relative
    ordering of values but reverses their positions within the local range.
    
    Args:
        input (Tensor): input tensor to invert
        dim (int or Sequence[int]): dimension(s) along which to compute the
            local range for inversion. If `None`, all dimensions will be specified.
            Default: None
    
    Returns:
        output (Tensor): inverted tensor with same shape as input
    
    Examples:
        >>> x = torch.tensor([[1.0, 2.0, 3.0],
        ...                   [4.0, 5.0, 6.0]])
        >>> 
        >>> # Invert along columns (dim=1)
        >>> invert(x, dim=1)
        tensor([[3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0]])
        >>> 
        >>> # Invert along rows (dim=0)
        >>> invert(x, dim=0)
        tensor([[4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0]])
        >>> 
        >>> # Invert along both dimensions
        >>> invert(x, dim=(0, 1))
        tensor([[6.0, 5.0, 4.0],
                [3.0, 2.0, 1.0]])
    
    Note:
        The inversion formula `max - input + min` ensures that the minimum value
        becomes the maximum, the maximum becomes the minimum, and intermediate
        values are linearly mapped to their opposite positions in the range.
    """
    min_values = amin(input, dim=dim, keepdim=True)
    max_values = amax(input, dim=dim, keepdim=True)
    return max_values - input + min_values


def invert_(input: Tensor, dim: Union[int, Sequence[int], None] = None) -> Tensor:
    """
    Inverts tensor values within their local range along specified dimensions in-place.
    
    This function computes the minimum and maximum values along the specified
    dimension(s) and maps each value to its opposite position within the local
    range: `output = max - input + min`. The operation preserves the relative
    ordering of values but reverses their positions within the local range.
    
    Args:
        input (Tensor): input tensor to invert
        dim (int or Sequence[int]): dimension(s) along which to compute the
            local range for inversion. If `None`, all dimensions will be specified.
            Default: None
    
    Returns:
        output (Tensor): inverted tensor with same shape and object as input
    
    Note:
        This is an in-place version of :func:`invert`.
        The inversion formula `max - input + min` ensures that the minimum value
        becomes the maximum, the maximum becomes the minimum, and intermediate
        values are linearly mapped to their opposite positions in the range.
    """
    min_values = amin(input, dim=dim, keepdim=True)
    max_values = amax(input, dim=dim, keepdim=True)
    return input.neg_().add_(max_values.add_(min_values))


def shift(input: Tensor, shift: Tensor, fill_value: Number = 0) -> Tensor:
    """
    Shifts tensor content according to per-channel and per-batch spatial offsets.
    
    This function applies spatial shifts to the input tensor, moving content according
    to the shift tensor. Empty regions created by the shift are filled with
    :attr:`fill_value`.
    
    The shift tensor should have shape `(batch_size, num_channels, ndim)` where ndim is the number of
    spatial dimensions, specifying the integer shift for each batch and channel.
    
    Args:
        input (Tensor): input tensor to shift
        shift (Tensor): shift tensor of shape `(batch_size, num_channels, ndim)` specifying spatial offsets
        fill_value (Number): value to fill empty regions after shifting.
            Default: 0
    
    Returns:
        output (Tensor): the shifted input tensor
    """
    out = torch.full_like(input, fill_value)
    destination_flat, source_flat, mask = _shift_arguments(input, shift)
    out.view(-1)[destination_flat[mask]] = input.view(-1)[source_flat[mask]]
    return out


def shift_(input: Tensor, shift: Tensor, fill_value: Number = 0) -> Tensor:
    """
    Shifts tensor content in-place according to per-channel and per-batch spatial offsets.
    
    This function applies spatial shifts to the input tensor, moving content according
    to the shift tensor. Empty regions created by the shift are filled with
    :attr:`fill_value`. The operation is performed in-place, modifying the input tensor.
    
    The shift tensor should have shape `(batch_size, num_channels, ndim)` where ndim is the number of
    spatial dimensions, specifying the integer shift for each batch and channel.
    
    Args:
        input (Tensor): input tensor to shift in-place
        shift (Tensor): shift tensor of shape `(batch_size, num_channels, ndim)` specifying spatial offsets
        fill_value (Number): value to fill empty regions after shifting.
            Default: 0
    
    Returns:
        output (Tensor): the shifted input tensor (same object, modified in-place)
    """
    destination_flat, source_flat, mask = _shift_arguments(input, shift)
    input.view(-1)[destination_flat[mask]] = input.view(-1)[source_flat[mask]]
    input.view(-1)[destination_flat[~mask]] = fill_value
    return input


def linspace(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    steps: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> Tensor:
    """
    Creates tensor of evenly spaced values between start and end.
    
    This function generates a sequence of :attr:`steps` values linearly interpolated
    between :attr:`start` and :attr:`end`. The result includes both endpoints.
    
    Args:
        start (Number or Tensor): the starting value of the sequence
        end (Number or Tensor): the ending value of the sequence
        steps (int): number of values to generate. Must be non-negative
        dtype (torch.dtype, optional): desired data type of returned tensor
        device (torch.device, optional): desired device of returned tensor
    
    Returns:
        output (Tensor): tensor of :attr:`steps` values from start to end
    """
    if steps == 0:
        return torch.tensor([], dtype=dtype, device=device)
    else:
        start = torch.as_tensor(start, dtype=dtype, device=device)
        if steps == 1:
            return start
        else:
            if steps < 0:
                raise RuntimeError("number of steps must be non-negative")
            end = torch.as_tensor(end, dtype=dtype, device=device)
            common_difference = torch.as_tensor((end - start) / (steps - 1))
            index = torch.arange(steps).to(common_difference.device)
            return start + common_difference * index.view(*index.shape, *((1,) * len(common_difference.shape)))


def linspace_at(
    index: int,
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    steps: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> Tensor:
    """
    Computes a single value from a linear sequence at the specified index.
    
    This function computes the value at position :attr:`index` in a linear sequence
    from :attr:`start` to :attr:`end` with :attr:`steps` points, without generating
    the full sequence. This is more memory-efficient than :func:`linspace` when only
    a single point is needed.
    
    Args:
        index (int): position in the sequence to compute (0 <= index < steps)
        start (Number or Tensor): the starting value of the sequence
        end (Number or Tensor): the ending value of the sequence
        steps (int): total number of points in the sequence. Must be non-negative
        dtype (torch.dtype, optional): desired data type of returned tensor
        device (torch.device, optional): desired device of returned tensor
    
    Returns:
        output (Tensor): the value at position :attr:`index` in the linear sequence
    """
    if steps == 0:
        return torch.tensor([], dtype=dtype, device=device)[index]
    else:
        start = torch.as_tensor(start, dtype=dtype, device=device)
        if steps == 1:
            return (start,)[index]
        else:
            if steps < 0:
                raise RuntimeError("number of steps must be non-negative")
            if index < 0:
                index = steps + index
            end = torch.as_tensor(end, dtype=dtype, device=device)
            common_difference = torch.as_tensor((end - start) / (steps - 1))
            return start + common_difference * index


def advanced_indexing(shape: Sequence[int], indices: Tensor, dim: int):
    """
    Generates advanced indexing arguments for selecting elements along a dimension.
    
    This function creates indexing tuples suitable for advanced indexing
    to select elements from tensors. It handles broadcasting of indices across
    other dimensions automatically.
    
    Args:
        shape (Sequence[int]): shape of the target tensor
        indices (Tensor): indices to select along dimension :attr:`dim`
        dim (int): dimension along which to apply indexing
    
    Returns:
        output (tuple): indexing tuple that can be used for tensor indexing
    """
    len_shape = len(shape)
    indices_shape = indices.shape
    left_args = _limited_dim_indexing(shape[:dim], len_shape)
    right_args = _limited_dim_indexing(shape[dim + 1:], len_shape, dim + 1)
    return *left_args, indices.view((*indices_shape,) + (1,) * (len_shape - len(indices_shape))), *right_args


def grow(input: Tensor, ndim: int, direction: str = 'leading') -> Tensor:
    """
    Adds singleton dimensions to a tensor to reach a target dimensionality.
    
    This function expands a tensor by adding singleton (size 1) dimensions either
    at the beginning ('leading') or end ('trailing') to match the specified
    number of dimensions.
    
    Args:
        input (Tensor): input tensor
        ndim (int): target number of dimensions
        direction (str): where to add singleton dimensions.
            Options: 'leading' (add at beginning) or 'trailing' (add at end).
            Default: 'leading'
    
    Returns:
        output (Tensor): expanded tensor with :attr:`ndim` dimensions
    
    Examples:
        >>> x = torch.randn(3, 4)
        >>> grow(x, 4, direction='leading').shape
        torch.Size([1, 1, 3, 4])
        
        >>> grow(x, 4, direction='trailing').shape
        torch.Size([3, 4, 1, 1])
    """
    tensor_shape = input.shape
    remaining_dims_num = ndim - len(tensor_shape)
    if direction == 'leading':
        return input.view((*((1,) * remaining_dims_num), *tensor_shape))
    elif direction == 'trailing':
        return input.view((*tensor_shape, *((1,) * remaining_dims_num)))
    else:
        return input


def apply(function: Callable, input: Tensor, dim: int = 0, otypes: Optional[Sequence[torch.dtype]] = None) -> Union[Tensor, Sequence[Tensor]]:
    """
    Applies a function to slices of a tensor along a specified dimension.
    
    This function splits the input tensor along dimension :attr:`dim`, applies
    the given function to each slice, and collects the results. It supports
    both single-output and multiple-output functions.
    
    Args:
        function (Callable): function to apply to each slice. Should take a tensor
            slice and return a value or tuple of values
        input (Tensor): input tensor
        dim (int): dimension along which to split and apply the function.
            Default: 0
        otypes (Sequence[torch.dtype], optional): output data types for the results.
            If multiple types are provided, the function should return multiple values
    
    Returns:
        output (Tensor or Sequence[Tensor]): collected results from applying the
            function to each slice. Returns a single tensor if :attr:`otypes` has
            length 1, otherwise returns a tuple of tensors
    """
    slices = (slice(None, None, None),) * dim
    if len(otypes) > 1:
        return tuple(torch.as_tensor(item, dtype=otypes[i], device=input.device) for i, item in enumerate(zip(*[function(input[slices + (i,)]) for i in range(input.shape[dim])])))
    elif len(otypes) == 1:
        return torch.as_tensor([function(input[slices + (i,)]) for i in range(input.shape[dim])], otypes[0], device=input.device)
    else:
        [function(input[slices + (i,)]) for i in range(input.shape[dim])]


def _shift_arguments(input: Tensor, shift: Tensor):
    batch_size, num_channels, *spatial_shape = input.shape
    length_ndim = len(spatial_shape)
    expected_shifts_size = torch.Size((batch_size, num_channels, length_ndim))
    
    if is_float(shift.dtype):
        shift = shift.round().int()
    if shift.shape != expected_shifts_size:
        shift = shift.broadcast_to(expected_shifts_size)
    del expected_shifts_size
    
    device = input.device

    aranged = torch.arange(amax(input.shape), device=device)
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


def _scalar_default_value(scalar_default, eps = 1e-6):
    if scalar_default == 'max':
        return 1 - eps
    elif scalar_default == 'min':
        return eps
    else:
        return 0.5


def _limited_dim_indexing(using_shape, len_shape, start: int = 0):
    return (torch.arange(dim_size).view((1,) * (i + start) + (dim_size,) + (1,) * (len_shape - (i + start) - 1)) for i, dim_size in enumerate(using_shape))