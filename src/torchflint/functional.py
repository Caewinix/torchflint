from typing import Sequence, Optional, Union
from functools import reduce
from itertools import chain
from numbers import Number
import torch
from torch import Tensor
from .nn import Buffer
from .utils._helper import __module__


@__module__(__package__)
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
        >>> class MyModule(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.my_buffer = torchflint.buffer(torch.randn(3, 4), persistent=True)
        >>>
        >>> my_module = MyModule()
        >>> my_module.my_buffer.shape
        torch.Size([3, 4])
    """
    return Buffer(tensor, persistent)


@__module__(__package__)
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
        >>> torchflint.promote_types(torch.int32, torch.float32)
        torch.float32
        >>> torchflint.promote_types(torch.int8, torch.int16, torch.int32)
        torch.int32
    """
    return reduce(torch.promote_types, dtypes)


@__module__(__package__)
def map_range(
    input: Tensor,
    interval: Sequence[int] = (0, 1),
    dim: Union[int, Sequence[int], None] = None,
    dtype: torch.dtype = None
) -> Tensor:
    """
    Linearly maps input values to a specified output range for specific dimension(s).
    
    This function computes the minimum and maximum values along specified dimension(s)
    and linearly transforms the input to the target interval. Constant inputs
    (where minimum == maximum) will remain unchanged.
    
    Args:
        input (Tensor): input tensor to normalize
        interval (pair of ints): target interval for the output values.
            Default: (0, 1)
        dim (int or ints, optional): dimension(s) along which to compute
            min and max. If `None`, uses all dimensions
        dtype (torch.dtype, optional): desired data type for output tensor
    
    Returns:
        output (Tensor): normalized tensor with values in the specified interval
    """
    min_value = amin(input, dim=dim, keepdim=True)
    max_value = amax(input, dim=dim, keepdim=True)
    max_min_difference = max_value - min_value
    max_min_equal_mask = max_min_difference == 0
    max_min_difference[max_min_equal_mask].detach_()
    min_value[max_min_equal_mask].detach_()
    max_min_difference.masked_fill_(max_min_equal_mask, 1)
    min_value.masked_fill(max_min_equal_mask, 0)
    return (((input - min_value) / max_min_difference).to(dtype) * (interval[1] - interval[0]) + interval[0])


@__module__(__package__)
def map_ranges(
    input: Tensor,
    intervals: Union[Sequence[Sequence[int]], Tensor] = [(0, 1)],
    dim: Union[int, Sequence[int], None] = None,
    dtype: Optional[torch.dtype] = None
) -> Tensor:
    """
    Maps input values to multiple output ranges simultaneously, broadcasting results.
    
    This function applies :func:`map_range` to the same input for multiple target
    intervals, producing a batched output where the first dimension contains the
    results for each interval. This is useful for generating multiple normalized
    versions of the same data in a single operation. Constant inputs
    (where minimum == maximum) will remain unchanged.
    
    Args:
        input (Tensor): input tensor to normalize
        intervals (many pairs of ints or Tesnor): list of target intervals for the output values.
            Default: [(0, 1)]
        dim (int or ints, optional): dimension(s) along which to compute
            min and max. If `None`, uses all dimensions
        dtype (torch.dtype, optional): desired data type for output tensor
    
    Returns:
        output (Tensor): batched normalized tensor where the first dimension corresponds
            to different intervals
    """
    min_value = amin(input, dim=dim, keepdim=True)
    max_value = amax(input, dim=dim, keepdim=True)
    intervals = torch.as_tensor(intervals, device=input.device, dtype=input.dtype)
    max_min_difference = max_value - min_value
    max_min_equal_mask = max_min_difference == 0
    max_min_difference[max_min_equal_mask].detach_()
    min_value[max_min_equal_mask].detach_()
    max_min_difference.masked_fill_(max_min_equal_mask, 1)
    min_value.masked_fill(max_min_equal_mask, 0)
    normalized = ((input - min_value) / max_min_difference).to(dtype=dtype)[None, ...]
    batched_ndim = normalized.ndim
    return normalized * as_ndim(intervals.diff(dim=1).squeeze(-1), batched_ndim) + as_ndim(intervals[..., 0], batched_ndim)


try:
    from torch import amin, amax
except:
    @__module__(__package__)
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


    @__module__(__package__)
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


@__module__(__package__)
def is_int(dtype: torch.dtype) -> bool:
    """
    Checks if a dtype represents integer data.
    
    Args:
        dtype (torch.dtype): the dtype to check
    
    Returns:
        output (bool): `True` if the dtype is an integer type, `False` otherwise
    
    Examples:
        >>> torchflint.is_int(torch.int32)
        True
        >>> torchflint.is_int(torch.float32)
        False
    """
    try:
        torch.iinfo(dtype)
        return True
    except TypeError:
        return False


@__module__(__package__)
def is_float(dtype: torch.dtype) -> bool:
    """
    Checks if a dtype represents floating-point data.
    
    Args:
        dtype (torch.dtype): the dtype to check
    
    Returns:
        output (bool): `True` if the dtype is a floating-point type, `False` otherwise
    
    Examples:
        >>> torchflint.is_float(torch.float32)
        True
        >>> torchflint.is_float(torch.int32)
        False
    """
    return dtype.is_floating_point


@__module__(__package__)
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
        >>> tensor = torch.tensor([[1.0, 2.0, 3.0],
        ...                        [4.0, 5.0, 6.0]])
        >>> 
        >>> # Invert along columns (dim=1)
        >>> torchflint.invert(tensor, dim=1)
        tensor([[3.0, 2.0, 1.0],
                [6.0, 5.0, 4.0]])
        >>> 
        >>> # Invert along rows (dim=0)
        >>> torchflint.invert(tensor, dim=0)
        tensor([[4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0]])
        >>> 
        >>> # Invert along both dimensions
        >>> torchflint.invert(tensor, dim=(0, 1))
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


@__module__(__package__)
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


@__module__(__package__)
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


@__module__(__package__)
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


_none_slice = slice(None)
@__module__(__package__)
def insert_ndim(input: Tensor, dim: int, ndim: int = 1) -> Tensor:
    """
    Inserts singleton dimensions at a specified position in a tensor.
    
    This function inserts :attr:`ndim` singleton dimensions at position
    :attr:`dim` in the input tensor. This is similar to :func:`torch.unsqueeze`
    but allows inserting multiple dimensions at once.
    
    The returned tensor shares the same underlying data with this tensor.

    A :attr:`dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``
    can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
    applied at :attr:`dim` = ``dim + input.dim() + 1``.
    
    Args:
        input (Tensor): input tensor
        dim (int): position at which to insert singleton dimensions
        ndim (int): number of singleton dimensions to insert. Default: 1
    
    Returns:
        output (Tensor): tensor with singleton dimensions inserted at the
            specified position
    
    Examples:
        >>> tensor = torch.randn(3, 4)
        >>> 
        >>> # Insert 2 singleton dimensions at position 1
        >>> torchflint.insert_ndim(tensor, dim=1, ndim=2).shape
        torch.Size([3, 1, 1, 4])
        >>> 
        >>> # Insert 1 singleton dimension at the end (same as unsqueeze(-1))
        >>> torchflint.insert_ndim(tensor, dim=-1, ndim=1).shape
        torch.Size([3, 4, 1])
        >>> 
        >>> # Insert 3 singleton dimensions at the beginning
        >>> torchflint.insert_ndim(tensor, dim=0, ndim=3).shape
        torch.Size([1, 1, 1, 3, 4])
    """
    if dim >= 0:
        return input[tuple(current for current in chain((_none_slice for _ in range(dim)), (None for _ in range(ndim))))]
    else:
        return input[(..., *(current for current in chain((None for _ in range(ndim)), (_none_slice for _ in range(-1 - dim)))))]


@__module__(__package__)
def as_ndim(input: Tensor, ndim: int, *, dim: int = 0) -> Tensor:
    """
    Ensures a tensor has at least a specified number of dimensions by inserting singleton dimensions.
    
    This function inserts singleton dimensions to guarantee the output tensor has
    exactly :attr:`ndim` dimensions. If the input already has :attr:`ndim` or more
    dimensions, it is returned unchanged. Otherwise, singleton dimensions are inserted
    at the specified position to reach the target dimensionality.
    
    The returned tensor shares the same underlying data with this tensor.
    
    A :attr:`dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``
    can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`
    applied at :attr:`dim` = ``dim + input.dim() + 1``.
    
    Args:
        input (Tensor): input tensor
        ndim (int): target output number of dimensions
        dim (int): position at which to insert singleton dimensions. Default: 0
    
    Returns:
        output (Tensor): tensor with at least :attr:`ndim` dimensions
    
    Examples:
        >>> tensor = torch.randn(3, 4)
        >>> 
        >>> # Ensure at least 4 dimensions (inserts 2 singleton dimensions at beginning)
        >>> torchflint.as_ndim(tensor, ndim=4).shape
        torch.Size([1, 1, 3, 4])
        >>> 
        >>> # Ensure at least 4 dimensions, insert at position 1
        >>> torchflint.as_ndim(tensor, ndim=4, dim=1).shape
        torch.Size([3, 1, 1, 4])
        >>> 
        >>> # Input already has enough dimensions
        >>> tensor = torch.randn(2, 3, 4, 5)
        >>> torchflint.as_ndim(tensor, ndim=3).shape
        torch.Size([2, 3, 4, 5]) # unchanged
    """
    remaining_ndim = ndim - input.ndim
    if remaining_ndim <= 0:
        return input
    return insert_ndim(input, dim, remaining_ndim)