from typing import Union, Sequence, Optional, Callable
import torch
from torch import Tensor
from .patches import unfold_space, fold_space, kernel_expand, _spatialize_tuple
from ..utils import _masked


def pool(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False,
    *,
    default_space: Union[float, str] = 0,
    reducer: Callable[[Tensor, tuple[int, ...]], Tensor]
) -> Tensor:
    """
    Applies a pooling operation over the input tensor by extracting spatial patches and reducing
    them using a specified reduction function. This operation extracts sliding local patches
    (blocks) from spatial dimensions and aggregates them to produce downsampled output.
    
    The input tensor should have the format where the first two dimensions are batch size and
    the number of channels, followed by all spatial dimensions. The operation first extracts
    patches, then applies the :attr:`reducer` function over the kernel dimensions of each patch
    to produce a single value per patch.
    
    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        kernel_size (int or ints): the size of the window to take over
        stride (int or ints): the stride of the window in the input spatial dimensions
        padding (int or ints): implicit padding to be added on both sides of input. Default: 0
        dilation (int or ints): a parameter that controls the stride of elements within the neighborhood.
            Default: 1
        ceil_mode (bool): If `True`, will use ceil instead of floor to compute the output shape.
            This ensures that every element in the input tensor is covered by a sliding window. Default: False
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
        reducer (Callable[[Tensor, tuple[int, ...]], Tensor]): reduction function that takes a patch tensor
            and a tuple of kernel dimensions, and returns a reduced tensor. The reducer should eliminate
            the kernel dimensions through aggregation
    
    Returns:
        output (Tensor): the pooled output tensor with shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n)`
    
    Note:
        The :attr:`reducer` function must return a tensor that has the kernel dimensions removed.
        Common reduction functions include `torch.mean`, `torch.max`, `torch.sum`, etc.
    """
    spatial_ndim = input.ndim - 2
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode, default_space=default_space)
    return reducer(output, tuple(range(-spatial_ndim, 0, 1)))


def unpool(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1
) -> Tensor:
    """
    Unpools the original spatial resolution by expanding and folding the input tensor.
    This operation performs the inverse of a pooling operation, increasing the spatial
    dimensions by distributing pooled output values across kernel regions. Overlapping
    regions will be averaged.
    
    The input tensor should have the format where the first two dimensions are batch size
    and the number of channels, followed by all spatial dimensions. The operation first
    expands the input by adding kernel dimensions of the specified :attr:`kernel_size`,
    then folds these dimensions back into spatial dimensions using the specified :attr:`stride`,
    :attr:`padding`, and :attr:`dilation` parameters.
    
    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        kernel_size (int or ints): the size of the kernel regions to distribute each input value
        stride (int or ints): the stride of the window in the input spatial dimensions
        padding (int or ints): implicit padding to be removed on both sides of output. Default: 0
        dilation (int or ints): a parameter that controls the stride of elements within the neighborhood.
            Default: 1
    
    Returns:
        output (Tensor): the unpooled output tensor with shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    spatial_ndim = input.ndim - 2
    kernel_size = _spatialize_tuple(kernel_size, spatial_ndim)
    input = kernel_expand(input, kernel_size)
    return fold_space(input, stride, padding, dilation)


def max_pool(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False
) -> Tensor:
    """
    Applies a max pooling over an input tensor composed of several input planes.

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.

    Note:
        When `ceil_mode=True`, sliding windows are allowed to go off-bounds if they start within the negative side padding
        or the input. Sliding windows that would start in the positive side padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        kernel_size (int or ints): the size of the window to take a max over
        stride (int or ints): the stride of the window
        padding (int or ints): implicit negative infinity padding to be added on both sides
        dilation (int or ints): a parameter that controls the stride of elements in the window
        ceil_mode (bool): when `True`, will use `ceil` instead of `floor` to compute the output shape
    
    Returns:
        output (Tensor): the max pooled output tensor with shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    spatial_ndim = input.ndim - 2
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode, default_space=float("-inf"))
    return output.amax(tuple(range(-spatial_ndim, 0, 1)))


def masked_max_pool(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False,
    *,
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Applies a max pooling over a masked input tensor composed of several input planes.

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.

    Note:
        When `ceil_mode=True`, sliding windows are allowed to go off-bounds if they start within the negative side padding
        or the input. Sliding windows that would start in the positive side padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        kernel_size (int or ints): the size of the window to take a max over
        stride (int or ints): the stride of the window
        padding (int or ints): implicit negative infinity padding to be added on both sides
        dilation (int or ints): a parameter that controls the stride of elements in the window
        ceil_mode (bool): when `True`, will use `ceil` instead of `floor` to compute the output shape
        mask (Tensor, optional): when given, only the mask region will be used for max pooling
    
    Returns:
        output (Tensor): the masked max pooled output tensor with shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    spatial_ndim = input.ndim - 2
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode, default_space=float("-inf"))
    if mask is not None:
        mask = unfold_space(mask, kernel_size, stride, padding, dilation, ceil_mode, default_space=0)
    return _masked.amax(output, tuple(range(-spatial_ndim, 0, 1)), mask=mask)


def min_pool(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False
) -> Tensor:
    """
    Applies a min pooling over an input tensor composed of several input planes.

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.

    Note:
        When `ceil_mode=True`, sliding windows are allowed to go off-bounds if they start within the negative side padding
        or the input. Sliding windows that would start in the positive side padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        kernel_size (int or ints): the size of the window to take a min over
        stride (int or ints): the stride of the window
        padding (int or ints): implicit positive infinity padding to be added on both sides
        dilation (int or ints): a parameter that controls the stride of elements in the window
        ceil_mode (bool): when `True`, will use `ceil` instead of `floor` to compute the output shape
    
    Returns:
        output (Tensor): the min pooled output tensor with shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    spatial_ndim = input.ndim - 2
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode, default_space=float("inf"))
    return output.amin(tuple(range(-spatial_ndim, 0, 1)))


def masked_min_pool(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False,
    *,
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Applies a min pooling over a masked input tensor composed of several input planes.

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.

    Note:
        When `ceil_mode=True`, sliding windows are allowed to go off-bounds if they start within the negative side padding
        or the input. Sliding windows that would start in the positive side padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        kernel_size (int or ints): the size of the window to take a min over
        stride (int or ints): the stride of the window
        padding (int or ints): implicit positive infinity padding to be added on both sides
        dilation (int or ints): a parameter that controls the stride of elements in the window
        ceil_mode (bool): when `True`, will use `ceil` instead of `floor` to compute the output shape
        mask (Tensor, optional): when given, only the mask region will be used for min pooling
    
    Returns:
        output (Tensor): the masked min pooled output tensor with shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    spatial_ndim = input.ndim - 2
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode, default_space=float("inf"))
    if mask is not None:
        mask = unfold_space(mask, kernel_size, stride, padding, dilation, ceil_mode, default_space=1)
    return _masked.amin(output, tuple(range(-spatial_ndim, 0, 1)), mask=mask)


def avg_pool(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None
) -> Tensor:
    """
    Applies an average pooling over an input tensor composed of several input planes.

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When `ceil_mode=True`, sliding windows are allowed to go off-bounds if they start within the negative side padding
        or the input. Sliding windows that would start in the positive side padded region are ignored.

    .. note::
        pad should be at most half of effective kernel size.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        kernel_size (int or ints): the size of the window
        stride (int or ints): the stride of the window. Default value is :attr:`kernel_size`
        padding (int or ints): implicit zero padding to be added on both sides
        ceil_mode (bool): when `True`, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad (bool): when `True`, will include the zero-padding in the averaging calculation
        divisor_override (int, optional): if specified, it will be used as divisor, otherwise size of the pooling region will be used
    
    Returns:
        output (Tensor): the average pooled output tensor with shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    spatial_ndim = input.ndim - 2
    dim = tuple(range(-spatial_ndim, 0, 1))
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode)
    if divisor_override is None:
        if count_include_pad:
            return output.mean(dim)
        else:
            mask = unfold_space(
                torch.ones((1,) * input.ndim, device=input.device, dtype=torch.bool).expand(input.size(0), 1, *input.shape[2:]),
                kernel_size, stride, padding, dilation, ceil_mode
            )
            return _masked.mean(output, dim, mask=mask)
    else:
        return output.sum(dim) / divisor_override


def masked_avg_pool(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    *,
    mask: Optional[Tensor] = None
) -> Tensor:
    """
    Applies an average pooling over a masked input tensor composed of several input planes.

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When `ceil_mode=True`, sliding windows are allowed to go off-bounds if they start within the negative side padding
        or the input. Sliding windows that would start in the positive side padded region are ignored.

    .. note::
        pad should be at most half of effective kernel size.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        kernel_size (int or ints): the size of the window
        stride (int or ints): the stride of the window. Default value is :attr:`kernel_size`
        padding (int or ints): implicit zero padding to be added on both sides
        ceil_mode (bool): when `True`, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad (bool): when `True`, will include the zero-padding in the averaging calculation
        divisor_override (int, optional): if specified, it will be used as divisor, otherwise size of the pooling region will be used
        mask (Tensor, optional): when given, only the mask region will be used for average pooling
    
    Returns:
        output (Tensor): the average pooled output tensor with shape
            `(batch_size, num_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    spatial_ndim = input.ndim - 2
    dim = tuple(range(-spatial_ndim, 0, 1))
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode)
    if divisor_override is None:
        if mask is None:
            if count_include_pad:
                mask = unfold_space(
                    torch.ones((1,) * input.ndim, device=input.device, dtype=torch.bool).expand(input.size(0), 1, *input.shape[2:]),
                    kernel_size, stride, padding, dilation, ceil_mode
                )
        else:
            mask = unfold_space(mask, kernel_size, stride, padding, dilation, ceil_mode, default_space=0)
        return _masked.mean(output, dim, mask=mask)
    else:
        if mask is not None:
            mask = unfold_space(mask, kernel_size, stride, padding, dilation, ceil_mode, default_space=0)
        return _masked.sum(output, dim, mask=mask) / divisor_override