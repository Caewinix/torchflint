from typing import Union, Sequence, Tuple, Optional, Callable
import torch
from torch import Tensor
import torch.nn.functional as F
from .patches import unfold_space, fold_space, fold_stack, patches_column, _spatialize_tuple
from ..utils import _masked


def conv_patches(input: Tensor, weight: Tensor, groups: int = 1) -> Tensor:
    """
    Applies a convolution operation directly on a pre-unfolded patches tensor.
    
    This function takes a tensor of pre-extracted spatial patches and a batched weight tensor,
    and computes the convolution output.
    
    Args:
        input (Tensor): input tensor containing pre-unfolded patches, with shape
            `(batch_size, in_channels, out_length_0, out_length_1, ..., out_length_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        weight (Tensor): weight tensor containing convolution kernels, with shape
            `(batch_size, out_channels, in_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
            
            If the batch dimension of weight is 1, the same weights are applied to all
            batch elements in the input (weight sharing across batch).
        groups (int): number of blocked connections from input channels to output channels.
            Both :attr:`in_channels` and :attr:`out_channels` must be divisible by :attr:`groups`.
            Default: 1
    
    Returns:
        output (Tensor): the convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    
    Note:
        This function requires that the input tensor has already been unfolded into patches
        (e.g., using :func:`unfold_space`). The kernel dimensions in the input must match
        the kernel dimensions in the weight tensor.
    """
    column = patches_column(input) # [B, C_in * K, L]
    column = column.view(column.size(0), groups, -1, column.size(-1)) # [B, G, C_in // G * K, L]
    flat_weight = weight.flatten(2) # [B, C_out, C_in // G * K]
    flat_weight = flat_weight.view(flat_weight.size(0), groups, -1, flat_weight.size(-1)) # [B, G, C_out // G, C_in // G * K]
    output = torch.matmul(flat_weight, column).view(input.size(0), -1, *input.shape[2:weight.ndim - 1]) # [B, G, C_out // G, L] -> [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    return output


def conv(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a convolution over an input tensor composed of several input planes.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(out_channels, in_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): padding added to all sides of the input. Default: 0
        dilation (int or ints): spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return batch_conv(
        input,
        weight.unsqueeze(0),
        bias,
        stride,
        padding,
        dilation,
        groups,
        default_space=default_space
    )


def batch_conv(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a convolution over an input tensor composed of several input planes, but the weight tensor
    and the bias tensor are batched.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, out_channels, in_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(batch_size, out_channels)` or `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): padding added to all sides of the input. Default: 0
        dilation (int or ints): spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    input = unfold_space(input, weight.shape[3:], stride, padding, dilation, default_space=default_space)
    output = conv_patches(input, weight, groups)
    if bias is not None:
        output = output + bias[(..., *((None,) * (input.ndim - 2)))] # [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    return output


def conv_patches_explicitly(input: Tensor, weight: Tensor, groups: int = 1):
    """
    Applies a convolution operation explicitly on a pre-unfolded patches tensor without matrix multiplication.
    
    This function takes a tensor of pre-extracted spatial patches and a batched weight tensor,
    and computes the convolution output.
    
    Args:
        input (Tensor): input tensor containing pre-unfolded patches, with shape
            `(batch_size, in_channels, out_length_0, out_length_1, ..., out_length_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        weight (Tensor): weight tensor containing convolution kernels, with shape
            `(batch_size, out_channels, in_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
            
            If the batch dimension of weight is 1, the same weights are applied to all
            batch elements in the input (weight sharing across batch).
        groups (int): number of blocked connections from input channels to output channels.
            Both :attr:`in_channels` and :attr:`out_channels` must be divisible by :attr:`groups`.
            Default: 1
    
    Returns:
        output (Tensor): the convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    
    Note:
        This function requires that the input tensor has already been unfolded into patches
        (e.g., using :func:`unfold_space`). The kernel dimensions in the input must match
        the kernel dimensions in the weight tensor.
    """
    spatial_ndim = weight.ndim - 3
    input = input.view(input.size(0), groups, -1, *input.shape[2:]) # [B, G, C_in // G, out_L_0, out_L_1, ..., out_L_n, k_L_0, k_L_1, ..., k_L_n]
    input = input.permute(0, 1, *range(3, weight.ndim), 2, *range(weight.ndim, input.ndim))[:, :, None] # [B, G, 1, out_L_0, out_L_1, ..., out_L_n, C_in // G, k_L_0, k_L_1, ..., k_L_n]
    weight = weight.view(weight.size(0), groups, -1, *((1,) * spatial_ndim), *weight.shape[2:]) # [B, G, C_out // G, 1 (0), 1 (1), ..., 1 (n), C_in // G, k_L_0, k_L_1, ..., k_L_n]
    output = torch.sum(input * weight, tuple(range(-spatial_ndim - 1, 0, 1))).flatten(1, 2) # [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    return output


def conv_explicitly(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a convolution explicitly over an input tensor composed of several input planes without matrix multiplication.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(out_channels, in_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): padding added to all sides of the input. Default: 0
        dilation (int or ints): spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return batch_conv_explicitly(
        input,
        weight.unsqueeze(0),
        bias,
        stride,
        padding,
        dilation,
        groups,
        default_space=default_space
    )


def batch_conv_explicitly(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a convolution explicitly over an input tensor composed of several input planes without matrix multiplication,
    but the weight tensor and the bias tensor are batched.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, out_channels, in_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(batch_size, out_channels)` or `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): padding added to all sides of the input. Default: 0
        dilation (int or ints): spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    input = unfold_space(input, weight.shape[3:], stride, padding, dilation, default_space=default_space)
    output = conv_patches_explicitly(input, weight, groups)
    if bias is not None:
        output = output + bias[(..., *((None,) * (input.ndim - 2)))] # [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    return output


def masked_conv_patches(
    input: Tensor,
    weight: Tensor,
    groups: int = 1,
    *,
    input_mask: Optional[Tensor] = None,
    weight_mask: Optional[Tensor] = None,
    soft_mode: bool = False
):
    """
    Applies a convolution operation directly on a pre-unfolded patches tensor in the mask (for input and weight) region.
    
    This function takes a tensor of pre-extracted spatial patches and a batched weight tensor,
    and computes the convolution output.
    
    Args:
        input (Tensor): input tensor containing pre-unfolded patches, with shape
            `(batch_size, in_channels, out_length_0, out_length_1, ..., out_length_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        weight (Tensor): weight tensor containing convolution kernels, with shape
            `(batch_size, out_channels, in_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
            
            If the batch dimension of weight is 1, the same weights are applied to all
            batch elements in the input (weight sharing across batch).
        groups (int): number of blocked connections from input channels to output channels.
            Both :attr:`in_channels` and :attr:`out_channels` must be divisible by :attr:`groups`.
            Default: 1
        input_mask (Tensor, optional): a mask for the input tensor of shape
            `(batch_size, in_channels, length_0, length_1, ..., length_n)`. Default: None
        weight_mask (Tensor, optional): a mask for the weight tensor of shape
            `(batch_size, out_channels, in_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`.
            Default: None
        soft_mode (bool): If `True`, the window with part of the mask will not be discarded. Default: False
    
    Returns:
        output (Tensor): the convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    
    Note:
        This function requires that the input tensor has already been unfolded into patches
        (e.g., using :func:`unfold_space`). The kernel dimensions in the input must match
        the kernel dimensions in the weight tensor.
    """
    spatial_ndim = weight.ndim - 3
    mask = None
    
    input_shape = input.shape
    new_input_shape = (input.size(0), groups, -1, *input.shape[2:]) # [B, G, C_in // G, out_L_0, out_L_1, ..., out_L_n, k_L_0, k_L_1, ..., k_L_n]
    input = input.view(new_input_shape)
    input_permutation = (0, 1, *range(3, weight.ndim), 2, *range(weight.ndim, input.ndim)) # [B, G, out_L_0, out_L_1, ..., out_L_n, C_in // G, k_L_0, k_L_1, ..., k_L_n]
    input = input.permute(input_permutation)[:, :, None] # [B, G, 1, out_L_0, out_L_1, ..., out_L_n, C_in // G, k_L_0, k_L_1, ..., k_L_n]
    
    weight_shape = weight.shape
    new_weight_shape = (weight.size(0), groups, -1, *((1,) * spatial_ndim), *weight.shape[2:]) # [B, G, C_out // G, 1 (0), 1 (1), ..., 1 (n), C_in // G, k_L_0, k_L_1, ..., k_L_n]
    weight = weight.view(new_weight_shape)
    
    weighted_input = input * weight
    
    if input_mask is not None:
        input_mask = input_mask.expand(input_shape).view(new_input_shape).permute(input_permutation)[:, :, None]
        if not soft_mode:
            input_mask = input_mask.amin(tuple(range(-spatial_ndim, 0, 1)), keepdim=True).expand(input_mask.shape)
        mask = input_mask
    
    if weight_mask is not None:
        weight_mask = weight_mask.expand(weight_shape).view(new_weight_shape)
        weighted_input_shape = weighted_input.shape
        if mask is None:
            mask = weight_mask.expand(weighted_input_shape)
        else:
            mask = (mask & weight_mask).expand(weighted_input_shape)
    
    output = _masked.sum(weighted_input, tuple(range(-spatial_ndim - 1, 0, 1)), mask=mask).flatten(1, 2) # [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    return output


def masked_conv(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    *,
    input_mask: Optional[Tensor] = None,
    weight_mask: Optional[Tensor] = None,
    soft_mode: bool = False,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a convolution over an input tensor composed of several input planes in the mask (for input and weight) region.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, out_channels, in_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(batch_size, out_channels)` or `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): padding added to all sides of the input. Default: 0
        dilation (int or ints): spacing between kernel elements. Default: 1
        groups (int): number of blocked connections from input channels to output channels.
            Default: 1
        input_mask (Tensor, optional): a mask for the input tensor of shape
            `(batch_size, in_channels, length_0, length_1, ..., length_n)`. Default: None
        weight_mask (Tensor, optional): a mask for the weight tensor of shape
            `(batch_size, out_channels, in_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`.
            Default: None
        soft_mode (bool): If `True`, the window with part of the mask will not be discarded. Default: False
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    kernel_size = weight.shape[3:]
    input = unfold_space(input, kernel_size, stride, padding, dilation, default_space=default_space)
    if input_mask is None:
        input_mask = None
    else:
        input_mask = unfold_space(input_mask, kernel_size, stride, padding, dilation)
    output = masked_conv_patches(input, weight, groups, input_mask=input_mask, weight_mask=weight_mask, soft_mode=soft_mode)
    if bias is not None:
        output = output + bias[(..., *((None,) * (input.ndim - 2)))] # [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    return output


def conv_transpose_patches(input: Tensor, weight: Tensor, groups: int = 1) -> Tensor:
    """
    Applies a transposed convolution operation by generating patches from input and weights,
    producing an output in patch format suitable for subsequent folding operations.
    
    This function performs the patch-based equivalent of a transposed convolution (also known
    as fractionally-strided convolution or deconvolution). Instead of directly producing the
    final spatial output, it generates a patch tensor that can be folded (using :func:`fold_space`
    or :func:`fold_stack`) to obtain the final upsampled result. This approach is particularly
    efficient for implementing transposed convolutions through patch operations.
    
    The function takes an input tensor and weight tensor, and computes the contribution of
    each input element to all positions in the corresponding output patch. The result is
    a patch tensor where each spatial position contains the kernel-sized contribution from
    the corresponding input element.
    
    The parameters :attr:`weight` should be batched (different weights per batch element).
    
    Args:
        input (Tensor): input tensor of shape
            `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        groups (int): number of blocked connections from input channels to output channels.
    
    Returns:
        output (Tensor): the transposed convolution patch tensor with shape
            `(batch_size, out_channels, length_0, length_1, ..., length_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
    
    Note:
        The output of this function is in patch format, not the final spatial output.
        To obtain the final transposed convolution result, the output should be passed
        to :func:`fold_space` or :func:`fold_stack` with appropriate stride, padding,
        and dilation parameters that match the original transposed convolution configuration.
    """
    input_ndim = input.ndim
    
    batch_size = input.size(0)
    flat_input = input.flatten(2) # [B, C_in, L]
    reshaped_input = flat_input.view(batch_size, groups, -1, flat_input.size(-1)).permute(0, 1, 3, 2) # [B, G, L, C_in // G]
    flat_weight = weight.flatten(2) # [B, C_in, (C_out // G) * K]
    reshaped_weight = flat_weight.view(weight.size(0), groups, -1, flat_weight.size(-1)) # [B, G, C_in // G, (C_out // G) * K]

    # The physical meaning calculated in this step is that for each pixel position (L) of the input,
    # a local block ((C_out // G) * K) will be generated.
    output = torch.matmul(reshaped_input, reshaped_weight) # [B, G, L, (C_out // G) * K]
    output = output.view(batch_size, groups, *input.shape[2:], -1, *weight.shape[3:]) # [B, G, L_0, L_1, ..., L_n, C_out // G, k_L_0, k_L_1, ..., k_L_n]
    output = output.permute(0, 1, input_ndim, *range(2, input_ndim), *range(input_ndim + 1, output.ndim)) # [B, G, C_out // G, L_0, L_1, ..., L_n, k_L_0, k_L_1, ..., k_L_n]
    output = output.reshape(batch_size, -1, *input.shape[2:], *weight.shape[3:]) # [B, C_out, L_0, L_1, ..., L_n, k_L_0, k_L_1, ..., k_L_n]
    
    return output


def conv_transpose(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a transposed convolution operator over an input tensor composed of several input planes.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): ``dilation * (kernel_size - 1) - padding`` padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        groups (int): number of blocked connections from input channels to output channels. Default: 1
        dilation (int or ints): spacing between kernel elements. Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the transposed convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return batch_conv_transpose(
        input,
        weight.unsqueeze(0),
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        default_space=default_space
    )


def batch_conv_transpose(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a transposed convolution operator over an input tensor composed of several input planes,
    but the weight tensor and the bias tensor are batched.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(batch_size, out_channels)` or `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): ``dilation * (kernel_size - 1) - padding`` padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        groups (int): number of blocked connections from input channels to output channels. Default: 1
        dilation (int or ints): spacing between kernel elements. Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the transposed convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return _conv_transpose_template(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        default_space=default_space,
        expander=conv_transpose_patches,
        folder=fold_stack
    )


def deconv(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a deconvolution operator over an input tensor composed of several input planes,
    and the overlapping regions will be averaged.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): ``dilation * (kernel_size - 1) - padding`` padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        groups (int): number of blocked connections from input channels to output channels. Default: 1
        dilation (int or ints): spacing between kernel elements. Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the deconvolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return batch_deconv(
        input,
        weight.unsqueeze(0),
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        default_space=default_space
    )


def batch_deconv(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a deconvolution operator over an input tensor composed of several input planes,
    and the overlapping regions will be averaged, but the weight tensor and the bias tensor are batched.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(batch_size, out_channels)` or `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): ``dilation * (kernel_size - 1) - padding`` padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        groups (int): number of blocked connections from input channels to output channels. Default: 1
        dilation (int or ints): spacing between kernel elements. Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the deconvolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return _conv_transpose_template(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        default_space=default_space,
        expander=conv_transpose_patches,
        folder=fold_space
    )


def conv_transpose_patches_explicitly(input: Tensor, weight: Tensor, groups: int = 1) -> Tensor:
    """
    Applies a transposed convolution operation by generating patches from input and weights explicitly
    without matrix multiplication, producing an output in patch format suitable for subsequent folding operations.
    
    This function performs the patch-based equivalent of a transposed convolution (also known
    as fractionally-strided convolution or deconvolution). Instead of directly producing the
    final spatial output, it generates a patch tensor that can be folded (using :func:`fold_space`
    or :func:`fold_stack`) to obtain the final upsampled result. This approach is particularly
    efficient for implementing transposed convolutions through patch operations.
    
    The function takes an input tensor and weight tensor, and computes the contribution of
    each input element to all positions in the corresponding output patch. The result is
    a patch tensor where each spatial position contains the kernel-sized contribution from
    the corresponding input element.
    
    The parameters :attr:`weight` should be batched (different weights per batch element).
    
    Args:
        input (Tensor): input tensor of shape
            `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        groups (int): number of blocked connections from input channels to output channels.
    
    Returns:
        output (Tensor): the transposed convolution patch tensor with shape
            `(batch_size, out_channels, length_0, length_1, ..., length_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
    
    Note:
        The output of this function is in patch format, not the final spatial output.
        To obtain the final transposed convolution result, the output should be passed
        to :func:`fold_space` or :func:`fold_stack` with appropriate stride, padding,
        and dilation parameters that match the original transposed convolution configuration.
    """
    spatial_ndim = input.ndim - 2
    
    reshaped_input = input.view(input.size(0), groups, -1, *input.shape[2:]) # [B, G, C_in // G, L_0, L_1, ..., L_n]
    reshaped_input = reshaped_input.permute(0, 1, *range(3, reshaped_input.ndim), 2) # [B, G, L_0, L_1, ..., L_n, C_in // G]
    reshaped_input = reshaped_input.unsqueeze(2)[(..., ) + (None,) * spatial_ndim] # [B, G, 1, L_0, L_1, ..., L_n, C_in // G, 1 (0), 1 (1), ..., 1 (n)]
    reshaped_weight = weight.view(weight.size(0), groups, -1, weight.size(2), *weight.shape[3:]) # [B, G, C_in // G, C_out // G, k_L_0, k_L_1, ..., k_L_n]
    reshaped_weight = reshaped_weight.permute(0, 1, 3, 2, *range(4, reshaped_weight.ndim)) # [B, G, C_out // G, C_in // G, k_L_0, k_L_1, ..., k_L_n]
    reshaped_weight = reshaped_weight[(*(slice(None),) * 3, *((None,) * spatial_ndim))] # [B, G, C_out // G, 1 (0), 1 (1), ..., 1 (n), C_in // G, k_L_0, k_L_1, ..., k_L_n]
    
    output = torch.sum(reshaped_input * reshaped_weight, dim=-spatial_ndim - 1).flatten(1, 2) # [B, C_out, L_0, L_1, ..., L_n, k_L_0, k_L_1, ..., k_L_n]
    return output


def conv_transpose_explicitly(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a transposed convolution operator over an input tensor composed of several input planes explicitly
    without matrix multiplication.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): ``dilation * (kernel_size - 1) - padding`` padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        groups (int): number of blocked connections from input channels to output channels. Default: 1
        dilation (int or ints): spacing between kernel elements. Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the transposed convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return batch_conv_transpose_explicitly(
        input,
        weight.unsqueeze(0),
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        default_space=default_space
    )


def batch_conv_transpose_explicitly(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a transposed convolution operator over an input tensor composed of several input planes explicitly
    without matrix multiplication, but the weight tensor and the bias tensor are batched.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(batch_size, out_channels)` or `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): ``dilation * (kernel_size - 1) - padding`` padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        groups (int): number of blocked connections from input channels to output channels. Default: 1
        dilation (int or ints): spacing between kernel elements. Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the transposed convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return _conv_transpose_template(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        default_space=default_space,
        expander=conv_transpose_patches_explicitly,
        folder=fold_stack
    )


def deconv_explicitly(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a deconvolution operator over an input tensor composed of several input planes explicitly
    without matrix multiplication, and the overlapping regions will be averaged.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): ``dilation * (kernel_size - 1) - padding`` padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        groups (int): number of blocked connections from input channels to output channels. Default: 1
        dilation (int or ints): spacing between kernel elements. Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the deconvolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return batch_deconv_explicitly(
        input,
        weight.unsqueeze(0),
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        default_space=default_space
    )


def batch_deconv_explicitly(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a deconvolution operator over an input tensor composed of several input planes explicitly
    without matrix multiplication, and the overlapping regions will be average, but the weight tensor
    and the bias tensor are batched.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(batch_size, out_channels)` or `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): ``dilation * (kernel_size - 1) - padding`` padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        groups (int): number of blocked connections from input channels to output channels. Default: 1
        dilation (int or ints): spacing between kernel elements. Default: 1
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the deconvolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return _conv_transpose_template(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        default_space=default_space,
        expander=conv_transpose_patches_explicitly,
        folder=fold_space
    )


def masked_conv_transpose_patches(
    input: Tensor,
    weight: Tensor,
    groups: int = 1,
    *,
    input_mask: Optional[Tensor] = None,
    weight_mask: Optional[Tensor] = None,
    soft_mode: bool = False
) -> Tensor:
    """
    Applies a transposed convolution operation by generating patches from input and weights
    in the mask (for input and weight) region, producing an output in patch format suitable
    for subsequent folding operations.
    
    This function performs the patch-based equivalent of a transposed convolution (also known
    as fractionally-strided convolution or deconvolution). Instead of directly producing the
    final spatial output, it generates a patch tensor that can be folded (using :func:`fold_space`
    or :func:`fold_stack`) to obtain the final upsampled result. This approach is particularly
    efficient for implementing transposed convolutions through patch operations.
    
    The function takes an input tensor and weight tensor, and computes the contribution of
    each input element to all positions in the corresponding output patch. The result is
    a patch tensor where each spatial position contains the kernel-sized contribution from
    the corresponding input element.
    
    The parameters :attr:`weight` should be batched (different weights per batch element).
    
    Args:
        input (Tensor): input tensor of shape
            `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        groups (int): number of blocked connections from input channels to output channels.
        input_mask (Tensor, optional): a mask for the input tensor of shape
            `(batch_size, in_channels, length_0, length_1, ..., length_n)`. Default: None
        weight_mask (Tensor, optional): a mask for the weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`.
            Default: None
        soft_mode (bool): If `True`, the window with part of the mask will not be discarded. Default: False
    
    Returns:
        output (Tensor): the transposed convolution patch tensor with shape
            `(batch_size, out_channels, length_0, length_1, ..., length_n, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
    
    Note:
        The output of this function is in patch format, not the final spatial output.
        To obtain the final transposed convolution result, the output should be passed
        to :func:`fold_space` or :func:`fold_stack` with appropriate stride, padding,
        and dilation parameters that match the original transposed convolution configuration.
    """
    spatial_ndim = input.ndim - 2
    mask = None
    
    input_shape = input.shape
    new_input_shape = (input.size(0), groups, -1, *input.shape[2:]) # [B, G, C_in // G, L_0, L_1, ..., L_n]
    reshaped_input = input.view(new_input_shape)
    input_permutation = (0, 1, *range(3, reshaped_input.ndim), 2) # [B, G, L_0, L_1, ..., L_n, C_in // G]
    reshaped_input = reshaped_input.permute(input_permutation)
    new_input_slice = (..., ) + (None,) * spatial_ndim
    reshaped_input = reshaped_input.unsqueeze(2)[new_input_slice] # [B, G, 1, L_0, L_1, ..., L_n, C_in // G, 1 (0), 1 (1), ..., 1 (n)]
    
    weight_shape = weight.shape
    new_weight_shape = (weight.size(0), groups, -1, weight.size(2), *weight.shape[3:]) # [B, G, C_in // G, C_out // G, k_L_0, k_L_1, ..., k_L_n]
    reshaped_weight = weight.view(new_weight_shape)
    weight_permutation = (0, 1, 3, 2, *range(4, reshaped_weight.ndim)) # [B, G, C_out // G, C_in // G, k_L_0, k_L_1, ..., k_L_n]
    reshaped_weight = reshaped_weight.permute(weight_permutation)
    new_weight_slice = (*(slice(None),) * 3, *((None,) * spatial_ndim))
    reshaped_weight = reshaped_weight[new_weight_slice] # [B, G, C_out // G, 1 (0), 1 (1), ..., 1 (n), C_in // G, k_L_0, k_L_1, ..., k_L_n]
    
    weighted_input = input * weight
    
    if input_mask is not None:
        input_mask = input_mask.expand(input_shape).view(new_input_shape).permute(input_permutation).unsqueeze(2)[new_input_slice]
        if not soft_mode:
            input_mask = input_mask.amin(tuple(range(-spatial_ndim, 0, 1)), keepdim=True).expand(input_mask.shape)
        mask = input_mask
    
    if weight_mask is not None:
        weight_mask = weight_mask.expand(weight_shape).view(new_weight_shape).permute(weight_permutation)[new_weight_slice]
        weighted_input_shape = weighted_input.shape
        if mask is None:
            mask = weight_mask.expand(weighted_input_shape)
        else:
            mask = (mask & weight_mask).expand(weighted_input_shape)
    
    output = _masked.sum(reshaped_input * reshaped_weight, dim=-spatial_ndim - 1, mask=mask).flatten(1, 2) # [B, C_out, L_0, L_1, ..., L_n, k_L_0, k_L_1, ..., k_L_n]
    return output


def masked_conv_transpose(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    input_mask: Optional[Tensor] = None,
    weight_mask: Optional[Tensor] = None,
    soft_mode: bool = False,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a transposed convolution operator over an input tensor composed of several input planes
    in the mask (for input and weight) region.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(batch_size, out_channels)` or `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): ``dilation * (kernel_size - 1) - padding`` padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        groups (int): number of blocked connections from input channels to output channels. Default: 1
        dilation (int or ints): spacing between kernel elements. Default: 1
        input_mask (Tensor, optional): a mask for the input tensor of shape
            `(batch_size, in_channels, length_0, length_1, ..., length_n)`. Default: None
        weight_mask (Tensor, optional): a mask for the weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`.
            Default: None
        soft_mode (bool): If `True`, the window with part of the mask will not be discarded. Default: False
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the transposed convolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return _conv_transpose_template(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        default_space=default_space,
        expander=masked_conv_transpose_patches,
        folder=fold_stack,
        input_mask=input_mask,
        weight_mask=weight_mask,
        soft_mode=soft_mode
    )


def masked_deconv(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    input_mask: Optional[Tensor] = None,
    weight_mask: Optional[Tensor] = None,
    soft_mode: bool = False,
    default_space: Union[float, str] = 0
) -> Tensor:
    """
    Applies a deconvolution operator over an input tensor composed of several input planes
    in the mask (for input and weight) region, and the overlapping regions will be averaged.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for all spatial dimensions
        - a ``tuple`` of many ints, corresponding to each spatial dimension

    Args:
        input (Tensor): input tensor of shape `(batch_size, in_channels, length_0, length_1, ..., length_n)`
        weight (Tensor): weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`
        bias (Tensor, optional): bias tensor of shape `(batch_size, out_channels)` or `(out_channels)`
        stride (int or ints): stride of the convolution. Default: 1
        padding (int or ints): ``dilation * (kernel_size - 1) - padding`` padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or ints): additional size added to one side of each dimension
            in the output shape. Default: 0
        groups (int): number of blocked connections from input channels to output channels. Default: 1
        dilation (int or ints): spacing between kernel elements. Default: 1
        input_mask (Tensor, optional): a mask for the input tensor of shape
            `(batch_size, in_channels, length_0, length_1, ..., length_n)`. Default: None
        weight_mask (Tensor, optional): a mask for the weight tensor of shape
            `(batch_size, in_channels, out_channels // groups, kernel_size_0, kernel_size_1, ..., kernel_size_n)`.
            Default: None
        soft_mode (bool): If `True`, the window with part of the mask will not be discarded. Default: False
        default_space (float or str): the default value or mode to fill padded regions. Default: 0
    
    Returns:
        output (Tensor): the deconvolution output tensor with shape
            `(batch_size, out_channels, out_length_0, out_length_1, ..., out_length_n)`
    """
    return _conv_transpose_template(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        default_space=default_space,
        expander=masked_conv_transpose_patches,
        folder=fold_space,
        input_mask=input_mask,
        weight_mask=weight_mask,
        soft_mode=soft_mode
    )


def multisize_conv(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    *,
    weight_box: Tensor,
    default_space: Union[float, str] = 0
) -> Tensor:
    """This function has not undergone rigorous testing and may change at any time.
    
    Args:
        input (Tensor): [B, C_in, L_0, L_1, ..., L_n]
        weight (Tensor): [B (1), C_out, C_in // groups, k_L_0, k_L_1, ..., k_L_n]
        bias (Tensor): [B (1), C_out]
        weight_box (Tensor): [B (1), C_out (1), C_in // groups (1), 2, n]
    
    Returns:
        output (Tensor): [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    """
    input, input_mask, weight, weight_mask = _multisize_kernel_preprocess(input, weight, weight_box, stride, padding, dilation)
    
    kernel_size = weight.shape[3:]
    input = unfold_space(input, kernel_size, stride, padding, dilation, default_space=default_space)
    input_mask = unfold_space(input_mask, kernel_size, stride, padding, dilation)
    output = masked_conv_patches(input, weight, groups, input_mask=input_mask, weight_mask=weight_mask)
    if bias is not None:
        output = output + bias[(..., *((None,) * (input.ndim - 2)))] # [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    return output


def multisize_masked_conv(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    *,
    weight_box: Tensor,
    input_mask: Optional[Tensor] = None,
    weight_mask: Optional[Tensor] = None,
    default_space: Union[float, str] = 0
) -> Tensor:
    """This function has not undergone rigorous testing and may change at any time.
    
    Args:
        input (Tensor): [B, C_in, L_0, L_1, ..., L_n]
        weight (Tensor): [B (1), C_out, C_in // groups, k_L_0, k_L_1, ..., k_L_n]
        bias (Tensor): [B (1), C_out]
        weight_box (Tensor): [B (1), C_out (1), C_in // groups (1), 2, n]
        input_mask (Tensor): [B (1), C_in (1), L_0, L_1, ..., L_n]
        weight_mask: [B (1), C_out (1), (C_in // groups) (1), k_L_0, k_L_1, ..., k_L_n]
    
    Returns:
        output (Tensor): [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    """
    input, content_mask, weight, box_mask = _multisize_kernel_preprocess(input, weight, weight_box, stride, padding, dilation)
    
    if input_mask is None:
        input_mask = content_mask
    else:
        input_mask = input_mask & content_mask
    if weight_mask is None:
        weight_mask = box_mask
    else:
        weight_mask = weight_mask & box_mask
    
    kernel_size = weight.shape[3:]
    input = unfold_space(input, kernel_size, stride, padding, dilation, default_space=default_space)
    input_mask = unfold_space(input_mask, kernel_size, stride, padding, dilation)
    output = masked_conv_patches(input, weight, groups, input_mask=input_mask, weight_mask=weight_mask)
    if bias is not None:
        output = output + bias[(..., *((None,) * (input.ndim - 2)))] # [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    return output


def _conv_transpose_template(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    output_padding: Union[int, Sequence[int]] = 0,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    *,
    default_space: Union[float, str] = 0,
    expander: Callable[[Tensor, Tensor, int], Tensor],
    folder: Callable[[Tensor, Union[int, Sequence[int]], Union[int, Sequence[int]], Union[int, Sequence[int]]], Tensor],
    **kwargs
) -> Tensor:
    spatial_ndim = input.ndim - 2
    
    stride = _spatialize_tuple(stride, spatial_ndim)
    
    patches = expander(input, weight, groups, **kwargs)
    output = folder(patches, stride, padding, output_padding, dilation, default_space=default_space)
    
    if bias is not None:
        output = output + bias[(..., *((None,) * spatial_ndim))] # [B, C_out, out_L_0, out_L_1, ..., out_L_n]
    
    return output


def _box_length(box: torch.Tensor) -> torch.Tensor:
    return box.diff(dim=-2).squeeze_(-2).add_(1) # [B, C, N]


def _box_mask(box: Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    batch_size, num_channels = box.size(0), box.size(1)
    spatial_ndim = box.size(-1)
    spatial_shape: Tensor = torch.as_tensor(spatial_shape, device=box.device)
    total_elements = torch.prod(spatial_shape)
    
    indices: Tensor = torch.arange(total_elements, device=box.device)
    
    # Calculate the step size for each dimension [spatial_ndim]
    reverse_cumprod = torch.cumprod(spatial_shape.flip(0), dim=0).flip(0)
    strides = torch.roll(reverse_cumprod, -1)
    strides[-1] = 1
    
    # Calculate the coordinates of each dimension [total_elements, spatial_ndim]
    indices_expanded = indices.unsqueeze(1).expand(total_elements, spatial_ndim)
    strides_expanded = strides.unsqueeze(0).expand(total_elements, spatial_ndim)
    shape_expanded = spatial_shape.unsqueeze(0).expand(total_elements, spatial_ndim)
    
    coordinates = ((indices_expanded // strides_expanded) % shape_expanded).view(*spatial_shape, spatial_ndim)
    coordinates = coordinates.view(1, 1, *spatial_shape, spatial_ndim).expand(batch_size, num_channels, -1, -1, -1)
    
    view_shape = (batch_size, num_channels, *((1,) * spatial_ndim), spatial_ndim)
    min_coordinates = box[:, :, 0, :].view(view_shape)
    max_coordinates = box[:, :, 1, :].view(view_shape)
    
    # Check if each point is located within the bounding box in all dimensions
    in_bounds = (coordinates >= min_coordinates) & (coordinates <= max_coordinates) # [B, C, L_0, L_1, ..., L_n, n]
    
    # Points must satisfy conditions in all dimensions to be within the bounding box
    mask = in_bounds.all(dim=-1).bool() # [B, C, L_0, L_1, ..., L_n]
    
    return mask


def _multisize_kernel_preprocess(
    input: Tensor,
    weight: Tensor,
    weight_box: Tensor,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Args:
        input (Tensor): [B, C_in, L_0, L_1, ..., L_n]
        weight (Tensor): [B (1), C_out, C_in // groups, k_L_0, k_L_1, ..., k_L_n]
        weight_box (Tensor): [B (1), C_out (1), C_in // groups (1), 2, n]
    
    """
    device = weight.device

    spatial_ndim = weight_box.size(-1)
    kernel_size = _box_length(weight_box) # [B, C_out, C_in // G, n]
    box_mask = _box_mask(weight_box.view(-1, 1, *weight_box.shape[-2:]), weight.shape[3:]).view_as(weight)
    if weight_box.any(weight_box[..., 0, :] != 0):
        aligned_box = torch.empty_like(weight_box)
        aligned_box[..., 0, :].fill_(0)
        aligned_box[..., 1, :].copy_(kernel_size).sub_(1)
        aligned_box_mask = _box_mask(aligned_box.view(-1, 1, *aligned_box.shape[-2:]), weight.shape[3:])
        new_weight = torch.zeros_like(weight)
        new_weight.masked_scatter_(aligned_box_mask, weight.masked_select(box_mask))
        weight = new_weight
        box_mask = aligned_box_mask
    
    spatial_shape: Tensor = torch.as_tensor(input.shape[2:], device=device)[None, None, None, ...]
    stride: Tensor = torch.as_tensor(_spatialize_tuple(stride, spatial_ndim), device=device)[None, None, None, ...]
    padding: Tensor = torch.as_tensor(_spatialize_tuple(padding, spatial_ndim), device=device)[None, None, None, ...]
    dilation: Tensor = torch.as_tensor(_spatialize_tuple(dilation, spatial_ndim), device=device)[None, None, None, ...]
    
    nature_factor = dilation * (1 - kernel_size) - 1
    out_spatial_shape = ((spatial_shape + 2 * padding + nature_factor) / stride).floor_().to(spatial_shape.dtype) # typically requires adding 1, but not necessary here
    max_dim = tuple(range(spatial_shape.ndim - 1))
    max_spatial_shape = out_spatial_shape.amax(dim=max_dim, keepdim=True)
    reverse_full_padding = max_spatial_shape * stride - nature_factor - spatial_shape
    max_full_padding = reverse_full_padding.amax(dim=max_dim)
    squeezed_spatial_shape = spatial_shape.squeeze()
    max_out_spatial_shape = squeezed_spatial_shape + max_full_padding
    
    content_mask = torch.zeros(max_out_spatial_shape.tolist(), device=input.device, dtype=torch.bool)[None, None, ...]
    content_mask[(..., *(slice(length) for length in squeezed_spatial_shape))].fill_(1)
    input = F.pad(input, [pad_value for value in reversed(max_full_padding) for pad_value in (0, value)], mode='constant', value=0)
    
    return input, content_mask, weight, box_mask