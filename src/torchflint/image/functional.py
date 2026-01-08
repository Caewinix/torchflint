from typing import Sequence, Union, Tuple, Literal, Optional
from itertools import chain
from numbers import Number
import torch
from torch import Tensor
import torch.nn.functional as F
from .. import functional
from ..utils._helper import __module__


def center_padding(old_spatial_shape: Sequence[int], new_spatial_shape: Sequence[int]) -> Sequence[int]:
    """
    Computes the padding values required to center an input with given spatial dimensions within a new
    target spatial shape.
    
    This function calculates symmetric padding for each spatial dimension such that the original content
    remains centered after padding. The padding is distributed equally on both sides when possible, with
    any remainder allocated to the end (negative) side of each dimension.
    
    Args:
        old_spatial_shape (ints): the original spatial dimensions of the input tensor
        new_spatial_shape (ints): the target spatial dimensions after padding
    
    Returns:
        padding (ints): a sequence of padding values in the format required by `torch.nn.functional.pad`,
            where each pair of consecutive elements represents padding for the beginning and end of a spatial dimension
            in reverse order (starting from the last spatial dimension).
    """
    spatial_shape_difference = [new_length - old_length for new_length, old_length in zip(new_spatial_shape, old_spatial_shape)]
    padding_positive_directions = [difference // 2 for difference in spatial_shape_difference]
    padding_negative_directions = [difference - start for difference, start in zip(spatial_shape_difference, padding_positive_directions)]
    padding = [value for pair in zip(reversed(padding_positive_directions), reversed(padding_negative_directions)) for value in pair]
    return padding


@__module__(__package__)
def center_expand(
    input: Tensor,
    spatial_shape: Sequence[int],
    mode: Literal['constant', 'reflect', 'replicate', 'circular'] = 'constant',
    value: Number = 0
) -> Tensor:
    """
    Pads the input tensor to achieve the specified spatial dimensions while keeping the original content centered.
    
    This function symmetrically pads the input tensor's spatial dimensions to match the target shape,
    distributing padding equally on both sides when possible. The operation is the inverse of :func:`center_crop`
    when used with the same target dimensions, though padding mode and value can be customized.
    
    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, *spatial_dims)`
        spatial_shape (ints): the target spatial dimensions after expansion
        mode (str): padding mode to use, as accepted by `torch.nn.functional.pad`. 
            Options include 'constant', 'reflect', 'replicate', 'circular'. Default: 'constant'
        value (Number): fill value for constant padding. Default: 0
    
    Returns:
        output (Tensor): a tensor with the same batch and channel dimensions as input, and spatial dimensions
            equal to `spatial_shape`, with original content centered within the padded region
    """
    return F.pad(input, center_padding(input.shape[2:], spatial_shape), mode, value)


@__module__(__package__)
def center_crop(input: Tensor, spatial_shape: Sequence[int]) -> Tensor:
    """
    Extracts a centered region of specified spatial dimensions from the input tensor.
    
    The region is taken from the center of each spatial dimension. If the requested spatial shape
    is larger than the input in any dimension, the behavior is to crop the available region
    (effectively a no-op for that dimension).
    
    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        spatial_shape (ints): the target spatial dimensions for the cropped output
    
    Returns:
        output (Tensor): a tensor containing the centered region with shape
            `(batch_size, num_channels, length_0, length_1, ..., length_n)`
    """
    centers = [old_length // 2 for old_length in input.shape]
    slice_starts = [center - new_length // 2 for center, new_length in zip(centers, spatial_shape)]
    return input[(..., *(slice(start, start + new_length) for start, new_length in zip(slice_starts, spatial_shape)))]


@__module__(__package__)
def convert_to_floating_image(image: Tensor) -> Tensor:
    """
    Converts an input tensor to a floating-point image with values normalized to the range [0, 1].
    
    If the input is already a floating-point tensor, it is returned unchanged. Otherwise, for integer
    tensors, each channel is independently normalized by its minimum and maximum values, mapping
    the original values to the [0, 1] range.
    
    Args:
        image (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
    
    Returns:
        output (Tensor): a floating-point tensor with the same shape as input, with values in [0, 1]
    """
    if image.is_floating_point():
        return image
    else:
        dim = tuple(range(1, image.ndim))
        min_value = functional.amin(image, dim)
        max_value = functional.amax(image, dim)
        return (image - min_value) / (max_value - min_value)


@__module__(__package__)
def sobel_edges(input: Tensor) -> Tensor:
    """
    Returns a tensor holding Sobel edge maps.
    
    Args:
        input (Tensor): tensor with shape (batch_size, num_channels, height, width), expected a floating point type
    
    Returns:
        output (Tensor): tensor holding edge maps for each channel, with shape
            (batch_size, num_channels, height, width, 2) where the last dimension holds (sobel_edge_y, sobel_edge_x).
    """
    # Sobel Filters
    kernels = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=input.dtype,
        device=input.device
    )
    kernels = kernels[:, None, :, :].repeat(input.size(1), 1, 1, 1)
    
    padded_image = F.pad(input, [1, 1, 1, 1], mode='reflect')
    output = F.conv2d(padded_image, kernels, groups=input.size(1)).view(input.size(0), input.size(1), -1, input.size(2), input.size(3)).permute(0, 1, 3, 4, 2)
    return output


@__module__(__package__)
def shift_image(input: Tensor, shift: Sequence[int], fill_value: Number = 0) -> Tensor:
    """
    Translates the content of the input tensor by specified offsets in each spatial dimension.
    
    The tensor is shifted along its spatial dimensions, with newly exposed regions filled with
    a constant value. The shift values can be positive (shifting content towards the end) or
    negative (shifting content towards the beginning) for each dimension.
    
    Args:
        input (Tensor): input tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
        shift (ints): offsets for each spatial dimension, with one value per dimension
        fill_value (Number): value to fill the empty regions created by the shift. Default: 0
    
    Returns:
        output (Tensor): a tensor of the same shape as input, containing the shifted content
    """
    empty = torch.full_like(input, fill_value=fill_value)
    slice_none = slice(None)
    shift_slices = (..., *(slice(current_shift, None, None) if current_shift > 0 else slice(None, current_shift, None) if current_shift < 0 else slice_none for current_shift in shift))
    container_slices = (..., *(slice(None, -current_shift, None) if current_shift > 0 else slice(-current_shift, None, None) if current_shift < 0 else slice_none for current_shift in shift))
    empty[container_slices] = input[shift_slices]
    return empty


@__module__(__package__)
def get_bounding_box(mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the bounding boxes enclosing all True/active regions in a boolean mask tensor.
    
    For each channel and batch element, the function finds the minimum and maximum coordinates
    where the mask is True along each spatial dimension. The result represents the tightest
    axis-aligned bounding box containing all active pixels.
    
    Args:
        mask (Tensor): a boolean mask tensor of shape `(batch_size, num_channels, length_0, length_1, ..., length_n)`
    
    Returns:
        box (Tensor): a tensor of shape `(batch_size, num_channels, 2, n)` where the
            third dimension (size 2) holds [min_coordinate, max_coordinate] pairs for each spatial dimension.
            If no active pixels are found in a channel, the corresponding box is set to (0, -1).
    """
    device = mask.device
    
    mask_shape = mask.shape
    batch_size, num_channels = mask_shape[:2]
    spatial_shape = mask_shape[2:]
    spatial_ndim = len(spatial_shape)
    
    max_coordinate = torch.arange(max(spatial_shape), device=device)
    coordinates = [
        max_coordinate[:length].view(1, 1, *(1 if j != i else length for j in range(spatial_ndim))).expand(mask_shape)
        for i, length in enumerate(spatial_shape)
    ] # Generate the coordinate grids
    min_ranges = [torch.where(mask, coordinate, length).flatten(2).amin(dim=2) for coordinate, length in zip(coordinates, spatial_shape)]
    max_ranges = [torch.where(mask, coordinate, -1).flatten(2).amax(dim=2) for coordinate in coordinates]
    
    box = torch.stack(tuple(chain(*zip(min_ranges, max_ranges))), dim=2).view(batch_size, num_channels, -1, 2).permute(0, 1, 3, 2) # [B, C, 2, spatial_ndim]
    box.masked_fill_(box.isposinf(), 0)
    box.masked_fill_(box.isneginf(), -1)
    return box


@__module__(__package__)
def get_box_length(box: torch.Tensor) -> torch.Tensor:
    """
    Computes the side lengths of bounding boxes from their min-max coordinate representation.
    
    The length for each spatial dimension is calculated as (max - min + 1), which represents
    the inclusive count of positions along that dimension within the bounding box.
    
    Args:
        box (Tensor): bounding box tensor of shape `(batch_size, num_channels, 2, spatial_ndim)`
            where the third dimension holds [min, max] coordinate pairs
    
    Returns:
        length (Tensor): a tensor of shape `(batch_size, num_channels, spatial_ndim)` containing
            the length (size) of each bounding box along each spatial dimension
    """
    return box.diff(dim=-2).squeeze_(-2) + 1 # [B, C, spatial_ndim]


@__module__(__package__)
def box_crop(
    box: torch.Tensor,
    *input: torch.Tensor,
    length: Union[int, Sequence[int], torch.Tensor, None] = None
) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]]:
    """
    Extracts fixed-size regions from input tensors based on bounding boxes, with automatic centering and
    boundary handling.
    
    This function crops multiple input tensors to regions defined by bounding boxes, optionally resizing to a
    specified length. If multiple boxes are provided (batch and channel dimensions), each box is used to crop
    the corresponding batch and channel slice. The cropped regions are centered within the boxes and adjusted
    to stay within original spatial boundaries.
    
    Args:
        box (Tensor): bounding box tensor of shape `(batch_size, num_channels, 2, n)`,
            where the third dimension contains [min_coordinate, max_coordinate] pairs for each spatial dimension
        *input (Tensor): one or more input tensors of shape
            `(batch_size, num_channels, length_0, length_1, ..., length_n)` to be cropped
        length (int, ints, or Tensor, optional): the target size for the cropped regions. If a scalar,
            it applies to all spatial dimensions; if ints, specifies each dimension separately. If None,
            the maximum box length across all boxes is used. Default: None
    
    Returns:
        output (Tensor or Tensors): the cropped region(s) with shape
            `(batch_size, num_channels, length_0, length_1, ..., length_n)` for each input tensor.
                Returns a single tensor if only one input is provided, otherwise a tuple of tensors
        meta (Tuple): metadata containing coordinate information needed for reconstruction via :func:`box_uncrop`.
            The tuple includes the original coordinates and spatial shape, enabling exact inverse operation
    """
    device = box.device
    batch_size = box.size(0)
    num_channels = box.size(1)
    num_lengths = box.size(-1)
    
    length_shape = torch.as_tensor(input[0].shape[2:], device=device)
    length_ndim = len(length_shape)
    canvas_length = length_shape[None, None, :]
    
    min_index = box[..., 0, :] # [B, C, N]
    max_index = box[..., 1, :] # [B, C, N]
    box_length = max_index - min_index + 1 # [B, C, N]
    
    if length is None:
        max_length = box_length.flatten(0, 1).amax(dim=0) # max_length: [N]
    else:
        if isinstance(length, int):
            length = torch.tensor([length], dtype=box.dtype, device=device)
        elif isinstance(length, Sequence) and not isinstance(length, torch.Tensor):
            length = torch.as_tensor(length, dtype=box.dtype, device=device)
        max_length = length
    max_length = max_length[None, None, :]
    
    # Calculate cropping parameters
    center = (min_index + max_index) / 2 # [B, C, N]
    start = (center - max_length / 2).ceil_().to(box.dtype).clamp(min=0) # [B, C, N]
    max_length_mask = start + max_length > canvas_length
    if torch.any(max_length_mask):
        start[max_length_mask] = (canvas_length - max_length).broadcast_to(start.shape)[max_length_mask]
    
    # Generate coordinate grid
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, *((1,) * length_ndim))
    length_indices = tuple(torch.arange(length.item(), device=device).view(1, 1, *(1 if j != i else length for j in range(length_ndim))) for i, length in enumerate(max_length.squeeze().broadcast_to(torch.Size((num_lengths,)))))
    
    # Calculate the original coordinates
    original_coordinates = tuple((indices + current_start[(..., *((None,) * length_ndim))]).clamp(max=length - 1) for indices, current_start, length in zip(length_indices, start.permute(2, 0, 1), length_shape))
    
    # Use advanced indexing to obtain cropped regions
    def yield_each_input():
        if num_channels == 1:
            for each in input:
                current_num_channels = each.size(1)
                current_channel_indices = torch.arange(current_num_channels, device=device).view(1, current_num_channels, *((1,) * length_ndim))
                yield each[(batch_indices, current_channel_indices, *original_coordinates)]
        else:
            channel_indices = torch.arange(num_channels, device=device).view(1, num_channels, *((1,) * length_ndim))
            for each in input:
                yield each[(batch_indices, channel_indices, *original_coordinates)]
    result = tuple(yield_each_input())
    if len(result) == 1:
        result = result[0]
    return result, (original_coordinates, length_shape)


@__module__(__package__)
def box_uncrop(
    input: torch.Tensor,
    meta: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int],
    *,
    background: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Reconstructs a full-sized tensor by placing cropped regions back into their original spatial positions.
    
    This is the inverse operation to :func:`box_crop`. The function takes cropped regions (output from :func:`box_crop`)
    and metadata describing their original positions, and reconstructs the full tensor by placing these
    regions at the appropriate coordinates. Areas not covered by any crop are filled with a background value.
    
    Args:
        input (Tensor): the cropped tensor(s) to be placed back, typically the output from :func:`box_crop`
        meta (Tuple): metadata returned by :func:`box_crop` containing coordinate information for reconstruction
        background (Tensor, optional): a background tensor to fill regions not covered by crops. If not provided,
            a zero tensor of appropriate shape is created. Default: None
    
    Returns:
        output (Tensor): the reconstructed tensor with cropped regions placed at their original positions
    """
    device = input.device
    original_coordinates, length_shape = meta
    
    batch_size, num_channels = input.shape[:2]
    
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1, 1)
    channel_indices = torch.arange(num_channels, device=device).view(1, num_channels, 1, 1)
    
    if background is None:
        background = torch.zeros(batch_size, num_channels, *length_shape, dtype=input.dtype, device=input.device)
    background[(batch_indices, channel_indices, *original_coordinates)] = input
    return background