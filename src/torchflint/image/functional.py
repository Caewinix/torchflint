from typing import Sequence
from numbers import Number
import torch
from torch import Tensor
import torch.nn.functional as F
from .. import functional


def convert_to_floating_image(image: Tensor) -> Tensor:
    if image.is_floating_point():
        return image
    else:
        dim = tuple(range(1, image.ndim))
        min_value = functional.amin(image, dim)
        max_value = functional.amax(image, dim)
        return (image - min_value) / (max_value - min_value)


def sobel_edges(image: Tensor) -> Tensor:
    """
    Returns a tensor holding Sobel edge maps.
    
    Args:
        image (Tensor): Image tensor with shape (batch_size, num_channels, height, width), expected a floating point type.
    
    Returns:
        output (Tensor): Tensor holding edge maps for each channel. Returns a tensor with shape
            (batch_size, num_channels, height, width, 2) where the last dimension holds (sobel_edge_y, sobel_edge_x).
    """
    # Sobel Filters
    kernels = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        dtype=image.dtype,
        device=image.device
    )
    kernels = kernels[:, None, :, :].repeat(image.size(1), 1, 1, 1)
    
    padded_image = F.pad(image, [1, 1, 1, 1], mode='reflect')
    output = F.conv2d(padded_image, kernels, groups=image.size(1)).view(image.size(0), image.size(1), -1, image.size(2), image.size(3)).permute(0, 1, 3, 4, 2)
    return output


def shift_image(input: torch.Tensor, shift: Sequence[int], fill_value: Number = 0) -> Tensor:
    empty = torch.full_like(input, fill_value=fill_value)
    slice_none = slice(None)
    shift_slices = (..., *(slice(current_shift, None, None) if current_shift > 0 else slice(None, current_shift, None) if current_shift < 0 else slice_none for current_shift in shift))
    container_slices = (..., *(slice(None, -current_shift, None) if current_shift > 0 else slice(-current_shift, None, None) if current_shift < 0 else slice_none for current_shift in shift))
    empty[container_slices] = input[shift_slices]
    return empty