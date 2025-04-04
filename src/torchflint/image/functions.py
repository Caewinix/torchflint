import torch
import torch.nn.functional as F
from typing import Union, Sequence


def convert_to_floating_image(image: torch.Tensor):
    if image.is_floating_point():
        return image
    else:
        dtype_max = torch.iinfo(image.dtype).max
        image = image / dtype_max
        return image


def sobel_edges(image: torch.Tensor):
    """Returns a tensor holding Sobel edge maps.
    
    :param image: Image tensor with shape [batch_size, channels, height, width], expected a floating point type.
    
    :return: Tensor holding edge maps for each channel. Returns a tensor with shape
             [batch_size, channels, height, width, 2] where the last dimension holds [sobel_edge_y, sobel_edge_x].
    """
    # Sobel Filters
    kernels = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=image.dtype, device=image.device)
    kernels = kernels[:, None, :, :].repeat(image.size(1), 1, 1, 1)
    
    padded_image = F.pad(image, [1, 1, 1, 1], mode='reflect')
    output = F.conv2d(padded_image, kernels, groups=image.size(1)).view(image.size(0), image.size(1), -1, image.size(2), image.size(3)).permute(0, 1, 3, 4, 2)
    return output


def shift_image(image: torch.Tensor, shift: Union[int, Sequence[int]], fill_value = 0):
    from ..functions import shift as shift_
    from torch.nn.modules.utils import _pair
    if not (shift is int or len(shift) == 2):
        raise TypeError(f"'shift' should be either an 'int' or a pair of 'int'.")
    shift = _pair(shift)
    return shift_(image, shift, fill_value)