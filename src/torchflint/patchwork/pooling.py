from typing import Union, Sequence, Optional, Callable
import torch
from torch import Tensor
from .patches import unfold_space


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
    spatial_ndim = input.ndim - 2
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode, default_space=default_space)
    return reducer(output, tuple(range(-spatial_ndim, 0, 1)))


def max_pool(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False
) -> Tensor:
    spatial_ndim = input.ndim - 2
    # import torch.nn.functional as F
    # if spatial_ndim == 2:
    #     return F.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    # elif spatial_ndim == 3:
    #     return F.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)
    # elif spatial_ndim == 1:
    #     return F.max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode, default_space=float("-inf"))
    return output.amax(tuple(range(-spatial_ndim, 0, 1)))


def min_pool(
    input: Tensor,
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
    ceil_mode: bool = False
) -> Tensor:
    spatial_ndim = input.ndim - 2
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode, default_space=float("inf"))
    return output.amin(tuple(range(-spatial_ndim, 0, 1)))


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
    spatial_ndim = input.ndim - 2
    # if dilation == 1 or (len(dilation) == spatial_ndim and all(value == 1 for value in dilation)):
    #     import torch.nn.functional as F
    #     if spatial_ndim == 2:
    #         return F.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    #     elif spatial_ndim == 3:
    #         return F.avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    #     elif spatial_ndim == 1 and divisor_override is None:
    #         return F.avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    dim = tuple(range(-spatial_ndim, 0, 1))
    output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode)
    if divisor_override is None:
        if count_include_pad:
            return output.mean(dim)
        else:
            output = unfold_space(input, kernel_size, stride, padding, dilation, ceil_mode)
            mask = unfold_space(
                torch.ones((1,) * input.ndim, device=input.device, dtype=torch.bool).expand(input.size(0), 1, *input.shape[2:]),
                kernel_size, stride, padding, dilation, ceil_mode
            )
            return torch.masked.mean(output, dim, mask=mask)
    else:
        return output.sum(dim) / divisor_override