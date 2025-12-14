try:
    from torch.masked import mean, sum, amax, amin
except:
    from typing import Optional, Union, Sequence
    import torch


    def mean(
        input: torch.Tensor,
        dim: Union[int, Sequence[int], None] = None,
        *,
        keepdim: bool = False,
        dtype: Optional[torch.dtype] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if dtype is not None:
            input = input.to(dtype=dtype)
        mask_float = mask.to(dtype=input.dtype)
        masked_data = input * mask_float 
        numerator = torch.sum(masked_data, dim=dim, keepdim=keepdim)
        denominator = torch.sum(mask_float, dim=dim, keepdim=keepdim)
        return numerator / denominator


    def sum(
        input: torch.Tensor,
        dim: Union[int, Sequence[int], None] = None,
        *,
        keepdim: bool = False,
        dtype: Optional[torch.dtype] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if dtype is not None:
            input = input.to(dtype=dtype)
        filled_data = input.masked_fill(~mask.bool(), 0.0)
        return torch.sum(filled_data, dim=dim, keepdim=keepdim)


    def amax(
        input: torch.Tensor,
        dim: Union[int, Sequence[int], None] = None,
        *,
        keepdim: bool = False,
        dtype: Optional[torch.dtype] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            return torch.amax(input, dim=dim, keepdim=keepdim)
        
        if dtype is not None:
            input = input.to(dtype=dtype)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        
        filled_input = input.masked_fill(~mask, float('-inf'))
        
        return torch.amax(filled_input, dim=dim, keepdim=keepdim)


    def amin(
        input: torch.Tensor,
        dim: Union[int, Sequence[int], None] = None,
        *,
        keepdim: bool = False,
        dtype: Optional[torch.dtype] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            return torch.amin(input, dim=dim, keepdim=keepdim)
        
        if dtype is not None:
            input = input.to(dtype=dtype)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        
        filled_input = input.masked_fill(~mask, float('inf'))
        
        return torch.amin(filled_input, dim=dim, keepdim=keepdim)