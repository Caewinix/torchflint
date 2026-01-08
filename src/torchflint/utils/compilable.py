from itertools import chain
import torch
from .typing import overload, Iterator
from .devices import supported_devices


try:
    from torch import compile
    
    
    try:
        from torch._dynamo import config as dynamo_config
        def recompile_limit() -> int: return dynamo_config.cache_size_limit
    except:
        def recompile_limit() -> int: return 8
    
    
    @compile
    def _empty(device: str) -> torch.Tensor: return torch.empty([], device=device)


    def _check_device_compilable(device: str) -> bool:
        try:
            _empty(device)
            return True
        except:
            return False
    
    
    _is_cpu_compilable = _check_device_compilable("cpu")
    
    
    try:
        from torch._inductor import utils
        _compilable_gpus = set(utils.GPU_TYPES)
    except AttributeError:
        try:
            from torch._dynamo import device_interface as _
            _compilable_gpus = {"cuda", "xpu"}
        except ImportError:
            _compilable_gpus = {"cuda"}
        except Exception as e: raise e from None
    except Exception:
        _compilable_gpus = set(device for device in supported_devices() if device != "cpu" and _check_device_compilable(device))
    
    
    def compilable_gpus() -> Iterator[str]:
        return iter(_compilable_gpus)
    
    
    def compilable_devices() -> Iterator[str]:
        if _is_cpu_compilable:
            return chain(("cpu",), compilable_gpus())
        else:
            return compilable_gpus()
    
    
    def is_device_compilable(device: str) -> bool:
        if device == 'cpu':
            return _is_cpu_compilable
        else:
            return device in _compilable_gpus
except:
    def recompile_limit() -> int: return 0
    
    
    def compilable_gpus() -> Iterator[str]: return (None for _ in range(0))
    
    
    def compilable_devices() -> Iterator[str]: return (None for _ in range(0))
    
    
    @overload
    def is_device_compilable(device: str) -> bool: return False
    
    
    @overload
    def compile(model = None, **kwargs):
        if model is None:
            def compiler(model): return model
            return compiler
        return model