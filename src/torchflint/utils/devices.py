from typing import overload, Iterator, Callable, Any, Optional
import torch


_broad_supported_devices = ['cpu', 'cuda', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 'msnpu', 'xla']
_supported_devices = []
def supported_devices() -> Iterator[str]:
    global _supported_devices
    if not _supported_devices:
        try:
            torch.device("_")
        except RuntimeError as e:
            try:
                msg = e.args[0]
                _supported_devices = [device.strip() for device in msg[16:-41].split(',') if device.strip()]
            except:
                _supported_devices = _broad_supported_devices
    return iter(_supported_devices)


class multidevices:
    __default__: Callable[..., Any]
    def __new__(cls, function: Callable[..., Any]):
        instance = super().__new__(cls)
        instance.__default__ = function
        instance.__functions = {}
        return instance
    
    @property
    def __code__(self):
        return self.__default__.__code__
    
    def __getitem__(self, device: str):
        return self.get(device)
    
    def __call__(self, *args, **kwargs):
        return self.__default__(*args, **kwargs)
    
    def get(self, device: str):
        return self.__functions.get(device, self.__default__)
    
    @overload
    def register_device(self, device: str, function: Callable[..., Any]):
        """It is used to register the function as the given device.
        """
        ...
    
    @overload
    def register_device(self, device: str):
        """It is a decorator for registering the function as the given device.
        """
        ...
    
    def register_device(self, device: str, function: Optional[Callable[..., Any]] = None):
        # Decorator Mode
        if function is None:
            def registration(function: Callable[..., Any]):
                return self.__registration(device, function)
            return registration
        elif callable(function):
            return self.__registration(device, function)
    
    def __registration(self, device: str, function: Callable[..., Any]):
        try:
            if getattr(torch, device).is_available():
                self.__functions[device] = function
                function.__manager__ = self
        except: ...
        return function