from torch import nn
from ..utils.typing import overload
from ..utils._helper import __module__


@__module__(__package__)
class Module(nn.Module):
    @overload
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        nn.Module.__init__(instance)
        return instance
    
    def _replicate_for_data_parallel(self):
        self.__new__ = nn.Module.__new__
        super()._replicate_for_data_parallel()
        del self.__new__