from torch import nn


class Module(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        nn.Module.__init__(instance)
        return instance
    
    def _replicate_for_data_parallel(self):
        self.__new__ = nn.Module.__new__
        super()._replicate_for_data_parallel()
        del self.__new__