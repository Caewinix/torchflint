from typing import Optional
import torch
from torch.nn import Module, DataParallel
from torch.nn.parallel import DistributedDataParallel


def refine_model(model: Module):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        return model.module
    else:
        return model