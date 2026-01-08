from torch.nn.functional import *


def refine_model(model):
    try:
        return model.module
    except AttributeError:
        return model