from torch.nn import Module


def refine_model(model: Module):
    try:
        return model.module
    except AttributeError:
        return model