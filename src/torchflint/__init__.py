from torch import *
from . import nn
from . import image
from . import patchwork
from .functional import (
    buffer,
    is_int,
    is_float,
    shift,
    shift_,
    promote_types,
    min,
    max,
    map_range,
    map_ranges,
    linspace,
    linspace_at,
    gamma,
    gamma_div,
    invert,
    advanced_indexing,
    grow,
    apply_from_dim
)