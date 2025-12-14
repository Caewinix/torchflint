from torch import *
from . import nn
from . import image
from . import patchwork
from .functional import (
    buffer,
    promote_types,
    map_range,
    map_ranges,
    amin,
    amax,
    min,
    max,
    imin,
    imax,
    is_int,
    is_float,
    invert,
    shift,
    shift_,
    linspace,
    linspace_at,
    advanced_indexing,
    grow,
    apply
)