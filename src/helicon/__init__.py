__version__ = "2024.11"

from .lib.util import setup_cache_dir

cache_dir = setup_cache_dir()

from .lib.analysis import *
from .lib.filters import *
from .lib.io import *
from .lib.transforms import *
from .lib.util import *

from .lib import dataset

try:
    from .lib import shiny
except:
    pass
