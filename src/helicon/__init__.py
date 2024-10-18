__version__ = "2024.09"

from os import environ, getenv
from pathlib import Path

cache_dir = (
    Path(os.getenv("HELION_CACHE_DIR"))
    if "HELION_CACHE_DIR" in environ
    else Path("~/.cache/helicon").expanduser()
)

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
