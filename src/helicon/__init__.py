__version__ = "2024.09"

from os import environ, getenv
from pathlib import Path

cache_dir = (
    Path(os.getenv("HELION_CACHE_DIR"))
    if "HELION_CACHE_DIR" in environ
    else Path("~/.cache/helicon").expanduser()
)

from .lib.io import *
from .lib.util import *
from .lib.analysis import *
from .lib import dataset
from .lib import shiny
