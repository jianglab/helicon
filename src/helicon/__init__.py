__version__ = "2024.09"

import os
from pathlib import Path

cache_dir = (
    Path(os.getenv("HELION_CACHE_DIR"))
    if "HELION_CACHE_DIR" in os.environ
    else Path.home() / ".cache" / "helicon"
)
if not os.access(cache_dir, os.W_OK):
    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / Path(os.getlogin()) / "helicon_cache"
if not cache_dir.exists():
    cache_dir.mkdir(parents=True, exist_ok=True)

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
