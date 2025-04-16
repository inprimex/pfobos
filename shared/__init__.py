# ./shared/__init__.py
from .fwrapper import FobosSDR, FobosException

# You can define __all__ to control what gets imported with "from shared import *"
__all__ = ['FobosSDR', 'FobosException']