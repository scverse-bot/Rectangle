from importlib.metadata import version

from . import pp, tl
from .rectangle import ConsensusResult, load_tutorial_data, rectangle, rectangle_consens

__all__ = ["pp", "tl", "load_tutorial_data", "rectangle", "rectangle_consens", "ConsensusResult"]

__version__ = version("rectanglepy")
