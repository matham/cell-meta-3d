try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from .napari.widget import (
    analyse_widget,
)

__all__ = ("analyse_widget",)
