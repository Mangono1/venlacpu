"""VENLACPU Python package.

CPU-first deep learning framework powered by the native C++17 core.
"""

from .version import __version__, version

try:
    from ._venlacpu import *
    _native_available = True
except ImportError:
    _native_available = False

__all__ = [
    "version",
    "__version__",
]

if _native_available:
    _native_names = [
        name
        for name in globals()
        if not name.startswith("_")
        and name not in {
            "version",
            "__version__",
        }
    ]

    __all__.extend(
        sorted(_native_names)
    )
