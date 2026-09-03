"""VENLACPU Python package.

CPU-first deep learning framework powered by the native
C++17 core.
"""

from .version import __version__, version

try:
    from . import _venlacpu as _native
    from ._venlacpu import *
    _native_available = True
except ImportError:
    _native = None
    _native_available = False


# ============================================================
# VENLACPU 2.4.0 — PYTHONIC FACTORY FACADE
# ============================================================

if _native_available:
    # Explicitly expose native factory functions.
    #
    # The native extension is named "_venlacpu", so these
    # aliases make the public Python API explicit and stable.

    zeros = _native.zeros
    ones = _native.ones
    empty = _native.empty


def native_available():
    """Return True when the native VENLACPU extension is available."""
    return _native_available


def generate_text(
    model,
    tokenizer,
    text,
    config=None,
):
    """Generate text using a native VENLACPU model.

    Pipeline:

        text
          -> tokenizer.encode()
          -> Tensor
          -> native generate()
          -> token IDs
          -> tokenizer.decode()
          -> text
    """

    if not _native_available:
        raise RuntimeError(
            "VENLACPU native extension is not available."
        )

    if config is None:
        config = GenerationConfig()

    token_ids = tokenizer.encode(text)

    prompt = tensor_int32(
        [int(item) for item in token_ids]
    )

    generated = generate(
        model,
        prompt,
        config,
    )

    generated_ids = tensor_to_int64(
        generated
    )

    return tokenizer.decode(
        generated_ids
    )


__all__ = [
    "version",
    "__version__",
    "generate_text",
    "native_available",
    "zeros",
    "ones",
    "empty",
]

if _native_available:
    _native_names = [
        name
        for name in globals()
        if not name.startswith("_")
        and name not in {
            "version",
            "__version__",
            "generate_text",
            "zeros",
            "ones",
            "empty",
        }
    ]

    __all__.extend(
        sorted(_native_names)
    )
