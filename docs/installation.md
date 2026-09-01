# Installation

## PyPI

Install VENLACPU with pip:

    python -m pip install venlacpu

Verify:

    import venlacpu
    print(venlacpu.__version__)

## Native Engine

VENLACPU uses C++17, CMake, pybind11 and scikit-build-core.

The Python package contains the native _venlacpu extension.

## Termux / Android

When a compatible Android wheel is unavailable, the source distribution can be built locally using Python, CMake and a C++17 compiler.
