# VENLACPU

**CPU-first deep learning framework modern C++17.**

VENLACPU is an actively developed deep learning framework focused on a clean, portable CPU implementation. The native engine is written in C++17 and is designed to work across desktop and mobile environments without requiring CUDA.

> **Status:** Alpha / active development. APIs may change before the first stable release.

## Author

VENLACPU is created and developed by **Frandika Imam Arifin**.

## Current capabilities

The current C++ core includes:

- Tensor storage, shape, stride, dtype, and device abstractions
- CPU tensor operations and manipulation
- Autograd and gradient propagation
- Elementwise mathematics and reductions
- Linear layers and sequential models
- Activation functions
- MSE and cross-entropy loss
- Embedding
- Positional encoding
- Layer normalization
- Multi-head attention
- Feed-forward networks
- Transformer encoder
- Transformer decoder
- KV cache and cached attention
- Causal language model
- Causal language-model dataset batching
- Training loop with gradient accumulation
- Adam optimizer
- Evaluation and autoregressive generation
- Tokenizer and vocabulary components

The project currently contains a comprehensive C++ test suite covering the implemented components.

## Architecture

```text
Physical CPU engine
           |
          T
ensor / Storage / Shape / Stride / Dtype / Device
           |
          Autograd
          |
          Math Ops
           |
          Neural network
          |
      Transformer / Language Model
          |
          Training System
```

The native implementation is the primary implementation at this stage.

## Build from source

Requirements:

- CMake 3.16 or newer
- C++17 compiler
- Git

```bash
cmake -S . -B build
cmake --build build -j2
```

Run the complete test suite:

```bash
ctest --test-dir build --output-on-failure
```

## Python / PyPI

VENLACPU has a modular Python package prepared for PyPI. The current Python package exposes the package version, while the native C++ engine remains the primary implementation.

Platform-specific native wheels will be added when the Python binding layer is ready. GitHub Actions is used as the build and packaging engine.

## Causal language-model training

The current training API supports:

- Causal token shifting
- Dataset batching
- Padding with ignore index
- Cross-entropy loss
- Autograd and gradient propagation
- Gradient accumulation
- Adam optimization
- Evaluation
- Autoregressive generation

## CPU-first design

VENLACPU is intentionally CPU-first. The current implementation does not require CUDA or a GPU runtime.

This makes the project suitable for desktop CPUs and ARM/mobile development environments.

## Repository layout

```text
venlacpu/
└ ── include/venla/
    └ ── autograd/
    │      core/
    └        math/
    └        nn/
    └ ── optim/
    │        tensor/
    └ ── tokenizer/
    └        training/
  │   src/
    │      core/
    └        math/
    └        nn/
    └ ── optim/
    │      tensor/
    │      tokenizer/
    │      training/
  │   tests/
  │  examples/
   │  benchmarks/
   │   docs/
   └ ── python/
   │  CMakeLists.tx
  │  pyproject.toml
   └ ── README.md
```

## Development philosophy

VENLACPU begins with the low-level foundations and builds upward. The long-term goal is a portable CPU-first framework with a native C++ core and convenient Python access.

## License

The project is currently in alpha and the final public-release license is being finalized.

## Links

- Repository: https://github.com/Mangono1/venlacpu
- Issues: https://github.com/Mangono1/venlacpu/issues
