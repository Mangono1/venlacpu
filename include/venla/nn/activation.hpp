#pragma once

#include "venla/tensor/tensor.hpp"

namespace venla {

// ============================================================
// ACTIVATION FUNCTIONS
//
// Supported:
//   ReLU
//   Sigmoid
//   Tanh
//
// Input:
//   Any Tensor rank >= 0
//
// Currently supported:
//   Float32
//   CPU
//
// All functions return a new Tensor.
// The input Tensor is never modified.
// ============================================================

// ------------------------------------------------------------
// ReLU
//
// y = max(0, x)
// ------------------------------------------------------------

Tensor relu(
    const Tensor& input
);

// ------------------------------------------------------------
// Sigmoid
//
// y = 1 / (1 + exp(-x))
//
// Numerically stable implementation.
// ------------------------------------------------------------

Tensor sigmoid(
    const Tensor& input
);

// ------------------------------------------------------------
// Tanh
//
// y = tanh(x)
// ------------------------------------------------------------

Tensor tanh(
    const Tensor& input
);

} // namespace venla
