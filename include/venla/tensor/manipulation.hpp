#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>
#include <vector>

namespace venla {

// ============================================================
// RESHAPE
// ============================================================

Tensor reshape(
    const Tensor& input,
    const Shape& shape
);

Tensor reshape(
    const Tensor& input,
    std::initializer_list<std::size_t> shape
);

// ============================================================
// FLATTEN
// ============================================================

Tensor flatten(
    const Tensor& input
);

Tensor flatten(
    const Tensor& input,
    std::size_t start_dim,
    std::size_t end_dim
);

// ============================================================
// SQUEEZE
// ============================================================

Tensor squeeze(
    const Tensor& input
);

Tensor squeeze(
    const Tensor& input,
    std::size_t dim
);

// ============================================================
// UNSQUEEZE
// ============================================================

Tensor unsqueeze(
    const Tensor& input,
    std::size_t dim
);

// ============================================================
// CONCATENATE
//
// Example:
//   concatenate({a, b}, 0)
// ============================================================

Tensor concatenate(
    const std::vector<Tensor>& tensors,
    std::size_t dim
);

// ============================================================
// STACK
//
// Example:
//   stack({a, b}, 0)
// ============================================================

Tensor stack(
    const std::vector<Tensor>& tensors,
    std::size_t dim
);

// ============================================================
// INDEX
//
// Multi-dimensional indexing.
//
// Example:
//   index(x, {1, 2})
//
// Returns a scalar Tensor.
// ============================================================

Tensor index(
    const Tensor& input,
    const std::vector<std::size_t>& indices
);

// Convenience single-axis index for 1D tensors.

Tensor index(
    const Tensor& input,
    std::size_t position
);

// ============================================================
// SLICE
//
// [start, stop) with positive step.
//
// Example:
//   slice(x, 1, 4, 0)
//
// means dimension 0, indices 1,2,3.
//
// ============================================================

Tensor slice(
    const Tensor& input,
    std::size_t dim,
    std::size_t start,
    std::size_t stop,
    std::size_t step = 1
);

} // namespace venla
