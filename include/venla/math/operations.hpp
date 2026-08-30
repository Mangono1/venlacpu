#pragma once

#include "venla/tensor/tensor.hpp"

namespace venla {

// ============================================================
// ELEMENTWISE
// ============================================================

Tensor add(const Tensor& a, const Tensor& b);

Tensor sub(const Tensor& a, const Tensor& b);

Tensor mul(const Tensor& a, const Tensor& b);

Tensor div(const Tensor& a, const Tensor& b);

Tensor neg(const Tensor& a);

// ============================================================
// REDUCTION
// ============================================================

Tensor sum(const Tensor& input);

Tensor mean(const Tensor& input);

Tensor max(const Tensor& input);

Tensor min(const Tensor& input);

// ============================================================
// LINEAR ALGEBRA
// ============================================================

Tensor dot(const Tensor& a, const Tensor& b);

Tensor transpose(const Tensor& input);

Tensor transpose(
    const Tensor& input,
    std::size_t dim0,
    std::size_t dim1
);

// ============================================================
// MATRIX MULTIPLICATION
// ============================================================

Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace venla
