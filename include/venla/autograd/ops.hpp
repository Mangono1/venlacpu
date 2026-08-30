#pragma once

#include "venla/autograd/autograd.hpp"

namespace venla {

// ============================================================
// ADD
// ============================================================

std::shared_ptr<AutogradNode>
make_add_node(
    const Tensor& a,
    const Tensor& b
);

// ============================================================
// SUB
// ============================================================

std::shared_ptr<AutogradNode>
make_sub_node(
    const Tensor& a,
    const Tensor& b
);

// ============================================================
// MUL
// ============================================================

std::shared_ptr<AutogradNode>
make_mul_node(
    const Tensor& a,
    const Tensor& b
);

// ============================================================
// DIV
// ============================================================

std::shared_ptr<AutogradNode>
make_div_node(
    const Tensor& a,
    const Tensor& b
);

// ============================================================
// NEG
// ============================================================

std::shared_ptr<AutogradNode>
make_neg_node(
    const Tensor& input
);

// ============================================================
// GENERIC BACKWARD HELPER
// ============================================================

void backward_tensor(
    const Tensor& tensor,
    const Tensor& gradient
);

// ============================================================
// REDUCTION AUTOGRAD
// ============================================================

std::shared_ptr<AutogradNode>
make_sum_node(
    const Tensor& input
);

std::shared_ptr<AutogradNode>
make_mean_node(
    const Tensor& input
);

std::shared_ptr<AutogradNode>
make_max_node(
    const Tensor& input
);

std::shared_ptr<AutogradNode>
make_min_node(
    const Tensor& input
);

// ============================================================
// MATRIX MULTIPLICATION AUTOGRAD
// ============================================================

std::shared_ptr<AutogradNode>
make_matmul_node(
    const Tensor& a,
    const Tensor& b
);

// ============================================================
// TRANSPOSE AUTOGRAD
// ============================================================

std::shared_ptr<AutogradNode>
make_transpose_node(
    const Tensor& input,
    std::size_t dim0,
    std::size_t dim1
);

} // namespace venla
