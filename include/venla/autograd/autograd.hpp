#pragma once

#include "venla/tensor/tensor.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace venla {

// ============================================================
// AUTOGRAD NODE
//
// Menyimpan:
//   - parent tensors
//   - fungsi backward
//
// Setiap operasi differentiable membuat satu node.
// ============================================================

struct AutogradNode {

    std::vector<Tensor> parents;

    std::function<void(const Tensor&)> backward;

    AutogradNode(
        std::vector<Tensor> parents_,
        std::function<void(const Tensor&)> backward_
    )
        : parents(std::move(parents_)),
          backward(std::move(backward_)) {
    }
};

} // namespace venla
