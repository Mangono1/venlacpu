#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>

namespace venla {

// ============================================================
// LAYER NORMALIZATION
//
// Normalizes the last dimension of the input.
//
// Formula:
//
//   mean = mean(x)
//
//   variance = mean((x - mean)^2)
//
//   normalized =
//       (x - mean) / sqrt(variance + eps)
//
//   output =
//       gamma * normalized + beta
//
// Input:
//
//   [features]
//   [batch, features]
//   [batch, seq, features]
//   [d1, d2, ..., features]
//
// Normalization is always performed over the last dimension.
//
// Parameters:
//
//   gamma: [normalized_shape]
//   beta : [normalized_shape]
//
// Both parameters require gradients.
// ============================================================

class LayerNorm {
public:

    LayerNorm(
        std::size_t normalized_shape,
        float eps = 1e-5f
    );

    // --------------------------------------------------------
    // Forward
    // --------------------------------------------------------

    Tensor forward(
        const Tensor& input
    ) const;

    // --------------------------------------------------------
    // Parameter initialization
    // --------------------------------------------------------

    void reset_parameters();

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------

    const Tensor& weight() const;

    Tensor& weight();

    const Tensor& bias() const;

    Tensor& bias();

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t normalized_shape() const;

    float eps() const;

private:

    std::size_t normalized_shape_;

    float eps_;

    Tensor weight_;

    Tensor bias_;
};

} // namespace venla

