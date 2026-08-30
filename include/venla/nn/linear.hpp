#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>

namespace venla {

// ============================================================
// LINEAR / DENSE LAYER
//
// y = xW + b
//
// Weight:
//   [in_features, out_features]
//
// Bias:
//   [out_features]
//
// Supported input ranks:
//   1D -> 1D
//   2D -> 2D
//   3D -> 3D
//   4D -> 4D
//
// General rule:
//
//   [..., in_features]
//          @
//   [in_features, out_features]
//
//   -> [..., out_features]
//
// Parameters:
//   Weight uses Xavier/Glorot uniform initialization.
//   Bias is initialized to zero.
//
// ============================================================

class Linear {
public:

    Linear(
        std::size_t in_features,
        std::size_t out_features,
        bool use_bias = true
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

    std::size_t in_features() const;

    std::size_t out_features() const;

    bool has_bias() const;

private:

    std::size_t in_features_;

    std::size_t out_features_;

    bool use_bias_;

    Tensor weight_;

    Tensor bias_;
};

} // namespace venla
