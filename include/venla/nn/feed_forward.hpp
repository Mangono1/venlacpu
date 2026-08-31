#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>

namespace venla {

// ============================================================
// TRANSFORMER FEED FORWARD NETWORK
//
// Standard Transformer MLP:
//
//   x
//    |
//   Linear(embed_dim, hidden_dim)
//    |
//   GELU
//    |
//   Linear(hidden_dim, embed_dim)
//    |
//   output
//
// Supported input:
//   [seq, embed_dim]
//   [batch, seq, embed_dim]
//
// Parameters:
//
//   input_weight  : [embed_dim, hidden_dim]
//   input_bias    : [hidden_dim]
//   output_weight : [hidden_dim, embed_dim]
//   output_bias   : [embed_dim]
//
// CPU / Float32.
//
// ============================================================

class FeedForward {
public:

    FeedForward(
        std::size_t embed_dim,
        std::size_t hidden_dim,
        bool use_bias = true
    );

    // --------------------------------------------------------
    // Forward
    // --------------------------------------------------------

    Tensor forward(
        const Tensor& input
    ) const;

    // --------------------------------------------------------
    // Initialization
    // --------------------------------------------------------

    void reset_parameters();

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------

    const Tensor& input_weight() const;
    Tensor& input_weight();

    const Tensor& input_bias() const;
    Tensor& input_bias();

    const Tensor& output_weight() const;
    Tensor& output_weight();

    const Tensor& output_bias() const;
    Tensor& output_bias();

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t embed_dim() const;

    std::size_t hidden_dim() const;

    bool has_bias() const;

private:

    std::size_t embed_dim_;

    std::size_t hidden_dim_;

    bool use_bias_;

    Tensor input_weight_;
    Tensor input_bias_;

    Tensor output_weight_;
    Tensor output_bias_;
};

} // namespace venla
