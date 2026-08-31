#pragma once

#include "venla/nn/feed_forward.hpp"
#include "venla/nn/layer_norm.hpp"
#include "venla/nn/multi_head_attention.hpp"
#include "venla/tensor/tensor.hpp"

#include <cstddef>

namespace venla {

// ============================================================
// TRANSFORMER ENCODER LAYER
//
// Standard Transformer Encoder:
//
//                 ┌──────────────────────┐
//                 │                      │
//                 │        Input         │
//                 │          │           │
//                 │          ▼           │
//                 │      MultiHead       │
//                 │      Attention       │
//                 │          │           │
//                 │          ▼           │
//                 │       Residual       │
//                 │          +           │
//                 │          │           │
//                 │          ▼           │
//                 │      LayerNorm       │
//                 │          │           │
//                 │          ▼           │
//                 │     FeedForward      │
//                 │          │           │
//                 │          ▼           │
//                 │       Residual       │
//                 │          +           │
//                 │          │           │
//                 │          ▼           │
//                 │      LayerNorm       │
//                 │          │           │
//                 │          ▼           │
//                 │        Output        │
//                 │                      │
//                 └──────────────────────┘
//
// Architecture:
//   x1 = LayerNorm(x + MHA(x))
//   y  = LayerNorm(x1 + FFN(x1))
//
// Supported input:
//   [seq, embed_dim]
//   [batch, seq, embed_dim]
//
// CPU / Float32.
//
// ============================================================

class TransformerEncoderLayer {
public:

    TransformerEncoderLayer(
        std::size_t embed_dim,
        std::size_t num_heads,
        std::size_t hidden_dim,
        bool use_bias = true,
        bool causal = false
    );

    // --------------------------------------------------------
    // Forward
    // --------------------------------------------------------

    Tensor forward(
        const Tensor& input
    ) const;

    // --------------------------------------------------------
    // Modules
    // --------------------------------------------------------

    const MultiHeadAttention& self_attention() const;
    MultiHeadAttention& self_attention();

    const FeedForward& feed_forward() const;
    FeedForward& feed_forward();

    const LayerNorm& norm1() const;
    LayerNorm& norm1();

    const LayerNorm& norm2() const;
    LayerNorm& norm2();

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t embed_dim() const;
    std::size_t num_heads() const;
    std::size_t hidden_dim() const;

    bool has_bias() const;
    bool is_causal() const;

    void set_causal(bool enabled);

private:

    std::size_t embed_dim_;
    std::size_t num_heads_;
    std::size_t hidden_dim_;

    bool use_bias_;

    MultiHeadAttention self_attention_;
    FeedForward feed_forward_;

    LayerNorm norm1_;
    LayerNorm norm2_;
};

} // namespace venla
