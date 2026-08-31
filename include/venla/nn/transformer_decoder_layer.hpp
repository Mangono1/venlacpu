#pragma once

#include "venla/nn/feed_forward.hpp"
#include "venla/nn/kv_cache.hpp"
#include "venla/nn/layer_norm.hpp"
#include "venla/nn/multi_head_attention.hpp"
#include "venla/tensor/tensor.hpp"

#include <cstddef>
#include <vector>

namespace venla {

// ============================================================
// TRANSFORMER DECODER LAYER
//
// Standard decoder-style causal self-attention block.
//
// Normal:
//
//   x
//    |
//    v
//   Causal MHA
//    |
//    +---- x
//    |
//    v
//   LayerNorm
//    |
//    v
//   FeedForward
//    |
//    +---- normalized
//    |
//    v
//   LayerNorm
//    |
//    v
//   output
//
// Cached:
//
//   x_new
//      |
//      v
//   Cached Causal MHA
//      |
//      v
//   LayerNorm
//      |
//      v
//   FeedForward
//      |
//      v
//   LayerNorm
//
// The KV cache belongs to this decoder layer.
//
// Supported input:
//
//   [seq, embed_dim]
//   [batch, seq, embed_dim]
//
// CPU / Float32.
//
// ============================================================

class TransformerDecoderLayer {
public:

    TransformerDecoderLayer(
        std::size_t embed_dim,
        std::size_t num_heads,
        std::size_t hidden_dim,
        bool use_bias = true
    );

    // --------------------------------------------------------
    // Normal Forward
    // --------------------------------------------------------

    Tensor forward(
        const Tensor& input
    ) const;

    // --------------------------------------------------------
    // Cached Forward
    //
    // Inference-only.
    //
    // The supplied KVCache belongs exclusively to this layer.
    //
    // Input:
    //
    //   [new_seq, embed_dim]
    //   [batch, new_seq, embed_dim]
    //
    // Output has the same shape.
    // --------------------------------------------------------

    Tensor forward_cached(
        const Tensor& input,
        KVCache& cache
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
    // Parameters
    // --------------------------------------------------------
    //
    // Mengembalikan seluruh parameter trainable layer:
    //
    //   MultiHeadAttention
    //   FeedForward
    //   LayerNorm 1
    //   LayerNorm 2
    //
    // Bias MHA dan FFN hanya dikembalikan jika menggunakan bias.
    // --------------------------------------------------------

    std::vector<Tensor*> parameters();

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t embed_dim() const;

    std::size_t num_heads() const;

    std::size_t hidden_dim() const;

    bool has_bias() const;

    bool is_causal() const;

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
