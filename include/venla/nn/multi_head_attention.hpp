#pragma once

#include "venla/nn/kv_cache.hpp"
#include "venla/tensor/tensor.hpp"

#include <cstddef>

namespace venla {

// ============================================================
// MULTI-HEAD SELF-ATTENTION
//
// Normal forward:
//
//   Input:
//     [seq, embed_dim]
//     [batch, seq, embed_dim]
//
// Cached forward:
//
//   Input:
//     [new_seq, embed_dim]
//     [batch, new_seq, embed_dim]
//
//   KVCache:
//     [past_seq, embed_dim]
//     [batch, past_seq, embed_dim]
//
// Cached inference computes:
//
//   Q = new tokens
//   K = past tokens + new tokens
//   V = past tokens + new tokens
//
// This avoids recomputing K/V for previously processed tokens.
//
// CPU / Float32.
//
// ============================================================

class MultiHeadAttention {
public:

    MultiHeadAttention(
        std::size_t embed_dim,
        std::size_t num_heads,
        bool use_bias = true,
        bool causal = false
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
    // Autoregressive inference path.
    //
    // The cache is updated in-place.
    //
    // Input may be:
    //
    //   [new_seq, embed_dim]
    //
    // or:
    //
    //   [batch, new_seq, embed_dim]
    //
    // The returned tensor has the same shape as input.
    //
    // This path is inference-only and does not construct
    // an autograd graph.
    // --------------------------------------------------------

    Tensor forward_cached(
        const Tensor& input,
        KVCache& cache
    ) const;

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------

    const Tensor& q_weight() const;
    Tensor& q_weight();

    const Tensor& k_weight() const;
    Tensor& k_weight();

    const Tensor& v_weight() const;
    Tensor& v_weight();

    const Tensor& out_weight() const;
    Tensor& out_weight();

    const Tensor& q_bias() const;
    Tensor& q_bias();

    const Tensor& k_bias() const;
    Tensor& k_bias();

    const Tensor& v_bias() const;
    Tensor& v_bias();

    const Tensor& out_bias() const;
    Tensor& out_bias();

    // --------------------------------------------------------
    // Initialization
    // --------------------------------------------------------

    void reset_parameters();

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t embed_dim() const;

    std::size_t num_heads() const;

    std::size_t head_dim() const;

    bool has_bias() const;

    bool is_causal() const;

    void set_causal(bool enabled);

private:

    std::size_t embed_dim_;

    std::size_t num_heads_;

    std::size_t head_dim_;

    bool use_bias_;

    bool causal_;

    Tensor q_weight_;
    Tensor k_weight_;
    Tensor v_weight_;
    Tensor out_weight_;

    Tensor q_bias_;
    Tensor k_bias_;
    Tensor v_bias_;
    Tensor out_bias_;
};

} // namespace venla
