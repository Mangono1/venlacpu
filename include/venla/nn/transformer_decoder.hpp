#pragma once

#include "venla/nn/transformer_decoder_layer.hpp"
#include "venla/tensor/tensor.hpp"

#include <cstddef>
#include <vector>

namespace venla {

// ============================================================
// TRANSFORMER DECODER STACK
//
// Stack of N TransformerDecoderLayer modules.
//
//
//
//                 Input
//                   |
//                   v
//          +-------------------+
//          | Decoder Layer 1   |
//          +-------------------+
//                   |
//                   v
//          +-------------------+
//          | Decoder Layer 2   |
//          +-------------------+
//                   |
//                  ...
//                   |
//                   v
//          +-------------------+
//          | Decoder Layer N   |
//          +-------------------+
//                   |
//                   v
//                 Output
//
// Each decoder layer contains:
//
//   Causal Multi-Head Self-Attention
//              +
//          Residual
//              |
//          LayerNorm
//              |
//          FeedForward
//              +
//          Residual
//              |
//          LayerNorm
//
// Supported input:
//
//   [seq, embed_dim]
//   [batch, seq, embed_dim]
//
// CPU / Float32.
//
// ============================================================

class TransformerDecoder {
public:

    TransformerDecoder(
        std::size_t embed_dim,
        std::size_t num_heads,
        std::size_t hidden_dim,
        std::size_t num_layers,
        bool use_bias = true
    );

    // --------------------------------------------------------
    // Forward
    // --------------------------------------------------------

    Tensor forward(
        const Tensor& input
    ) const;

    // --------------------------------------------------------
    // Layer access
    // --------------------------------------------------------

    const TransformerDecoderLayer& layer(
        std::size_t index
    ) const;

    TransformerDecoderLayer& layer(
        std::size_t index
    );

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    //
    // Mengembalikan seluruh parameter trainable dari semua
    // decoder layer secara berurutan.
    // --------------------------------------------------------

    std::vector<Tensor*> parameters();

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t embed_dim() const;

    std::size_t num_heads() const;

    std::size_t hidden_dim() const;

    std::size_t num_layers() const;

    bool has_bias() const;

private:

    std::size_t embed_dim_;

    std::size_t num_heads_;

    std::size_t hidden_dim_;

    std::size_t num_layers_;

    bool use_bias_;

    std::vector<TransformerDecoderLayer> layers_;
};

} // namespace venla
