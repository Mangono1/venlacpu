#pragma once

#include "venla/nn/embedding.hpp"
#include "venla/nn/linear.hpp"
#include "venla/nn/positional_encoding.hpp"
#include "venla/nn/transformer_decoder.hpp"
#include "venla/tensor/tensor.hpp"

#include <cstddef>
#include <vector>

namespace venla {

// ============================================================
// VENLA CAUSAL LANGUAGE MODEL
//
// Pipeline:
//
//     token IDs
//         |
//         v
//      Embedding
//         |
//         v
//   Positional Encoding
//         |
//         v
//   Transformer Decoder
//         |
//         v
//      LM Head
//         |
//         v
//       Logits
//
// Input:
//
//     [seq]
//     [batch, seq]
//
// Output:
//
//     [seq, vocab_size]
//     [batch, seq, vocab_size]
//
// CPU / Float32.
//
// ============================================================

class LanguageModel {
public:

    LanguageModel(
        std::size_t vocab_size,
        std::size_t max_seq_len,
        std::size_t embed_dim,
        std::size_t num_heads,
        std::size_t hidden_dim,
        std::size_t num_layers,
        bool use_bias = true
    );

    // --------------------------------------------------------
    // Forward
    //
    // Input token IDs:
    //
    //     [seq]
    //     [batch, seq]
    //
    // Output logits:
    //
    //     [seq, vocab_size]
    //     [batch, seq, vocab_size]
    // --------------------------------------------------------

    Tensor forward(
        const Tensor& input
    ) const;

    // --------------------------------------------------------
    // Embedding
    // --------------------------------------------------------

    const Embedding& embedding() const;

    Embedding& embedding();

    // --------------------------------------------------------
    // Positional Encoding
    // --------------------------------------------------------

    const PositionalEncoding& positional_encoding() const;

    PositionalEncoding& positional_encoding();

    // --------------------------------------------------------
    // Transformer Decoder
    // --------------------------------------------------------

    const TransformerDecoder& decoder() const;

    TransformerDecoder& decoder();

    // --------------------------------------------------------
    // Language Model Head
    //
    // embed_dim -> vocab_size
    // --------------------------------------------------------

    const Linear& lm_head() const;

    Linear& lm_head();

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    //
    // Mengembalikan seluruh parameter trainable LanguageModel:
    //
    //   Embedding
    //   Transformer Decoder
    //   LM Head
    //
    // Positional encoding tidak termasuk karena bukan parameter
    // trainable.
    // --------------------------------------------------------

    std::vector<Tensor*> parameters();

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t vocab_size() const;

    std::size_t max_seq_len() const;

    std::size_t embed_dim() const;

    std::size_t num_heads() const;

    std::size_t hidden_dim() const;

    std::size_t num_layers() const;

    bool has_bias() const;

private:

    std::size_t vocab_size_;

    std::size_t max_seq_len_;

    std::size_t embed_dim_;

    std::size_t num_heads_;

    std::size_t hidden_dim_;

    std::size_t num_layers_;

    bool use_bias_;

    Embedding embedding_;

    PositionalEncoding positional_encoding_;

    TransformerDecoder decoder_;

    Linear lm_head_;
};

} // namespace venla
