#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>

namespace venla {

// ============================================================
// SINUSOIDAL POSITIONAL ENCODING
//
// Positional encoding memberikan informasi posisi token kepada
// model Transformer.
//
// Formula:
//
//   PE(pos, 2i)
//       = sin(pos / 10000^(2i / embedding_dim))
//
//   PE(pos, 2i + 1)
//       = cos(pos / 10000^(2i / embedding_dim))
//
// Input:
//
//   [seq, embedding_dim]
//
//   [batch, seq, embedding_dim]
//
//   [d1, d2, ..., seq, embedding_dim]
//
// Output:
//
//   sama dengan shape input.
//
// Encoding ditambahkan ke input:
//
//   output = input + positional_encoding
//
// Positional encoding bukan parameter trainable.
// ============================================================

class PositionalEncoding {
public:

    PositionalEncoding(
        std::size_t max_seq_len,
        std::size_t embedding_dim
    );

    // --------------------------------------------------------
    // Forward
    // --------------------------------------------------------

    Tensor forward(
        const Tensor& input
    ) const;

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t max_seq_len() const;

    std::size_t embedding_dim() const;

    // --------------------------------------------------------
    // Encoding table
    //
    // Shape:
    //
    //   [max_seq_len, embedding_dim]
    // --------------------------------------------------------

    const Tensor& encoding() const;

private:

    std::size_t max_seq_len_;

    std::size_t embedding_dim_;

    Tensor encoding_;
};

} // namespace venla

