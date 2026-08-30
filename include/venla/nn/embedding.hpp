#pragma once

#include "venla/tensor/tensor.hpp"

#include <cstddef>

namespace venla {

// ============================================================
// EMBEDDING
//
// Mengubah token ID menjadi vector embedding.
//
// Weight:
//     [vocab_size, embedding_dim]
//
// Input:
//     Int32 / Int64
//
// Supported input:
//     [seq]
//     [batch, seq]
//     [d1, d2, ..., dn]
//
// Output:
//     [seq, embedding_dim]
//     [batch, seq, embedding_dim]
//     [d1, d2, ..., dn, embedding_dim]
//
// Contoh:
//
//     vocab_size   = 10000
//     embedding_dim = 128
//
//     input  = [2, 15, 42]
//
//     output = [
//         embedding[2],
//         embedding[15],
//         embedding[42]
//     ]
//
// Autograd:
//
//     Gradient hanya dikumpulkan pada row embedding yang
//     digunakan oleh input.
//
// Jika token muncul beberapa kali:
//
//     [5, 5, 7]
//
// maka gradient embedding[5] dijumlahkan dua kali.
//
// ============================================================

class Embedding {
public:

    Embedding(
        std::size_t vocab_size,
        std::size_t embedding_dim
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

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t vocab_size() const;

    std::size_t embedding_dim() const;

private:

    std::size_t vocab_size_;

    std::size_t embedding_dim_;

    Tensor weight_;
};

} // namespace venla
