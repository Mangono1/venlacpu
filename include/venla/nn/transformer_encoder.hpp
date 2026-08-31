#pragma once

#include "venla/nn/transformer_encoder_layer.hpp"
#include "venla/tensor/tensor.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace venla {

// ============================================================
// TRANSFORMER ENCODER
//
// Stack of N TransformerEncoderLayer instances.
//
// Architecture:
//
//   input
//      |
//      v
//   EncoderLayer #0
//      |
//      v
//   EncoderLayer #1
//      |
//      v
//      ...
//      |
//      v
//   EncoderLayer #(num_layers - 1)
//      |
//      v
//   output
//
// Each layer receives the output of the previous layer.
//
// Supported input:
//   [seq, embed_dim]
//   [batch, seq, embed_dim]
//
// CPU / Float32.
//
// Example:
//
//   TransformerEncoder encoder(
//       8,      // embed_dim
//       2,      // num_heads
//       32,     // hidden_dim
//       4       // num_layers
//   );
//
//   Tensor output = encoder.forward(input);
//
// ============================================================

class TransformerEncoder {
public:

    // --------------------------------------------------------
    // Construction
    // --------------------------------------------------------

    TransformerEncoder(
        std::size_t embed_dim,
        std::size_t num_heads,
        std::size_t hidden_dim,
        std::size_t num_layers,
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
    // Layer access
    // --------------------------------------------------------

    const TransformerEncoderLayer& layer(
        std::size_t index
    ) const;

    TransformerEncoderLayer& layer(
        std::size_t index
    );

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t embed_dim() const;

    std::size_t num_heads() const;

    std::size_t hidden_dim() const;

    std::size_t num_layers() const;

    bool has_bias() const;

    bool is_causal() const;

    // --------------------------------------------------------
    // Causal mode
    //
    // Updates every encoder layer.
    // --------------------------------------------------------

    void set_causal(
        bool enabled
    );

private:

    std::size_t embed_dim_;

    std::size_t num_heads_;

    std::size_t hidden_dim_;

    std::size_t num_layers_;

    bool use_bias_;

    bool causal_;

    std::vector<
        std::unique_ptr<
            TransformerEncoderLayer
        >
    > layers_;
};

} // namespace venla
