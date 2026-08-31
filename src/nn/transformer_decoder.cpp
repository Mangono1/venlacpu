#include "venla/nn/transformer_decoder.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace venla {

// ============================================================
// CONSTRUCTOR
// ============================================================

TransformerDecoder::TransformerDecoder(
    std::size_t embed_dim,
    std::size_t num_heads,
    std::size_t hidden_dim,
    std::size_t num_layers,
    bool use_bias
)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      hidden_dim_(hidden_dim),
      num_layers_(num_layers),
      use_bias_(use_bias),
      layers_() {

    if (embed_dim == 0) {

        throw std::invalid_argument(
            "TransformerDecoder: "
            "embed_dim must be greater than zero"
        );
    }

    if (num_heads == 0) {

        throw std::invalid_argument(
            "TransformerDecoder: "
            "num_heads must be greater than zero"
        );
    }

    if (hidden_dim == 0) {

        throw std::invalid_argument(
            "TransformerDecoder: "
            "hidden_dim must be greater than zero"
        );
    }

    if (num_layers == 0) {

        throw std::invalid_argument(
            "TransformerDecoder: "
            "num_layers must be greater than zero"
        );
    }

    if (embed_dim % num_heads != 0) {

        throw std::invalid_argument(
            "TransformerDecoder: "
            "embed_dim must be divisible by num_heads"
        );
    }

    layers_.reserve(
        num_layers_
    );

    for (std::size_t i = 0;
         i < num_layers_;
         ++i) {

        layers_.emplace_back(
            embed_dim_,
            num_heads_,
            hidden_dim_,
            use_bias_
        );
    }
}

// ============================================================
// FORWARD
//
// Pass input through every decoder layer.
//
//   output = layer_N(
//                layer_N-1(
//                  ...
//                    layer_1(input)
//                  ...
//                )
//            )
//
// Each layer preserves:
//
//   [seq, embed_dim]
//
// or:
//
//   [batch, seq, embed_dim]
//
// ============================================================

Tensor TransformerDecoder::forward(
    const Tensor& input
) const {

    if (input.ndim() != 2 &&
        input.ndim() != 3) {

        throw std::runtime_error(
            "TransformerDecoder::forward: "
            "input must be 2D or 3D"
        );
    }

    if (input.dtype() != DType::Float32) {

        throw std::runtime_error(
            "TransformerDecoder::forward: "
            "only Float32 is currently supported"
        );
    }

    if (!input.device().is_cpu()) {

        throw std::runtime_error(
            "TransformerDecoder::forward: "
            "only CPU device is currently supported"
        );
    }

    const std::size_t features =
        input.shape()[
            input.ndim() - 1
        ];

    if (features != embed_dim_) {

        std::ostringstream message;

        message
            << "TransformerDecoder::forward: "
            << "input last dimension must be "
            << embed_dim_
            << ", got "
            << features
            << " for input shape "
            << input.shape().to_string();

        throw std::runtime_error(
            message.str()
        );
    }

    const std::size_t sequence_length =
        input.shape()[
            input.ndim() - 2
        ];

    if (sequence_length == 0) {

        throw std::runtime_error(
            "TransformerDecoder::forward: "
            "sequence length must be greater than zero"
        );
    }

    // --------------------------------------------------------
    // Sequential decoder stack
    // --------------------------------------------------------

    Tensor output =
        input;

    for (const TransformerDecoderLayer& decoder_layer :
         layers_) {

        output =
            decoder_layer.forward(
                output
            );
    }

    return output;
}

// ============================================================
// LAYER ACCESS
// ============================================================

const TransformerDecoderLayer&
TransformerDecoder::layer(
    std::size_t index
) const {

    if (index >= layers_.size()) {

        std::ostringstream message;

        message
            << "TransformerDecoder::layer: "
            << "index "
            << index
            << " out of range for "
            << layers_.size()
            << " layers";

        throw std::out_of_range(
            message.str()
        );
    }

    return layers_[index];
}

TransformerDecoderLayer&
TransformerDecoder::layer(
    std::size_t index
) {

    if (index >= layers_.size()) {

        std::ostringstream message;

        message
            << "TransformerDecoder::layer: "
            << "index "
            << index
            << " out of range for "
            << layers_.size()
            << " layers";

        throw std::out_of_range(
            message.str()
        );
    }

    return layers_[index];
}

// ============================================================
// METADATA
// ============================================================

std::size_t
TransformerDecoder::embed_dim() const {
    return embed_dim_;
}

std::size_t
TransformerDecoder::num_heads() const {
    return num_heads_;
}

std::size_t
TransformerDecoder::hidden_dim() const {
    return hidden_dim_;
}

std::size_t
TransformerDecoder::num_layers() const {
    return num_layers_;
}

bool
TransformerDecoder::has_bias() const {
    return use_bias_;
}

} // namespace venla
