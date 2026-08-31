#include "venla/nn/transformer_encoder.hpp"

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace venla {

// ============================================================
// CONSTRUCTOR
// ============================================================

TransformerEncoder::TransformerEncoder(
    std::size_t embed_dim,
    std::size_t num_heads,
    std::size_t hidden_dim,
    std::size_t num_layers,
    bool use_bias,
    bool causal
)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      hidden_dim_(hidden_dim),
      num_layers_(num_layers),
      use_bias_(use_bias),
      causal_(causal),
      layers_() {

    // --------------------------------------------------------
    // Validation
    // --------------------------------------------------------

    if (embed_dim == 0) {

        throw std::invalid_argument(
            "TransformerEncoder: "
            "embed_dim must be greater than zero"
        );
    }

    if (num_heads == 0) {

        throw std::invalid_argument(
            "TransformerEncoder: "
            "num_heads must be greater than zero"
        );
    }

    if (hidden_dim == 0) {

        throw std::invalid_argument(
            "TransformerEncoder: "
            "hidden_dim must be greater than zero"
        );
    }

    if (num_layers == 0) {

        throw std::invalid_argument(
            "TransformerEncoder: "
            "num_layers must be greater than zero"
        );
    }

    if (embed_dim % num_heads != 0) {

        throw std::invalid_argument(
            "TransformerEncoder: "
            "embed_dim must be divisible by num_heads"
        );
    }

    // --------------------------------------------------------
    // Allocate layer stack
    // --------------------------------------------------------

    layers_.reserve(
        num_layers_
    );

    for (std::size_t i = 0;
         i < num_layers_;
         ++i) {

        layers_.push_back(
            std::make_unique<
                TransformerEncoderLayer
            >(
                embed_dim_,
                num_heads_,
                hidden_dim_,
                use_bias_,
                causal_
            )
        );
    }
}

// ============================================================
// FORWARD
//
// Sequentially applies every TransformerEncoderLayer.
//
//   current = input
//
//   current = layer[0](current)
//   current = layer[1](current)
//   current = layer[2](current)
//   ...
//
// The returned Tensor keeps the autograd chain:
//
//   input
//      |
//   layer 0
//      |
//   layer 1
//      |
//     ...
//      |
//   layer N-1
//
// ============================================================

Tensor TransformerEncoder::forward(
    const Tensor& input
) const {

    // --------------------------------------------------------
    // Validation
    // --------------------------------------------------------

    if (input.ndim() != 2 &&
        input.ndim() != 3) {

        throw std::runtime_error(
            "TransformerEncoder::forward: "
            "input must be 2D or 3D"
        );
    }

    if (input.dtype() != DType::Float32) {

        throw std::runtime_error(
            "TransformerEncoder::forward: "
            "only Float32 is currently supported"
        );
    }

    if (!input.device().is_cpu()) {

        throw std::runtime_error(
            "TransformerEncoder::forward: "
            "only CPU device is currently supported"
        );
    }

    if (layers_.empty()) {

        throw std::runtime_error(
            "TransformerEncoder::forward: "
            "encoder contains no layers"
        );
    }

    const std::size_t features =
        input.shape()[
            input.ndim() - 1
        ];

    if (features != embed_dim_) {

        std::ostringstream message;

        message
            << "TransformerEncoder::forward: "
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
            "TransformerEncoder::forward: "
            "sequence length must be greater than zero"
        );
    }

    // --------------------------------------------------------
    // Stack execution
    // --------------------------------------------------------

    Tensor current =
        input;

    for (const auto& encoder_layer : layers_) {

        if (!encoder_layer) {

            throw std::runtime_error(
                "TransformerEncoder::forward: "
                "encoder layer is not initialized"
            );
        }

        current =
            encoder_layer->forward(
                current
            );
    }

    return current;
}

// ============================================================
// LAYER CONST ACCESS
// ============================================================

const TransformerEncoderLayer&
TransformerEncoder::layer(
    std::size_t index
) const {

    if (index >= layers_.size()) {

        throw std::out_of_range(
            "TransformerEncoder::layer: "
            "layer index out of range"
        );
    }

    if (!layers_[index]) {

        throw std::runtime_error(
            "TransformerEncoder::layer: "
            "selected layer is not initialized"
        );
    }

    return *layers_[index];
}

// ============================================================
// LAYER MUTABLE ACCESS
// ============================================================

TransformerEncoderLayer&
TransformerEncoder::layer(
    std::size_t index
) {

    if (index >= layers_.size()) {

        throw std::out_of_range(
            "TransformerEncoder::layer: "
            "layer index out of range"
        );
    }

    if (!layers_[index]) {

        throw std::runtime_error(
            "TransformerEncoder::layer: "
            "selected layer is not initialized"
        );
    }

    return *layers_[index];
}

// ============================================================
// METADATA
// ============================================================

std::size_t
TransformerEncoder::embed_dim() const {
    return embed_dim_;
}

std::size_t
TransformerEncoder::num_heads() const {
    return num_heads_;
}

std::size_t
TransformerEncoder::hidden_dim() const {
    return hidden_dim_;
}

std::size_t
TransformerEncoder::num_layers() const {
    return num_layers_;
}

bool
TransformerEncoder::has_bias() const {
    return use_bias_;
}

bool
TransformerEncoder::is_causal() const {
    return causal_;
}

// ============================================================
// SET CAUSAL
//
// Propagates the setting to every encoder layer.
// ============================================================

void
TransformerEncoder::set_causal(
    bool enabled
) {

    causal_ =
        enabled;

    for (auto& encoder_layer : layers_) {

        if (!encoder_layer) {

            throw std::runtime_error(
                "TransformerEncoder::set_causal: "
                "encoder layer is not initialized"
            );
        }

        encoder_layer->set_causal(
            enabled
        );
    }
}

} // namespace venla
