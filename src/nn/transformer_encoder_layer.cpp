#include "venla/nn/transformer_encoder_layer.hpp"

#include "venla/math/operations.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace venla {

// ============================================================
// CONSTRUCTOR
// ============================================================

TransformerEncoderLayer::TransformerEncoderLayer(
    std::size_t embed_dim,
    std::size_t num_heads,
    std::size_t hidden_dim,
    bool use_bias,
    bool causal
)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      hidden_dim_(hidden_dim),
      use_bias_(use_bias),
      self_attention_(
          embed_dim,
          num_heads,
          use_bias,
          causal
      ),
      feed_forward_(
          embed_dim,
          hidden_dim,
          use_bias
      ),
      norm1_(
          embed_dim
      ),
      norm2_(
          embed_dim
      ) {

    if (embed_dim == 0) {
        throw std::invalid_argument(
            "TransformerEncoderLayer: "
            "embed_dim must be greater than zero"
        );
    }

    if (num_heads == 0) {
        throw std::invalid_argument(
            "TransformerEncoderLayer: "
            "num_heads must be greater than zero"
        );
    }

    if (hidden_dim == 0) {
        throw std::invalid_argument(
            "TransformerEncoderLayer: "
            "hidden_dim must be greater than zero"
        );
    }

    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument(
            "TransformerEncoderLayer: "
            "embed_dim must be divisible by num_heads"
        );
    }
}

// ============================================================
// FORWARD
//
// Post-Norm Transformer:
//
//   attention = MHA(x)
//
//   residual1 = x + attention
//
//   normalized1 = LN(residual1)
//
//   feed_forward = FFN(normalized1)
//
//   residual2 = normalized1 + feed_forward
//
//   output = LN(residual2)
//
// ============================================================

Tensor TransformerEncoderLayer::forward(
    const Tensor& input
) const {

    if (input.ndim() != 2 &&
        input.ndim() != 3) {

        throw std::runtime_error(
            "TransformerEncoderLayer::forward: "
            "input must be 2D or 3D"
        );
    }

    if (input.dtype() != DType::Float32) {

        throw std::runtime_error(
            "TransformerEncoderLayer::forward: "
            "only Float32 is currently supported"
        );
    }

    if (!input.device().is_cpu()) {

        throw std::runtime_error(
            "TransformerEncoderLayer::forward: "
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
            << "TransformerEncoderLayer::forward: "
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
            "TransformerEncoderLayer::forward: "
            "sequence length must be greater than zero"
        );
    }

    // --------------------------------------------------------
    // SELF ATTENTION
    // --------------------------------------------------------

    Tensor attention =
        self_attention_.forward(
            input
        );

    // --------------------------------------------------------
    // FIRST RESIDUAL
    //
    // x + MHA(x)
    //
    // Uses existing autograd-aware add().
    // --------------------------------------------------------

    Tensor residual1 =
        add(
            input,
            attention
        );

    // --------------------------------------------------------
    // FIRST NORMALIZATION
    // --------------------------------------------------------

    Tensor normalized1 =
        norm1_.forward(
            residual1
        );

    // --------------------------------------------------------
    // FEED FORWARD
    // --------------------------------------------------------

    Tensor feed_forward =
        feed_forward_.forward(
            normalized1
        );

    // --------------------------------------------------------
    // SECOND RESIDUAL
    //
    // normalized1 + FFN(normalized1)
    // --------------------------------------------------------

    Tensor residual2 =
        add(
            normalized1,
            feed_forward
        );

    // --------------------------------------------------------
    // SECOND NORMALIZATION
    // --------------------------------------------------------

    Tensor output =
        norm2_.forward(
            residual2
        );

    return output;
}

// ============================================================
// SELF ATTENTION
// ============================================================

const MultiHeadAttention&
TransformerEncoderLayer::self_attention() const {
    return self_attention_;
}

MultiHeadAttention&
TransformerEncoderLayer::self_attention() {
    return self_attention_;
}

// ============================================================
// FEED FORWARD
// ============================================================

const FeedForward&
TransformerEncoderLayer::feed_forward() const {
    return feed_forward_;
}

FeedForward&
TransformerEncoderLayer::feed_forward() {
    return feed_forward_;
}

// ============================================================
// NORM 1
// ============================================================

const LayerNorm&
TransformerEncoderLayer::norm1() const {
    return norm1_;
}

LayerNorm&
TransformerEncoderLayer::norm1() {
    return norm1_;
}

// ============================================================
// NORM 2
// ============================================================

const LayerNorm&
TransformerEncoderLayer::norm2() const {
    return norm2_;
}

LayerNorm&
TransformerEncoderLayer::norm2() {
    return norm2_;
}

// ============================================================
// METADATA
// ============================================================

std::size_t
TransformerEncoderLayer::embed_dim() const {
    return embed_dim_;
}

std::size_t
TransformerEncoderLayer::num_heads() const {
    return num_heads_;
}

std::size_t
TransformerEncoderLayer::hidden_dim() const {
    return hidden_dim_;
}

bool
TransformerEncoderLayer::has_bias() const {
    return use_bias_;
}

bool
TransformerEncoderLayer::is_causal() const {
    return self_attention_.is_causal();
}

void
TransformerEncoderLayer::set_causal(
    bool enabled
) {
    self_attention_.set_causal(
        enabled
    );
}

} // namespace venla
