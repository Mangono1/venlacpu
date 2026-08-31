#include "venla/nn/transformer_decoder_layer.hpp"

#include "venla/math/operations.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace venla {

// ============================================================
// CONSTRUCTOR
// ============================================================

TransformerDecoderLayer::TransformerDecoderLayer(
    std::size_t embed_dim,
    std::size_t num_heads,
    std::size_t hidden_dim,
    bool use_bias
)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      hidden_dim_(hidden_dim),
      use_bias_(use_bias),
      self_attention_(
          embed_dim,
          num_heads,
          use_bias,
          true
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
            "TransformerDecoderLayer: "
            "embed_dim must be greater than zero"
        );
    }

    if (num_heads == 0) {

        throw std::invalid_argument(
            "TransformerDecoderLayer: "
            "num_heads must be greater than zero"
        );
    }

    if (hidden_dim == 0) {

        throw std::invalid_argument(
            "TransformerDecoderLayer: "
            "hidden_dim must be greater than zero"
        );
    }

    if (embed_dim % num_heads != 0) {

        throw std::invalid_argument(
            "TransformerDecoderLayer: "
            "embed_dim must be divisible by num_heads"
        );
    }
}

// ============================================================
// VALIDATION
// ============================================================

namespace {

void validate_decoder_input(
    const Tensor& input,
    std::size_t embed_dim,
    const char* operation
) {
    if (input.ndim() != 2 &&
        input.ndim() != 3) {

        throw std::runtime_error(
            std::string(operation) +
            ": input must be 2D or 3D"
        );
    }

    if (input.dtype() != DType::Float32) {

        throw std::runtime_error(
            std::string(operation) +
            ": only Float32 is currently supported"
        );
    }

    if (!input.device().is_cpu()) {

        throw std::runtime_error(
            std::string(operation) +
            ": only CPU device is currently supported"
        );
    }

    const std::size_t features =
        input.shape()[
            input.ndim() - 1
        ];

    if (features != embed_dim) {

        std::ostringstream message;

        message
            << operation
            << ": input last dimension must be "
            << embed_dim
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
            std::string(operation) +
            ": sequence length must be greater than zero"
        );
    }
}

} // namespace

// ============================================================
// NORMAL FORWARD
// ============================================================

Tensor TransformerDecoderLayer::forward(
    const Tensor& input
) const {

    validate_decoder_input(
        input,
        embed_dim_,
        "TransformerDecoderLayer::forward"
    );

    // --------------------------------------------------------
    // Causal self attention
    // --------------------------------------------------------

    Tensor attention =
        self_attention_.forward(
            input
        );

    // --------------------------------------------------------
    // First residual
    // --------------------------------------------------------

    Tensor residual1 =
        add(
            input,
            attention
        );

    // --------------------------------------------------------
    // First normalization
    // --------------------------------------------------------

    Tensor normalized1 =
        norm1_.forward(
            residual1
        );

    // --------------------------------------------------------
    // Feed forward
    // --------------------------------------------------------

    Tensor feed_forward =
        feed_forward_.forward(
            normalized1
        );

    // --------------------------------------------------------
    // Second residual
    // --------------------------------------------------------

    Tensor residual2 =
        add(
            normalized1,
            feed_forward
        );

    // --------------------------------------------------------
    // Second normalization
    // --------------------------------------------------------

    Tensor output =
        norm2_.forward(
            residual2
        );

    return output;
}

// ============================================================
// CACHED FORWARD
//
// Inference path:
//
//   new tokens
//       |
//       v
//   Cached Self Attention
//       |
//       v
//   residual
//       |
//       v
//   LayerNorm
//       |
//       v
//   FeedForward
//       |
//       v
//   residual
//       |
//       v
//   LayerNorm
//
// Only the attention K/V state is cached.
//
// FeedForward and LayerNorm operate only on the newly
// generated token(s).
// ============================================================

Tensor TransformerDecoderLayer::forward_cached(
    const Tensor& input,
    KVCache& cache
) const {

    validate_decoder_input(
        input,
        embed_dim_,
        "TransformerDecoderLayer::forward_cached"
    );

    // --------------------------------------------------------
    // Cached causal self attention
    //
    // cache belongs to THIS decoder layer.
    // --------------------------------------------------------

    Tensor attention =
        self_attention_.forward_cached(
            input,
            cache
        );

    // --------------------------------------------------------
    // First residual
    // --------------------------------------------------------

    Tensor residual1 =
        add(
            input,
            attention
        );

    // --------------------------------------------------------
    // First normalization
    // --------------------------------------------------------

    Tensor normalized1 =
        norm1_.forward(
            residual1
        );

    // --------------------------------------------------------
    // Feed Forward
    //
    // Only new tokens are processed.
    // --------------------------------------------------------

    Tensor feed_forward =
        feed_forward_.forward(
            normalized1
        );

    // --------------------------------------------------------
    // Second residual
    // --------------------------------------------------------

    Tensor residual2 =
        add(
            normalized1,
            feed_forward
        );

    // --------------------------------------------------------
    // Second normalization
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
TransformerDecoderLayer::self_attention() const {
    return self_attention_;
}

MultiHeadAttention&
TransformerDecoderLayer::self_attention() {
    return self_attention_;
}

// ============================================================
// FEED FORWARD
// ============================================================

const FeedForward&
TransformerDecoderLayer::feed_forward() const {
    return feed_forward_;
}

FeedForward&
TransformerDecoderLayer::feed_forward() {
    return feed_forward_;
}

// ============================================================
// NORM 1
// ============================================================

const LayerNorm&
TransformerDecoderLayer::norm1() const {
    return norm1_;
}

LayerNorm&
TransformerDecoderLayer::norm1() {
    return norm1_;
}

// ============================================================
// NORM 2
// ============================================================

const LayerNorm&
TransformerDecoderLayer::norm2() const {
    return norm2_;
}

LayerNorm&
TransformerDecoderLayer::norm2() {
    return norm2_;
}

// ============================================================
// METADATA
// ============================================================

std::size_t
TransformerDecoderLayer::embed_dim() const {
    return embed_dim_;
}

std::size_t
TransformerDecoderLayer::num_heads() const {
    return num_heads_;
}

std::size_t
TransformerDecoderLayer::hidden_dim() const {
    return hidden_dim_;
}

bool
TransformerDecoderLayer::has_bias() const {
    return use_bias_;
}

bool
TransformerDecoderLayer::is_causal() const {
    return self_attention_.is_causal();
}

} // namespace venla
