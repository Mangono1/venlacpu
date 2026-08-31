#include "venla/nn/language_model.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace venla {

// ============================================================
// CONSTRUCTOR
// ============================================================

LanguageModel::LanguageModel(
    std::size_t vocab_size,
    std::size_t max_seq_len,
    std::size_t embed_dim,
    std::size_t num_heads,
    std::size_t hidden_dim,
    std::size_t num_layers,
    bool use_bias
)
    : vocab_size_(vocab_size),
      max_seq_len_(max_seq_len),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      hidden_dim_(hidden_dim),
      num_layers_(num_layers),
      use_bias_(use_bias),
      embedding_(
          vocab_size,
          embed_dim
      ),
      positional_encoding_(
          max_seq_len,
          embed_dim
      ),
      decoder_(
          embed_dim,
          num_heads,
          hidden_dim,
          num_layers,
          use_bias
      ),
      lm_head_(
          embed_dim,
          vocab_size,
          use_bias
      ) {

    if (vocab_size == 0) {
        throw std::invalid_argument(
            "LanguageModel: "
            "vocab_size must be greater than zero"
        );
    }

    if (max_seq_len == 0) {
        throw std::invalid_argument(
            "LanguageModel: "
            "max_seq_len must be greater than zero"
        );
    }

    if (embed_dim == 0) {
        throw std::invalid_argument(
            "LanguageModel: "
            "embed_dim must be greater than zero"
        );
    }

    if (num_heads == 0) {
        throw std::invalid_argument(
            "LanguageModel: "
            "num_heads must be greater than zero"
        );
    }

    if (hidden_dim == 0) {
        throw std::invalid_argument(
            "LanguageModel: "
            "hidden_dim must be greater than zero"
        );
    }

    if (num_layers == 0) {
        throw std::invalid_argument(
            "LanguageModel: "
            "num_layers must be greater than zero"
        );
    }

    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument(
            "LanguageModel: "
            "embed_dim must be divisible by num_heads"
        );
    }
}

// ============================================================
// FORWARD
//
// Pipeline:
//
//     token IDs
//        |
//        v
//     Embedding
//        |
//        v
// Positional Encoding
//        |
//        v
// Transformer Decoder
//        |
//        v
//      LM Head
//        |
//        v
//      Logits
//
// ============================================================

Tensor LanguageModel::forward(
    const Tensor& input
) const {

    // --------------------------------------------------------
    // Validate input rank.
    // --------------------------------------------------------

    if (input.ndim() != 1 &&
        input.ndim() != 2) {

        throw std::runtime_error(
            "LanguageModel::forward: "
            "input must be 1D [seq] or "
            "2D [batch, seq]"
        );
    }

    // --------------------------------------------------------
    // Token IDs.
    // --------------------------------------------------------

    if (
        input.dtype() != DType::Int32 &&
        input.dtype() != DType::Int64
    ) {

        throw std::runtime_error(
            "LanguageModel::forward: "
            "input must use Int32 or Int64 token IDs"
        );
    }

    if (!input.device().is_cpu()) {

        throw std::runtime_error(
            "LanguageModel::forward: "
            "only CPU device is currently supported"
        );
    }

    // --------------------------------------------------------
    // Sequence length.
    //
    // [seq]
    //       -> shape[0]
    //
    // [batch, seq]
    //       -> shape[1]
    // --------------------------------------------------------

    const std::size_t sequence_length =
        input.shape()[
            input.ndim() - 1
        ];

    if (sequence_length == 0) {

        throw std::runtime_error(
            "LanguageModel::forward: "
            "sequence length must be greater than zero"
        );
    }

    if (sequence_length > max_seq_len_) {

        std::ostringstream message;

        message
            << "LanguageModel::forward: "
            << "sequence length "
            << sequence_length
            << " exceeds max_seq_len "
            << max_seq_len_;

        throw std::out_of_range(
            message.str()
        );
    }

    // --------------------------------------------------------
    // 1. Token embedding
    //
    // [seq]
    //      ->
    // [seq, embed_dim]
    //
    // [batch, seq]
    //      ->
    // [batch, seq, embed_dim]
    // --------------------------------------------------------

    Tensor embedded =
        embedding_.forward(
            input
        );

    // --------------------------------------------------------
    // 2. Positional encoding
    //
    // Adds:
    //
    //     PE[position]
    //
    // to every token representation.
    // --------------------------------------------------------

    Tensor positioned =
        positional_encoding_.forward(
            embedded
        );

    // --------------------------------------------------------
    // 3. Transformer decoder
    //
    // Causal self-attention ensures token position t
    // cannot attend to future positions > t.
    // --------------------------------------------------------

    Tensor decoded =
        decoder_.forward(
            positioned
        );

    // --------------------------------------------------------
    // 4. LM Head
    //
    // [seq, embed_dim]
    //      ->
    // [seq, vocab_size]
    //
    // [batch, seq, embed_dim]
    //      ->
    // [batch, seq, vocab_size]
    // --------------------------------------------------------

    Tensor logits =
        lm_head_.forward(
            decoded
        );

    return logits;
}

// ============================================================
// EMBEDDING
// ============================================================

const Embedding&
LanguageModel::embedding() const {
    return embedding_;
}

Embedding&
LanguageModel::embedding() {
    return embedding_;
}

// ============================================================
// POSITIONAL ENCODING
// ============================================================

const PositionalEncoding&
LanguageModel::positional_encoding() const {
    return positional_encoding_;
}

PositionalEncoding&
LanguageModel::positional_encoding() {
    return positional_encoding_;
}

// ============================================================
// DECODER
// ============================================================

const TransformerDecoder&
LanguageModel::decoder() const {
    return decoder_;
}

TransformerDecoder&
LanguageModel::decoder() {
    return decoder_;
}

// ============================================================
// LM HEAD
// ============================================================

const Linear&
LanguageModel::lm_head() const {
    return lm_head_;
}

Linear&
LanguageModel::lm_head() {
    return lm_head_;
}

// ============================================================
// PARAMETERS
// ============================================================

std::vector<Tensor*>
LanguageModel::parameters() {

    std::vector<Tensor*> result;

    // --------------------------------------------------------
    // Embedding
    // --------------------------------------------------------

    result.push_back(
        &embedding_.weight()
    );

    // --------------------------------------------------------
    // Transformer Decoder
    // --------------------------------------------------------

    std::vector<Tensor*> decoder_parameters =
        decoder_.parameters();

    result.insert(
        result.end(),
        decoder_parameters.begin(),
        decoder_parameters.end()
    );

    // --------------------------------------------------------
    // LM Head
    // --------------------------------------------------------

    result.push_back(
        &lm_head_.weight()
    );

    if (use_bias_) {

        result.push_back(
            &lm_head_.bias()
        );
    }

    return result;
}

// ============================================================
// METADATA
// ============================================================

std::size_t
LanguageModel::vocab_size() const {
    return vocab_size_;
}

std::size_t
LanguageModel::max_seq_len() const {
    return max_seq_len_;
}

std::size_t
LanguageModel::embed_dim() const {
    return embed_dim_;
}

std::size_t
LanguageModel::num_heads() const {
    return num_heads_;
}

std::size_t
LanguageModel::hidden_dim() const {
    return hidden_dim_;
}

std::size_t
LanguageModel::num_layers() const {
    return num_layers_;
}

bool
LanguageModel::has_bias() const {
    return use_bias_;
}

} // namespace venla
