#include "venla/nn/positional_encoding.hpp"

#include "venla/math/operations.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace venla {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// ============================================================
// BUILD SINUSOIDAL TABLE
// ============================================================

void initialize_encoding(
    Tensor& encoding,
    std::size_t max_seq_len,
    std::size_t embedding_dim
) {
    if (encoding.dtype() != DType::Float32) {
        throw std::runtime_error(
            "PositionalEncoding: encoding must be Float32"
        );
    }

    if (!encoding.device().is_cpu()) {
        throw std::runtime_error(
            "PositionalEncoding: "
            "only CPU device is currently supported"
        );
    }

    float* values =
        encoding.data_as<float>();

    for (std::size_t position = 0;
         position < max_seq_len;
         ++position) {

        for (std::size_t dimension = 0;
             dimension < embedding_dim;
             ++dimension) {

            // ------------------------------------------------
            // angle rate
            //
            // angle =
            //   position /
            //   10000^(2 * floor(dimension/2) / dimension)
            // ------------------------------------------------

            const std::size_t pair_index =
                dimension / 2;

            const double exponent =
                static_cast<double>(
                    2 * pair_index
                ) /
                static_cast<double>(
                    embedding_dim
                );

            const double denominator =
                std::pow(
                    10000.0,
                    exponent
                );

            const double angle =
                static_cast<double>(
                    position
                ) /
                denominator;

            if (dimension % 2 == 0) {
                values[
                    position * embedding_dim +
                    dimension
                ] =
                    static_cast<float>(
                        std::sin(angle)
                    );
            }
            else {
                values[
                    position * embedding_dim +
                    dimension
                ] =
                    static_cast<float>(
                        std::cos(angle)
                    );
            }
        }
    }
}

} // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

PositionalEncoding::PositionalEncoding(
    std::size_t max_seq_len,
    std::size_t embedding_dim
)
    : max_seq_len_(max_seq_len),
      embedding_dim_(embedding_dim),
      encoding_() {

    if (max_seq_len == 0) {
        throw std::invalid_argument(
            "PositionalEncoding: "
            "max_seq_len must be greater than zero"
        );
    }

    if (embedding_dim == 0) {
        throw std::invalid_argument(
            "PositionalEncoding: "
            "embedding_dim must be greater than zero"
        );
    }

    encoding_ =
        Tensor::empty(
            {
                max_seq_len_,
                embedding_dim_
            },
            DType::Float32,
            Device::cpu()
        );

    initialize_encoding(
        encoding_,
        max_seq_len_,
        embedding_dim_
    );
}

// ============================================================
// FORWARD
// ============================================================

Tensor PositionalEncoding::forward(
    const Tensor& input
) const {

    if (input.ndim() < 2) {
        throw std::runtime_error(
            "PositionalEncoding::forward: "
            "input must have at least 2 dimensions"
        );
    }

    if (input.dtype() != DType::Float32) {
        throw std::runtime_error(
            "PositionalEncoding::forward: "
            "input must be Float32"
        );
    }

    if (!input.device().is_cpu()) {
        throw std::runtime_error(
            "PositionalEncoding::forward: "
            "only CPU device is currently supported"
        );
    }

    if (!encoding_.device().is_cpu()) {
        throw std::runtime_error(
            "PositionalEncoding::forward: "
            "encoding must be on CPU"
        );
    }

    // --------------------------------------------------------
    // Input:
    //
    //   [..., seq, embedding_dim]
    //
    // Positional encoding:
    //
    //   [max_seq_len, embedding_dim]
    //
    // Ambil bagian:
    //
    //   [seq, embedding_dim]
    //
    // lalu gunakan broadcasting melalui add().
    // --------------------------------------------------------

    const std::size_t sequence_length =
        input.shape()[input.ndim() - 2];

    const std::size_t input_embedding_dim =
        input.shape()[input.ndim() - 1];

    if (input_embedding_dim !=
        embedding_dim_) {

        throw std::runtime_error(
            "PositionalEncoding::forward: "
            "input embedding dimension does not match "
            "configured embedding dimension"
        );
    }

    if (sequence_length > max_seq_len_) {
        throw std::out_of_range(
            "PositionalEncoding::forward: "
            "sequence length exceeds max_seq_len"
        );
    }

    // --------------------------------------------------------
    // Fast path:
    //
    // Karena Tensor::slice() belum digunakan untuk membuat
    // view khusus di sini, kita membuat tensor encoding yang
    // tepat berukuran [seq, embedding_dim].
    //
    // Ini juga menjaga implementasi tetap sederhana dan aman
    // untuk backend CPU saat ini.
    // --------------------------------------------------------

    Tensor position_values =
        Tensor::zeros(
            {
                sequence_length,
                embedding_dim_
            },
            DType::Float32,
            input.device()
        );

    const float* source =
        encoding_.data_as<float>();

    float* destination =
        position_values.data_as<float>();

    const std::size_t values_count =
        sequence_length *
        embedding_dim_;

    for (std::size_t i = 0;
         i < values_count;
         ++i) {

        destination[i] =
            source[i];
    }

    // --------------------------------------------------------
    // Broadcasting:
    //
    // input:
    //
    //   [batch, seq, dim]
    //
    // position:
    //
    //   [seq, dim]
    //
    // result:
    //
    //   [batch, seq, dim]
    //
    // Karena add() sudah memiliki autograd, gradient input
    // akan otomatis diteruskan.
    // --------------------------------------------------------

    return add(
        input,
        position_values
    );
}

// ============================================================
// METADATA
// ============================================================

std::size_t PositionalEncoding::max_seq_len() const {
    return max_seq_len_;
}

std::size_t PositionalEncoding::embedding_dim() const {
    return embedding_dim_;
}

// ============================================================
// ENCODING
// ============================================================

const Tensor& PositionalEncoding::encoding() const {
    return encoding_;
}

} // namespace venla

