#include "venla/nn/embedding.hpp"

#include "venla/autograd/autograd.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace venla {

namespace {

// ============================================================
// READ TOKEN INDEX
//
// Mendukung:
//
//     Int32
//     Int64
//
// ============================================================

std::size_t read_token_index(
    const Tensor& input,
    std::size_t index,
    std::size_t vocab_size
) {
    std::int64_t token = 0;

    if (input.dtype() == DType::Int32) {

        const std::int32_t* values =
            input.data_as<std::int32_t>();

        token =
            static_cast<std::int64_t>(
                values[index]
            );
    }
    else if (input.dtype() == DType::Int64) {

        const std::int64_t* values =
            input.data_as<std::int64_t>();

        token =
            values[index];
    }
    else {

        throw std::runtime_error(
            "Embedding: input dtype must be Int32 or Int64"
        );
    }

    if (token < 0) {

        throw std::out_of_range(
            "Embedding: token index cannot be negative"
        );
    }

    if (
        static_cast<std::uint64_t>(token) >=
        static_cast<std::uint64_t>(vocab_size)
    ) {

        throw std::out_of_range(
            "Embedding: token index is outside vocabulary"
        );
    }

    return static_cast<std::size_t>(token);
}

// ============================================================
// EMBEDDING AUTOGRAD NODE
// ============================================================

std::shared_ptr<AutogradNode>
make_embedding_node(
    const Tensor& input,
    const Tensor& weight,
    std::size_t vocab_size,
    std::size_t embedding_dim
) {
    if (!weight.requires_grad()) {
        return nullptr;
    }

    return std::make_shared<AutogradNode>(
        std::vector<Tensor>{weight},

        [
            input,
            weight,
            vocab_size,
            embedding_dim
        ](const Tensor& gradient) mutable {

            // ------------------------------------------------
            // Gradient output harus sama dengan hasil forward.
            //
            // input shape:
            //
            //     [d1, d2, ..., dn]
            //
            // output shape:
            //
            //     [d1, d2, ..., dn, embedding_dim]
            // ------------------------------------------------

            if (gradient.ndim() !=
                input.ndim() + 1) {

                throw std::runtime_error(
                    "Embedding backward: gradient rank mismatch"
                );
            }

            if (
                gradient.shape()
                [gradient.ndim() - 1]
                != embedding_dim
            ) {

                throw std::runtime_error(
                    "Embedding backward: "
                    "gradient embedding dimension mismatch"
                );
            }

            for (std::size_t i = 0;
                 i < input.ndim();
                 ++i) {

                if (
                    gradient.shape()[i]
                    != input.shape()[i]
                ) {

                    throw std::runtime_error(
                        "Embedding backward: "
                        "gradient shape mismatch"
                    );
                }
            }

            if (gradient.dtype() !=
                DType::Float32) {

                throw std::runtime_error(
                    "Embedding backward: "
                    "gradient must be Float32"
                );
            }

            // ------------------------------------------------
            // Gradient weight
            //
            // [vocab_size, embedding_dim]
            //
            // Semua row dimulai dari nol.
            // Hanya token yang digunakan yang mendapat gradient.
            // ------------------------------------------------

            Tensor grad_weight =
                Tensor::zeros(
                    {
                        vocab_size,
                        embedding_dim
                    },
                    DType::Float32,
                    weight.device()
                );

            const std::size_t token_count =
                input.numel();

            const float* grad_output =
                gradient.data_as<float>();

            float* grad_weight_data =
                grad_weight.data_as<float>();

            // ------------------------------------------------
            // Accumulate gradient per token.
            //
            // Karena output embedding tersusun:
            //
            // token0 -> embedding_dim values
            // token1 -> embedding_dim values
            // ...
            //
            // maka setiap token memiliki blok sepanjang
            // embedding_dim.
            // ------------------------------------------------

            for (std::size_t i = 0;
                 i < token_count;
                 ++i) {

                const std::size_t token =
                    read_token_index(
                        input,
                        i,
                        vocab_size
                    );

                const float* source =
                    grad_output +
                    i * embedding_dim;

                float* destination =
                    grad_weight_data +
                    token * embedding_dim;

                for (std::size_t d = 0;
                     d < embedding_dim;
                     ++d) {

                    destination[d] +=
                        source[d];
                }
            }

            // ------------------------------------------------
            // Kirim gradient ke weight.
            // ------------------------------------------------

            weight.accumulate_grad(
                grad_weight
            );

            // Weight adalah leaf parameter,
            // sehingga biasanya tidak memiliki grad_fn.
            //
            // Kalau suatu saat weight dibuat sebagai hasil
            // operasi tensor lain, teruskan backward-nya.
            // ------------------------------------------------

            if (weight.grad_state()->grad_fn) {

                weight.grad_state()
                    ->grad_fn
                    ->backward(
                        grad_weight
                    );
            }
        }
    );
}

} // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

Embedding::Embedding(
    std::size_t vocab_size,
    std::size_t embedding_dim
)
    : vocab_size_(vocab_size),
      embedding_dim_(embedding_dim),
      weight_() {

    if (vocab_size == 0) {

        throw std::invalid_argument(
            "Embedding: vocab_size must be greater than zero"
        );
    }

    if (embedding_dim == 0) {

        throw std::invalid_argument(
            "Embedding: embedding_dim must be greater than zero"
        );
    }

    weight_ =
        Tensor::empty(
            {
                vocab_size_,
                embedding_dim_
            },
            DType::Float32,
            Device::cpu()
        );

    weight_.requires_grad_(true);

    reset_parameters();
}

// ============================================================
// RESET PARAMETERS
//
// Uniform initialization:
//
//     limit = 1 / sqrt(embedding_dim)
//
//     weight ~ U(-limit, +limit)
//
// Seed dibuat deterministic untuk sementara.
// ============================================================

void Embedding::reset_parameters() {

    if (weight_.dtype() !=
        DType::Float32) {

        throw std::runtime_error(
            "Embedding::reset_parameters: "
            "weight must be Float32"
        );
    }

    if (!weight_.device().is_cpu()) {

        throw std::runtime_error(
            "Embedding::reset_parameters: "
            "only CPU device is currently supported"
        );
    }

    const float limit =
        1.0f /
        std::sqrt(
            static_cast<float>(
                embedding_dim_
            )
        );

    std::mt19937 generator(
        0x56454E4Cu
    );

    std::uniform_real_distribution<float>
        distribution(
            -limit,
            limit
        );

    float* values =
        weight_.data_as<float>();

    for (std::size_t i = 0;
         i < weight_.numel();
         ++i) {

        values[i] =
            distribution(generator);
    }
}

// ============================================================
// FORWARD
//
// Input:
//
//     [N]
//
// Output:
//
//     [N, embedding_dim]
//
// Atau:
//
//     [B, N]
//
// Output:
//
//     [B, N, embedding_dim]
//
// Secara umum:
//
//     [...input_shape..., embedding_dim]
//
// ============================================================

Tensor Embedding::forward(
    const Tensor& input
) const {

    if (input.ndim() == 0) {

        throw std::runtime_error(
            "Embedding::forward: "
            "input must have at least one dimension"
        );
    }

    if (
        input.dtype() != DType::Int32 &&
        input.dtype() != DType::Int64
    ) {

        throw std::runtime_error(
            "Embedding::forward: "
            "input must use Int32 or Int64"
        );
    }

    if (!input.device().is_cpu()) {

        throw std::runtime_error(
            "Embedding::forward: "
            "only CPU device is currently supported"
        );
    }

    if (!weight_.device().is_cpu()) {

        throw std::runtime_error(
            "Embedding::forward: "
            "weight must be on CPU"
        );
    }

    // --------------------------------------------------------
    // Output shape
    //
    // Input:
    //
    //     [B, S]
    //
    // menjadi:
    //
    //     [B, S, D]
    // --------------------------------------------------------

    std::vector<std::size_t>
        output_dimensions =
            input.shape().dimensions();

    output_dimensions.push_back(
        embedding_dim_
    );

    Tensor output =
        Tensor::zeros(
            Shape(output_dimensions),
            DType::Float32,
            input.device()
        );

    const float* weight_data =
        weight_.data_as<float>();

    float* output_data =
        output.data_as<float>();

    // --------------------------------------------------------
    // Lookup
    // --------------------------------------------------------

    for (std::size_t i = 0;
         i < input.numel();
         ++i) {

        const std::size_t token =
            read_token_index(
                input,
                i,
                vocab_size_
            );

        const float* source =
            weight_data +
            token * embedding_dim_;

        float* destination =
            output_data +
            i * embedding_dim_;

        for (std::size_t d = 0;
             d < embedding_dim_;
             ++d) {

            destination[d] =
                source[d];
        }
    }

    // --------------------------------------------------------
    // Autograd
    //
    // Output menjadi non-leaf karena memiliki grad_fn.
    // --------------------------------------------------------

    if (weight_.requires_grad()) {

        std::shared_ptr<AutogradNode>
            node =
                make_embedding_node(
                    input,
                    weight_,
                    vocab_size_,
                    embedding_dim_
                );

        output.set_grad_fn(
            node
        );
    }

    return output;
}

// ============================================================
// WEIGHT
// ============================================================

const Tensor& Embedding::weight() const {
    return weight_;
}

Tensor& Embedding::weight() {
    return weight_;
}

// ============================================================
// METADATA
// ============================================================

std::size_t Embedding::vocab_size() const {
    return vocab_size_;
}

std::size_t Embedding::embedding_dim() const {
    return embedding_dim_;
}

} // namespace venla
