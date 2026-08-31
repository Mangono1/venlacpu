#include "venla/nn/multi_head_attention.hpp"
#include "venla/math/operations.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

using namespace venla;

namespace {

void expect(
    bool condition,
    const char* message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_close(
    float actual,
    float expected,
    float tolerance,
    const char* message
) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void expect_finite(
    const Tensor& tensor,
    const char* message
) {
    const float* data =
        tensor.data_as<float>();

    for (std::size_t i = 0;
         i < tensor.numel();
         ++i) {

        if (!std::isfinite(data[i])) {
            throw std::runtime_error(message);
        }
    }
}

} // namespace

int main() {

    try {

        // ====================================================
        // CONSTRUCTION
        // ====================================================

        MultiHeadAttention attention(
            8,
            2
        );

        expect(
            attention.embed_dim() == 8,
            "embed_dim mismatch"
        );

        expect(
            attention.num_heads() == 2,
            "num_heads mismatch"
        );

        expect(
            attention.head_dim() == 4,
            "head_dim mismatch"
        );

        expect(
            attention.has_bias(),
            "bias should be enabled by default"
        );

        expect(
            !attention.is_causal(),
            "causal should be disabled by default"
        );

        // ====================================================
        // PARAMETER SHAPES
        // ====================================================

        expect(
            attention.q_weight().shape()
                == Shape({8, 8}),
            "q weight shape mismatch"
        );

        expect(
            attention.k_weight().shape()
                == Shape({8, 8}),
            "k weight shape mismatch"
        );

        expect(
            attention.v_weight().shape()
                == Shape({8, 8}),
            "v weight shape mismatch"
        );

        expect(
            attention.out_weight().shape()
                == Shape({8, 8}),
            "out weight shape mismatch"
        );

        expect(
            attention.q_bias().shape()
                == Shape({8}),
            "q bias shape mismatch"
        );

        expect(
            attention.k_bias().shape()
                == Shape({8}),
            "k bias shape mismatch"
        );

        expect(
            attention.v_bias().shape()
                == Shape({8}),
            "v bias shape mismatch"
        );

        expect(
            attention.out_bias().shape()
                == Shape({8}),
            "out bias shape mismatch"
        );

        expect(
            attention.q_weight().requires_grad(),
            "q weight must require grad"
        );

        expect(
            attention.k_weight().requires_grad(),
            "k weight must require grad"
        );

        expect(
            attention.v_weight().requires_grad(),
            "v weight must require grad"
        );

        expect(
            attention.out_weight().requires_grad(),
            "out weight must require grad"
        );

        // ====================================================
        // 2D FORWARD
        // ====================================================

        Tensor input =
            Tensor::zeros(
                {3, 8},
                DType::Float32,
                Device::cpu()
            );

        float* input_data =
            input.data_as<float>();

        for (std::size_t i = 0;
             i < input.numel();
             ++i) {

            input_data[i] =
                static_cast<float>(
                    i + 1
                ) /
                10.0f;
        }

        Tensor output =
            attention.forward(
                input
            );

        expect(
            output.shape()
                == Shape({3, 8}),
            "2D output shape mismatch"
        );

        expect_finite(
            output,
            "2D output contains non-finite values"
        );

        // ====================================================
        // 3D BATCHED FORWARD
        // ====================================================

        Tensor batch =
            Tensor::zeros(
                {2, 3, 8},
                DType::Float32,
                Device::cpu()
            );

        float* batch_data =
            batch.data_as<float>();

        for (std::size_t i = 0;
             i < batch.numel();
             ++i) {

            batch_data[i] =
                static_cast<float>(
                    (i % 17) + 1
                ) /
                17.0f;
        }

        Tensor batch_output =
            attention.forward(
                batch
            );

        expect(
            batch_output.shape()
                == Shape({2, 3, 8}),
            "3D output shape mismatch"
        );

        expect_finite(
            batch_output,
            "3D output contains non-finite values"
        );

        // ====================================================
        // INVALID EMBED DIMENSION
        // ====================================================

        Tensor wrong_dimension =
            Tensor::zeros(
                {3, 7},
                DType::Float32,
                Device::cpu()
            );

        bool dimension_failed =
            false;

        try {
            attention.forward(
                wrong_dimension
            );
        }
        catch (const std::runtime_error&) {
            dimension_failed = true;
        }

        expect(
            dimension_failed,
            "wrong embedding dimension must throw"
        );

        // ====================================================
        // INVALID RANK
        // ====================================================

        Tensor wrong_rank =
            Tensor::zeros(
                {8},
                DType::Float32,
                Device::cpu()
            );

        bool rank_failed =
            false;

        try {
            attention.forward(
                wrong_rank
            );
        }
        catch (const std::runtime_error&) {
            rank_failed = true;
        }

        expect(
            rank_failed,
            "1D input must throw"
        );

        // ====================================================
        // INVALID HEAD CONFIGURATION
        // ====================================================

        bool head_failed =
            false;

        try {
            MultiHeadAttention invalid(
                7,
                2
            );
        }
        catch (const std::invalid_argument&) {
            head_failed = true;
        }

        expect(
            head_failed,
            "non-divisible head configuration must throw"
        );

        // ====================================================
        // CAUSAL MODE
        // ====================================================

        MultiHeadAttention causal_attention(
            4,
            2,
            true,
            true
        );

        expect(
            causal_attention.is_causal(),
            "causal flag mismatch"
        );

        Tensor causal_input =
            Tensor::ones(
                {4, 4},
                DType::Float32,
                Device::cpu()
            );

        Tensor causal_output =
            causal_attention.forward(
                causal_input
            );

        expect(
            causal_output.shape()
                == Shape({4, 4}),
            "causal output shape mismatch"
        );

        expect_finite(
            causal_output,
            "causal output contains non-finite values"
        );

        causal_attention.set_causal(
            false
        );

        expect(
            !causal_attention.is_causal(),
            "set_causal(false) failed"
        );

        // ====================================================
        // AUTOGRAD INPUT
        // ====================================================

        Tensor trainable =
            Tensor::ones(
                {2, 3, 8},
                DType::Float32,
                Device::cpu()
            );

        trainable.requires_grad_(true);

        Tensor encoded =
            attention.forward(
                trainable
            );

        expect(
            encoded.requires_grad(),
            "attention output must require grad"
        );

        Tensor loss =
            sum(encoded);

        loss.backward();

        expect(
            trainable.has_grad(),
            "input gradient missing"
        );

        expect(
            trainable.grad().shape()
                == Shape({2, 3, 8}),
            "input gradient shape mismatch"
        );

        expect_finite(
            trainable.grad(),
            "input gradient contains non-finite values"
        );

        // ====================================================
        // PARAMETER GRADIENTS
        // ====================================================

        expect(
            attention.q_weight().has_grad(),
            "q weight gradient missing"
        );

        expect(
            attention.k_weight().has_grad(),
            "k weight gradient missing"
        );

        expect(
            attention.v_weight().has_grad(),
            "v weight gradient missing"
        );

        expect(
            attention.out_weight().has_grad(),
            "output weight gradient missing"
        );

        expect(
            attention.q_bias().has_grad(),
            "q bias gradient missing"
        );

        expect(
            attention.k_bias().has_grad(),
            "k bias gradient missing"
        );

        expect(
            attention.v_bias().has_grad(),
            "v bias gradient missing"
        );

        expect(
            attention.out_bias().has_grad(),
            "output bias gradient missing"
        );

        expect_finite(
            attention.q_weight().grad(),
            "q weight gradient contains non-finite values"
        );

        expect_finite(
            attention.k_weight().grad(),
            "k weight gradient contains non-finite values"
        );

        expect_finite(
            attention.v_weight().grad(),
            "v weight gradient contains non-finite values"
        );

        expect_finite(
            attention.out_weight().grad(),
            "output weight gradient contains non-finite values"
        );

        // ====================================================
        // PASS
        // ====================================================

        std::cout
            << "VENLACPU Multi-Head Attention Test: PASS"
            << std::endl;

        return 0;
    }

    catch (const std::exception& error) {

        std::cerr
            << "VENLACPU Multi-Head Attention Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
