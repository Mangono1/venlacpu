#include "venla/nn/positional_encoding.hpp"

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

} // namespace

int main() {

    try {

        // ====================================================
        // CONSTRUCTION
        // ====================================================

        PositionalEncoding positional(
            16,
            4
        );

        expect(
            positional.max_seq_len() == 16,
            "max_seq_len mismatch"
        );

        expect(
            positional.embedding_dim() == 4,
            "embedding_dim mismatch"
        );

        expect(
            positional.encoding().shape()
                == Shape({16, 4}),
            "encoding shape mismatch"
        );

        expect(
            positional.encoding().dtype()
                == DType::Float32,
            "encoding dtype mismatch"
        );

        expect(
            !positional.encoding().requires_grad(),
            "positional encoding must not require grad"
        );

        // ====================================================
        // CHECK POSITION 0
        //
        // sin(0) = 0
        // cos(0) = 1
        // ====================================================

        const float* encoding =
            positional.encoding().data_as<float>();

        expect_close(
            encoding[0],
            0.0f,
            1e-6f,
            "position 0 dimension 0 mismatch"
        );

        expect_close(
            encoding[1],
            1.0f,
            1e-6f,
            "position 0 dimension 1 mismatch"
        );

        expect_close(
            encoding[2],
            0.0f,
            1e-6f,
            "position 0 dimension 2 mismatch"
        );

        expect_close(
            encoding[3],
            1.0f,
            1e-6f,
            "position 0 dimension 3 mismatch"
        );

        // ====================================================
        // CHECK POSITION 1
        //
        // dim 0:
        //
        // sin(1)
        //
        // dim 1:
        //
        // cos(1)
        //
        // dim 2:
        //
        // sin(1 / 100)
        //
        // dim 3:
        //
        // cos(1 / 100)
        // ====================================================

        expect_close(
            encoding[4],
            static_cast<float>(std::sin(1.0)),
            1e-5f,
            "position 1 dimension 0 mismatch"
        );

        expect_close(
            encoding[5],
            static_cast<float>(std::cos(1.0)),
            1e-5f,
            "position 1 dimension 1 mismatch"
        );

        expect_close(
            encoding[6],
            static_cast<float>(std::sin(0.01)),
            1e-5f,
            "position 1 dimension 2 mismatch"
        );

        expect_close(
            encoding[7],
            static_cast<float>(std::cos(0.01)),
            1e-5f,
            "position 1 dimension 3 mismatch"
        );

        // ====================================================
        // FORWARD 2D
        // ====================================================

        Tensor input =
            Tensor::zeros(
                {3, 4},
                DType::Float32,
                Device::cpu()
            );

        float* input_values =
            input.data_as<float>();

        for (std::size_t i = 0;
             i < input.numel();
             ++i) {

            input_values[i] = 10.0f;
        }

        Tensor output =
            positional.forward(
                input
            );

        expect(
            output.shape()
                == Shape({3, 4}),
            "2D output shape mismatch"
        );

        const float* output_values =
            output.data_as<float>();

        // Position 0:
        //
        // [10, 11, 10, 11]

        expect_close(
            output_values[0],
            10.0f,
            1e-6f,
            "forward position 0 dim 0 mismatch"
        );

        expect_close(
            output_values[1],
            11.0f,
            1e-6f,
            "forward position 0 dim 1 mismatch"
        );

        expect_close(
            output_values[2],
            10.0f,
            1e-6f,
            "forward position 0 dim 2 mismatch"
        );

        expect_close(
            output_values[3],
            11.0f,
            1e-6f,
            "forward position 0 dim 3 mismatch"
        );

        // Position 1.

        expect_close(
            output_values[4],
            10.0f +
                static_cast<float>(
                    std::sin(1.0)
                ),
            1e-5f,
            "forward position 1 dim 0 mismatch"
        );

        expect_close(
            output_values[5],
            10.0f +
                static_cast<float>(
                    std::cos(1.0)
                ),
            1e-5f,
            "forward position 1 dim 1 mismatch"
        );

        // ====================================================
        // BATCHED INPUT
        // ====================================================

        Tensor batch =
            Tensor::zeros(
                {2, 3, 4},
                DType::Float32,
                Device::cpu()
            );

        Tensor batch_output =
            positional.forward(
                batch
            );

        expect(
            batch_output.shape()
                == Shape({2, 3, 4}),
            "batched output shape mismatch"
        );

        const float* batch_values =
            batch_output.data_as<float>();

        // Both batches must receive exactly the same
        // positional encoding.

        for (std::size_t position = 0;
             position < 3;
             ++position) {

            for (std::size_t dimension = 0;
                 dimension < 4;
                 ++dimension) {

                const std::size_t first =
                    position * 4 +
                    dimension;

                const std::size_t second =
                    12 +
                    position * 4 +
                    dimension;

                expect_close(
                    batch_values[first],
                    batch_values[second],
                    1e-6f,
                    "batch positional encoding mismatch"
                );
            }
        }

        // ====================================================
        // AUTOGRAD
        //
        // output = input + constant
        //
        // output shape:
        //
        //   [2, 3, 4]
        //
        // Since output is non-scalar, backward() without
        // an explicit gradient is intentionally invalid in
        // VENLACPU.
        //
        // Use an upstream gradient of ones:
        //
        //   dL/doutput = 1
        //
        // Therefore:
        //
        //   dL/dinput = 1
        // ====================================================

        Tensor trainable =
            Tensor::ones(
                {2, 3, 4},
                DType::Float32,
                Device::cpu()
            );

        trainable.requires_grad_(true);

        Tensor encoded =
            positional.forward(
                trainable
            );

        expect(
            encoded.requires_grad(),
            "positional output must require grad"
        );

        Tensor upstream_gradient =
            Tensor::ones(
                encoded.shape(),
                DType::Float32,
                encoded.device()
            );

        encoded.backward(
            upstream_gradient
        );

        expect(
            trainable.has_grad(),
            "input gradient missing"
        );

        const Tensor& gradient =
            trainable.grad();

        expect(
            gradient.shape()
                == Shape({2, 3, 4}),
            "input gradient shape mismatch"
        );

        const float* gradient_values =
            gradient.data_as<float>();

        for (std::size_t i = 0;
             i < gradient.numel();
             ++i) {

            expect_close(
                gradient_values[i],
                1.0f,
                1e-6f,
                "positional encoding gradient mismatch"
            );
        }

        // Positional encoding itself must remain constant.

        expect(
            !positional.encoding().has_grad(),
            "positional encoding unexpectedly has gradient"
        );

        // ====================================================
        // INVALID EMBEDDING DIMENSION
        // ====================================================

        Tensor wrong_dimension =
            Tensor::zeros(
                {3, 5},
                DType::Float32,
                Device::cpu()
            );

        bool dimension_failed =
            false;

        try {

            positional.forward(
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
        // INVALID SEQUENCE LENGTH
        // ====================================================

        Tensor too_long =
            Tensor::zeros(
                {17, 4},
                DType::Float32,
                Device::cpu()
            );

        bool length_failed =
            false;

        try {

            positional.forward(
                too_long
            );

        }
        catch (const std::out_of_range&) {

            length_failed = true;
        }

        expect(
            length_failed,
            "sequence longer than max_seq_len must throw"
        );

        // ====================================================
        // INVALID RANK
        // ====================================================

        Tensor one_dimensional =
            Tensor::zeros(
                {4},
                DType::Float32,
                Device::cpu()
            );

        bool rank_failed =
            false;

        try {

            positional.forward(
                one_dimensional
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
        // INVALID DTYPE
        // ====================================================

        Tensor wrong_dtype =
            Tensor::zeros(
                {3, 4},
                DType::Int32,
                Device::cpu()
            );

        bool dtype_failed =
            false;

        try {

            positional.forward(
                wrong_dtype
            );

        }
        catch (const std::runtime_error&) {

            dtype_failed = true;
        }

        expect(
            dtype_failed,
            "non-Float32 input must throw"
        );

        // ====================================================
        // PASS
        // ====================================================

        std::cout
            << "VENLACPU Positional Encoding Test: PASS"
            << std::endl;

        return 0;
    }

    catch (const std::exception& error) {

        std::cerr
            << "VENLACPU Positional Encoding Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
