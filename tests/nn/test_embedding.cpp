#include "venla/nn/embedding.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
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

        Embedding embedding(
            10,
            4
        );

        expect(
            embedding.vocab_size() == 10,
            "vocab_size mismatch"
        );

        expect(
            embedding.embedding_dim() == 4,
            "embedding_dim mismatch"
        );

        expect(
            embedding.weight().shape()
                == Shape({10, 4}),
            "weight shape mismatch"
        );

        expect(
            embedding.weight().dtype()
                == DType::Float32,
            "weight dtype mismatch"
        );

        expect(
            embedding.weight().requires_grad(),
            "embedding weight must require grad"
        );

        // ====================================================
        // MANUAL WEIGHT
        //
        // Supaya lookup dapat diverifikasi dengan exact value.
        // ====================================================

        float* weight =
            embedding.weight().data_as<float>();

        for (std::size_t token = 0;
             token < 10;
             ++token) {

            for (std::size_t dim = 0;
                 dim < 4;
                 ++dim) {

                weight[
                    token * 4 + dim
                ] =
                    static_cast<float>(
                        token * 10 + dim
                    );
            }
        }

        // ====================================================
        // INT32 LOOKUP
        // ====================================================

        Tensor input =
            Tensor::zeros(
                {3},
                DType::Int32,
                Device::cpu()
            );

        std::int32_t* ids =
            input.data_as<std::int32_t>();

        ids[0] = 2;
        ids[1] = 5;
        ids[2] = 7;

        Tensor output =
            embedding.forward(
                input
            );

        expect(
            output.shape()
                == Shape({3, 4}),
            "Int32 output shape mismatch"
        );

        const float* values =
            output.data_as<float>();

        // token 2 -> [20,21,22,23]
        expect_close(
            values[0],
            20.0f,
            1e-6f,
            "lookup token 2 dim 0 failed"
        );

        expect_close(
            values[1],
            21.0f,
            1e-6f,
            "lookup token 2 dim 1 failed"
        );

        // token 5 -> [50,51,52,53]
        expect_close(
            values[4],
            50.0f,
            1e-6f,
            "lookup token 5 dim 0 failed"
        );

        // token 7 -> [70,71,72,73]
        expect_close(
            values[8],
            70.0f,
            1e-6f,
            "lookup token 7 dim 0 failed"
        );

        expect(
            output.requires_grad(),
            "embedding output must require grad"
        );

        // ====================================================
        // BATCHED INPUT
        // ====================================================

        Tensor batch =
            Tensor::zeros(
                {2, 3},
                DType::Int64,
                Device::cpu()
            );

        std::int64_t* batch_ids =
            batch.data_as<std::int64_t>();

        batch_ids[0] = 1;
        batch_ids[1] = 2;
        batch_ids[2] = 3;

        batch_ids[3] = 4;
        batch_ids[4] = 5;
        batch_ids[5] = 6;

        Tensor batch_output =
            embedding.forward(
                batch
            );

        expect(
            batch_output.shape()
                == Shape({2, 3, 4}),
            "batched output shape mismatch"
        );

        const float* batch_values =
            batch_output.data_as<float>();

        expect_close(
            batch_values[0],
            10.0f,
            1e-6f,
            "batch token 1 failed"
        );

        expect_close(
            batch_values[20],
            60.0f,
            1e-6f,
            "batch token 6 failed"
        );

        // ====================================================
        // REPEATED TOKEN
        // ====================================================

        Tensor repeated =
            Tensor::zeros(
                {3},
                DType::Int32,
                Device::cpu()
            );

        std::int32_t* repeated_ids =
            repeated.data_as<std::int32_t>();

        repeated_ids[0] = 3;
        repeated_ids[1] = 3;
        repeated_ids[2] = 7;

        Tensor repeated_output =
            embedding.forward(
                repeated
            );

        expect(
            repeated_output.shape()
                == Shape({3, 4}),
            "repeated output shape mismatch"
        );

        // ====================================================
        // BACKWARD
        //
        // repeated_output memiliki shape [3, 4], sehingga
        // backward() tanpa argumen TIDAK boleh digunakan.
        //
        // Tensor::backward() implicit gradient hanya berlaku
        // untuk tensor scalar.
        //
        // Karena test ini ingin gradient = 1 untuk setiap
        // elemen output, buat explicit gradient:
        //
        //     [3, 4] = seluruhnya 1
        //
        // Token 3 muncul dua kali.
        //
        // Jadi:
        //
        //     grad[3] = [2,2,2,2]
        //     grad[7] = [1,1,1,1]
        // ====================================================

        embedding.weight().zero_grad();

        Tensor output_gradient =
            Tensor::ones(
                repeated_output.shape(),
                DType::Float32,
                Device::cpu()
            );

        repeated_output.backward(
            output_gradient
        );

        expect(
            embedding.weight().has_grad(),
            "embedding weight has no gradient"
        );

        const Tensor& grad =
            embedding.weight().grad();

        expect(
            grad.shape()
                == Shape({10, 4}),
            "embedding gradient shape mismatch"
        );

        const float* grad_values =
            grad.data_as<float>();

        for (std::size_t d = 0;
             d < 4;
             ++d) {

            expect_close(
                grad_values[3 * 4 + d],
                2.0f,
                1e-6f,
                "repeated token gradient failed"
            );

            expect_close(
                grad_values[7 * 4 + d],
                1.0f,
                1e-6f,
                "token 7 gradient failed"
            );
        }

        // Token yang tidak digunakan harus nol.
        for (std::size_t token = 0;
             token < 10;
             ++token) {

            if (token == 3 || token == 7) {
                continue;
            }

            for (std::size_t d = 0;
                 d < 4;
                 ++d) {

                expect_close(
                    grad_values[
                        token * 4 + d
                    ],
                    0.0f,
                    1e-6f,
                    "unused embedding row has gradient"
                );
            }
        }

        // ====================================================
        // INVALID NEGATIVE TOKEN
        // ====================================================

        Tensor negative =
            Tensor::zeros(
                {1},
                DType::Int32,
                Device::cpu()
            );

        negative.data_as<std::int32_t>()[0] =
            -1;

        bool negative_failed =
            false;

        try {

            embedding.forward(
                negative
            );

        }
        catch (const std::out_of_range&) {

            negative_failed = true;
        }

        expect(
            negative_failed,
            "negative token must throw"
        );

        // ====================================================
        // INVALID OUT-OF-RANGE TOKEN
        // ====================================================

        Tensor invalid =
            Tensor::zeros(
                {1},
                DType::Int64,
                Device::cpu()
            );

        invalid.data_as<std::int64_t>()[0] =
            10;

        bool invalid_failed =
            false;

        try {

            embedding.forward(
                invalid
            );

        }
        catch (const std::out_of_range&) {

            invalid_failed = true;
        }

        expect(
            invalid_failed,
            "out-of-range token must throw"
        );

        // ====================================================
        // INVALID FLOAT INPUT
        // ====================================================

        Tensor wrong_dtype =
            Tensor::zeros(
                {1},
                DType::Float32,
                Device::cpu()
            );

        bool dtype_failed =
            false;

        try {

            embedding.forward(
                wrong_dtype
            );

        }
        catch (const std::runtime_error&) {

            dtype_failed = true;
        }

        expect(
            dtype_failed,
            "Float32 token input must throw"
        );

        // ====================================================
        // SUCCESS
        // ====================================================

        std::cout
            << "VENLACPU Embedding Test: PASS"
            << std::endl;

        return 0;

    }
    catch (const std::exception& error) {

        std::cerr
            << "VENLACPU Embedding Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}

