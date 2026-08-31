#include "venla/nn/layer_norm.hpp"
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

void expect_shape(
    const Tensor& tensor,
    const Shape& expected,
    const char* message
) {
    if (tensor.shape() != expected) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {

    try {

        // ====================================================
        // CONSTRUCTION
        // ====================================================

        LayerNorm layer_norm(
            4
        );

        expect(
            layer_norm.normalized_shape() == 4,
            "normalized_shape mismatch"
        );

        expect_close(
            layer_norm.eps(),
            1e-5f,
            1e-12f,
            "epsilon mismatch"
        );

        expect(
            layer_norm.weight().shape() == Shape({4}),
            "weight shape mismatch"
        );

        expect(
            layer_norm.bias().shape() == Shape({4}),
            "bias shape mismatch"
        );

        expect(
            layer_norm.weight().dtype() == DType::Float32,
            "weight dtype mismatch"
        );

        expect(
            layer_norm.bias().dtype() == DType::Float32,
            "bias dtype mismatch"
        );

        // LayerNorm weight and bias are trainable parameters.
        expect(
            layer_norm.weight().requires_grad(),
            "weight must require grad"
        );

        expect(
            layer_norm.bias().requires_grad(),
            "bias must require grad"
        );

        // ====================================================
        // SIMPLE 2D INPUT
        //
        // Each row:
        //
        // [1, 2, 3, 4]
        //
        // mean = 2.5
        // variance = 1.25
        //
        // normalized:
        //
        // [-1.34164, -0.44721,
        //   0.44721,  1.34164]
        // ====================================================

        Tensor input =
            Tensor::zeros(
                {2, 4},
                DType::Float32,
                Device::cpu()
            );

        float* input_values =
            input.data_as<float>();

        input_values[0] = 1.0f;
        input_values[1] = 2.0f;
        input_values[2] = 3.0f;
        input_values[3] = 4.0f;

        input_values[4] = 5.0f;
        input_values[5] = 6.0f;
        input_values[6] = 7.0f;
        input_values[7] = 8.0f;

        Tensor output =
            layer_norm.forward(
                input
            );

        expect_shape(
            output,
            Shape({2, 4}),
            "2D output shape mismatch"
        );

        const float* output_values =
            output.data_as<float>();

        const float expected0 =
            -1.3416408f;

        const float expected1 =
            -0.4472136f;

        const float expected2 =
             0.4472136f;

        const float expected3 =
             1.3416408f;

        expect_close(
            output_values[0],
            expected0,
            1e-4f,
            "row 0 dim 0 mismatch"
        );

        expect_close(
            output_values[1],
            expected1,
            1e-4f,
            "row 0 dim 1 mismatch"
        );

        expect_close(
            output_values[2],
            expected2,
            1e-4f,
            "row 0 dim 2 mismatch"
        );

        expect_close(
            output_values[3],
            expected3,
            1e-4f,
            "row 0 dim 3 mismatch"
        );

        expect_close(
            output_values[4],
            expected0,
            1e-4f,
            "row 1 dim 0 mismatch"
        );

        expect_close(
            output_values[5],
            expected1,
            1e-4f,
            "row 1 dim 1 mismatch"
        );

        expect_close(
            output_values[6],
            expected2,
            1e-4f,
            "row 1 dim 2 mismatch"
        );

        expect_close(
            output_values[7],
            expected3,
            1e-4f,
            "row 1 dim 3 mismatch"
        );

        // ====================================================
        // NORMALIZED MEAN
        // ====================================================

        for (std::size_t row = 0;
             row < 2;
             ++row) {

            float row_sum = 0.0f;

            for (std::size_t dimension = 0;
                 dimension < 4;
                 ++dimension) {

                row_sum +=
                    output_values[
                        row * 4 +
                        dimension
                    ];
            }

            const float row_mean =
                row_sum / 4.0f;

            expect_close(
                row_mean,
                0.0f,
                1e-5f,
                "normalized row mean mismatch"
            );
        }

        // ====================================================
        // NORMALIZED VARIANCE
        // ====================================================

        for (std::size_t row = 0;
             row < 2;
             ++row) {

            float row_sum = 0.0f;

            for (std::size_t dimension = 0;
                 dimension < 4;
                 ++dimension) {

                row_sum +=
                    output_values[
                        row * 4 +
                        dimension
                    ];
            }

            const float mean =
                row_sum / 4.0f;

            float variance = 0.0f;

            for (std::size_t dimension = 0;
                 dimension < 4;
                 ++dimension) {

                const float value =
                    output_values[
                        row * 4 +
                        dimension
                    ];

                const float difference =
                    value - mean;

                variance +=
                    difference *
                    difference;
            }

            variance /= 4.0f;

            expect_close(
                variance,
                1.0f,
                1e-4f,
                "normalized variance mismatch"
            );
        }

        // ====================================================
        // BATCHED 3D INPUT
        // ====================================================

        Tensor batch =
            Tensor::zeros(
                {2, 3, 4},
                DType::Float32,
                Device::cpu()
            );

        float* batch_values =
            batch.data_as<float>();

        for (std::size_t i = 0;
             i < batch.numel();
             ++i) {

            batch_values[i] =
                static_cast<float>(
                    i + 1
                );
        }

        Tensor batch_output =
            layer_norm.forward(
                batch
            );

        expect_shape(
            batch_output,
            Shape({2, 3, 4}),
            "3D output shape mismatch"
        );

        const float* normalized_batch =
            batch_output.data_as<float>();

        for (std::size_t batch_index = 0;
             batch_index < 2;
             ++batch_index) {

            for (std::size_t position = 0;
                 position < 3;
                 ++position) {

                const std::size_t offset =
                    batch_index * 12 +
                    position * 4;

                float row_sum = 0.0f;

                for (std::size_t dimension = 0;
                     dimension < 4;
                     ++dimension) {

                    row_sum +=
                        normalized_batch[
                            offset + dimension
                        ];
                }

                const float mean =
                    row_sum / 4.0f;

                expect_close(
                    mean,
                    0.0f,
                    1e-5f,
                    "3D normalized mean mismatch"
                );

                float variance = 0.0f;

                for (std::size_t dimension = 0;
                     dimension < 4;
                     ++dimension) {

                    const float value =
                        normalized_batch[
                            offset + dimension
                        ];

                    const float difference =
                        value - mean;

                    variance +=
                        difference *
                        difference;
                }

                variance /= 4.0f;

                expect_close(
                    variance,
                    1.0f,
                    1e-4f,
                    "3D normalized variance mismatch"
                );
            }
        }

        // ====================================================
        // AFFINE PARAMETERS
        //
        // gamma = 2
        // beta  = 1
        //
        // output = normalized * 2 + 1
        // ====================================================

        layer_norm.weight().data_as<float>()[0] = 2.0f;
        layer_norm.weight().data_as<float>()[1] = 2.0f;
        layer_norm.weight().data_as<float>()[2] = 2.0f;
        layer_norm.weight().data_as<float>()[3] = 2.0f;

        layer_norm.bias().data_as<float>()[0] = 1.0f;
        layer_norm.bias().data_as<float>()[1] = 1.0f;
        layer_norm.bias().data_as<float>()[2] = 1.0f;
        layer_norm.bias().data_as<float>()[3] = 1.0f;

        Tensor affine_output =
            layer_norm.forward(
                input
            );

        const float* affine_values =
            affine_output.data_as<float>();

        expect_close(
            affine_values[0],
            expected0 * 2.0f + 1.0f,
            1e-4f,
            "affine output dim 0 mismatch"
        );

        expect_close(
            affine_values[1],
            expected1 * 2.0f + 1.0f,
            1e-4f,
            "affine output dim 1 mismatch"
        );

        expect_close(
            affine_values[2],
            expected2 * 2.0f + 1.0f,
            1e-4f,
            "affine output dim 2 mismatch"
        );

        expect_close(
            affine_values[3],
            expected3 * 2.0f + 1.0f,
            1e-4f,
            "affine output dim 3 mismatch"
        );

        // ====================================================
        // AUTOGRAD
        //
        // Reset:
        //
        // gamma = 1
        // beta  = 0
        // ====================================================

        layer_norm.weight().data_as<float>()[0] = 1.0f;
        layer_norm.weight().data_as<float>()[1] = 1.0f;
        layer_norm.weight().data_as<float>()[2] = 1.0f;
        layer_norm.weight().data_as<float>()[3] = 1.0f;

        layer_norm.bias().data_as<float>()[0] = 0.0f;
        layer_norm.bias().data_as<float>()[1] = 0.0f;
        layer_norm.bias().data_as<float>()[2] = 0.0f;
        layer_norm.bias().data_as<float>()[3] = 0.0f;

        Tensor trainable =
            Tensor::zeros(
                {2, 4},
                DType::Float32,
                Device::cpu()
            );

        float* trainable_values =
            trainable.data_as<float>();

        trainable_values[0] = 1.0f;
        trainable_values[1] = 2.0f;
        trainable_values[2] = 3.0f;
        trainable_values[3] = 4.0f;

        trainable_values[4] = 5.0f;
        trainable_values[5] = 6.0f;
        trainable_values[6] = 7.0f;
        trainable_values[7] = 8.0f;

        trainable.requires_grad_(true);

        Tensor encoded =
            layer_norm.forward(
                trainable
            );

        expect(
            encoded.requires_grad(),
            "LayerNorm output must require grad"
        );

        Tensor loss =
            venla::sum(
                encoded
            );

        expect(
            loss.numel() == 1,
            "LayerNorm loss must be scalar"
        );

        loss.backward();

        expect(
            trainable.has_grad(),
            "LayerNorm input gradient missing"
        );

        const Tensor& trainable_gradient =
            trainable.grad();

        expect_shape(
            trainable_gradient,
            Shape({2, 4}),
            "LayerNorm input gradient shape mismatch"
        );

        const float* gradient_values =
            trainable_gradient.data_as<float>();

        // For gamma=1 and beta=0:
        //
        // sum(LayerNorm(x))
        //
        // is theoretically constant for each row.
        //
        // Therefore:
        //
        // dLoss/dx ~= 0

        for (std::size_t i = 0;
             i < trainable_gradient.numel();
             ++i) {

            expect_close(
                gradient_values[i],
                0.0f,
                1e-3f,
                "LayerNorm input gradient mismatch"
            );
        }

        // ====================================================
        // WEIGHT / BIAS GRADIENT
        // ====================================================

        expect(
            layer_norm.weight().has_grad(),
            "LayerNorm weight gradient missing"
        );

        expect(
            layer_norm.bias().has_grad(),
            "LayerNorm bias gradient missing"
        );

        expect_shape(
            layer_norm.weight().grad(),
            Shape({4}),
            "LayerNorm weight gradient shape mismatch"
        );

        expect_shape(
            layer_norm.bias().grad(),
            Shape({4}),
            "LayerNorm bias gradient shape mismatch"
        );

        // ====================================================
        // INVALID NORMALIZED DIMENSION
        // ====================================================

        Tensor wrong_dimension =
            Tensor::zeros(
                {2, 5},
                DType::Float32,
                Device::cpu()
            );

        bool dimension_failed =
            false;

        try {

            layer_norm.forward(
                wrong_dimension
            );
        }
        catch (const std::runtime_error&) {

            dimension_failed =
                true;
        }

        expect(
            dimension_failed,
            "wrong normalized dimension must throw"
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

            layer_norm.forward(
                one_dimensional
            );
        }
        catch (const std::runtime_error&) {

            rank_failed =
                true;
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
                {2, 4},
                DType::Int32,
                Device::cpu()
            );

        bool dtype_failed =
            false;

        try {

            layer_norm.forward(
                wrong_dtype
            );
        }
        catch (const std::runtime_error&) {

            dtype_failed =
                true;
        }

        expect(
            dtype_failed,
            "non-Float32 input must throw"
        );

        // ====================================================
        // PASS
        // ====================================================

        std::cout
            << "VENLACPU LayerNorm Test: PASS"
            << std::endl;

        return 0;
    }
    catch (const std::exception& error) {

        std::cerr
            << "VENLACPU LayerNorm Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
