#include <cassert>
#include <cmath>
#include <iostream>

#include "venla/optim/optimizer.hpp"
#include "venla/math/operations.hpp"
#include "venla/tensor/tensor.hpp"

namespace {

void assert_close(
    float actual,
    float expected,
    float tolerance = 1e-5f
) {
    assert(
        std::fabs(actual - expected)
        <= tolerance
    );
}

} // namespace

int main() {

    // ========================================================
    // SGD
    // ========================================================

    {
        venla::Tensor parameter =
            venla::Tensor::zeros(
                {2},
                venla::DType::Float32
            );

        parameter.requires_grad_(true);

        float* data =
            parameter.data_as<float>();

        data[0] = 1.0f;
        data[1] = 2.0f;

        venla::Tensor gradient =
            venla::Tensor::ones(
                {2},
                venla::DType::Float32
            );

        parameter.accumulate_grad(
            gradient
        );

        venla::SGD optimizer(
            0.1f
        );

        optimizer.add_parameter(
            parameter
        );

        assert(
            optimizer.parameter_count()
            == 1
        );

        optimizer.step();

        data =
            parameter.data_as<float>();

        assert_close(
            data[0],
            0.9f
        );

        assert_close(
            data[1],
            1.9f
        );

        optimizer.zero_grad();

        assert(
            !parameter.has_grad()
        );
    }

    // ========================================================
    // SGD MOMENTUM
    // ========================================================

    {
        venla::Tensor parameter =
            venla::Tensor::zeros(
                {1},
                venla::DType::Float32
            );

        parameter.requires_grad_(true);

        parameter.data_as<float>()[0] =
            1.0f;

        venla::SGD optimizer(
            0.1f,
            0.9f
        );

        optimizer.add_parameter(
            parameter
        );

        venla::Tensor gradient =
            venla::Tensor::ones(
                {1},
                venla::DType::Float32
            );

        parameter.accumulate_grad(
            gradient
        );

        optimizer.step();

        assert_close(
            parameter.data_as<float>()[0],
            0.9f
        );

        optimizer.zero_grad();

        parameter.accumulate_grad(
            gradient
        );

        optimizer.step();

        // Momentum buffer:
        //
        // v1 = 1
        // p1 = 1 - 0.1 * 1 = 0.9
        //
        // v2 = 0.9 * 1 + 1 = 1.9
        // p2 = 0.9 - 0.1 * 1.9 = 0.71

        assert_close(
            parameter.data_as<float>()[0],
            0.71f
        );
    }

    // ========================================================
    // ADAM
    // ========================================================

    {
        venla::Tensor parameter =
            venla::Tensor::zeros(
                {2},
                venla::DType::Float32
            );

        parameter.requires_grad_(true);

        float* data =
            parameter.data_as<float>();

        data[0] = 1.0f;
        data[1] = 2.0f;

        venla::Adam optimizer(
            0.1f
        );

        optimizer.add_parameter(
            parameter
        );

        venla::Tensor gradient =
            venla::Tensor::ones(
                {2},
                venla::DType::Float32
            );

        parameter.accumulate_grad(
            gradient
        );

        optimizer.step();

        // Pada step pertama Adam dengan grad=1:
        //
        // m_hat = 1
        // v_hat = 1
        //
        // parameter -= 0.1

        data =
            parameter.data_as<float>();

        assert_close(
            data[0],
            0.9f
        );

        assert_close(
            data[1],
            1.9f
        );

        assert(
            optimizer.step_count()
            == 1
        );

        optimizer.zero_grad();

        parameter.accumulate_grad(
            gradient
        );

        optimizer.step();

        data =
            parameter.data_as<float>();

        assert(
            data[0] < 0.9f
        );

        assert(
            data[1] < 1.9f
        );

        assert(
            optimizer.step_count()
            == 2
        );
    }

    // ========================================================
    // ADAM LEARNING RATE
    // ========================================================

    {
        venla::Adam optimizer(
            0.001f
        );

        assert_close(
            optimizer.learning_rate(),
            0.001f
        );

        optimizer.set_learning_rate(
            0.01f
        );

        assert_close(
            optimizer.learning_rate(),
            0.01f
        );
    }

    std::cout
        << "VENLACPU optimizer tests passed\n";

    return 0;
}
