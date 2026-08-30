#include "venla/nn/mse_loss.hpp"
#include "venla/tensor/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

using namespace venla;

namespace {

void expect_close(
    float actual,
    float expected,
    float tolerance = 1e-5f
) {
    if (std::fabs(actual - expected) > tolerance) {

        throw std::runtime_error(
            "value mismatch: actual=" +
            std::to_string(actual) +
            " expected=" +
            std::to_string(expected)
        );
    }
}

void fill(
    Tensor& tensor,
    const float* values
) {
    float* data =
        tensor.data_as<float>();

    for (std::size_t i = 0;
         i < tensor.numel();
         ++i) {

        data[i] =
            values[i];
    }
}

// ============================================================
// MEAN
// ============================================================

void test_mean_autograd() {

    Tensor prediction =
        Tensor::zeros(
            {3},
            DType::Float32,
            Device::cpu()
        );

    Tensor target =
        Tensor::zeros(
            {3},
            DType::Float32,
            Device::cpu()
        );

    const float p[] = {
        1.0f,
        2.0f,
        4.0f
    };

    const float t[] = {
        0.0f,
        1.0f,
        2.0f
    };

    fill(prediction, p);
    fill(target, t);

    prediction.requires_grad_();

    MSELoss loss_fn(
        Reduction::Mean
    );

    Tensor loss =
        loss_fn.forward(
            prediction,
            target
        );

    expect_close(
        loss.data_as<float>()[0],
        2.0f
    );

    loss.backward();

    const Tensor& grad =
        prediction.grad();

    expect_close(
        grad.data_as<float>()[0],
        2.0f / 3.0f
    );

    expect_close(
        grad.data_as<float>()[1],
        2.0f / 3.0f
    );

    expect_close(
        grad.data_as<float>()[2],
        4.0f / 3.0f
    );
}

// ============================================================
// SUM
// ============================================================

void test_sum_autograd() {

    Tensor prediction =
        Tensor::zeros(
            {3},
            DType::Float32,
            Device::cpu()
        );

    Tensor target =
        Tensor::zeros(
            {3},
            DType::Float32,
            Device::cpu()
        );

    const float p[] = {
        1.0f,
        2.0f,
        4.0f
    };

    const float t[] = {
        0.0f,
        1.0f,
        2.0f
    };

    fill(prediction, p);
    fill(target, t);

    prediction.requires_grad_();

    MSELoss loss_fn(
        Reduction::Sum
    );

    Tensor loss =
        loss_fn.forward(
            prediction,
            target
        );

    expect_close(
        loss.data_as<float>()[0],
        6.0f
    );

    loss.backward();

    const Tensor& grad =
        prediction.grad();

    expect_close(
        grad.data_as<float>()[0],
        2.0f
    );

    expect_close(
        grad.data_as<float>()[1],
        2.0f
    );

    expect_close(
        grad.data_as<float>()[2],
        4.0f
    );
}

// ============================================================
// NONE
// ============================================================

void test_none_autograd() {

    Tensor prediction =
        Tensor::zeros(
            {3},
            DType::Float32,
            Device::cpu()
        );

    Tensor target =
        Tensor::zeros(
            {3},
            DType::Float32,
            Device::cpu()
        );

    const float p[] = {
        1.0f,
        2.0f,
        4.0f
    };

    const float t[] = {
        0.0f,
        1.0f,
        2.0f
    };

    fill(prediction, p);
    fill(target, t);

    prediction.requires_grad_();

    MSELoss loss_fn(
        Reduction::None
    );

    Tensor loss =
        loss_fn.forward(
            prediction,
            target
        );

    expect_close(
        loss.data_as<float>()[0],
        1.0f
    );

    expect_close(
        loss.data_as<float>()[1],
        1.0f
    );

    expect_close(
        loss.data_as<float>()[2],
        4.0f
    );

    Tensor upstream =
        Tensor::ones(
            {3},
            DType::Float32,
            Device::cpu()
        );

    loss.backward(
        upstream
    );

    const Tensor& grad =
        prediction.grad();

    expect_close(
        grad.data_as<float>()[0],
        2.0f
    );

    expect_close(
        grad.data_as<float>()[1],
        2.0f
    );

    expect_close(
        grad.data_as<float>()[2],
        4.0f
    );
}

} // namespace

int main() {

    try {

        test_mean_autograd();
        test_sum_autograd();
        test_none_autograd();

        std::cout
            << "MSE autograd tests passed."
            << std::endl;

        return 0;

    }
    catch (const std::exception& error) {

        std::cerr
            << "MSE autograd test failed: "
            << error.what()
            << std::endl;

        return 1;
    }
}
