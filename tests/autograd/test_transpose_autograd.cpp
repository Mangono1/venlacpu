#include "venla/math/operations.hpp"
#include "venla/tensor/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace {

void expect_close(
    float actual,
    float expected,
    float tolerance = 1e-5f
) {
    if (std::fabs(actual - expected) >
        tolerance) {

        throw std::runtime_error(
            "value mismatch"
        );
    }
}

void test_2d_transpose_backward() {

    venla::Tensor x =
        venla::Tensor::zeros(
            {2, 3},
            venla::DType::Float32,
            venla::Device::cpu()
        );

    float* x_data =
        x.data_as<float>();

    x_data[0] = 1.0f;
    x_data[1] = 2.0f;
    x_data[2] = 3.0f;

    x_data[3] = 4.0f;
    x_data[4] = 5.0f;
    x_data[5] = 6.0f;

    x.requires_grad_(true);

    venla::Tensor y =
        venla::transpose(
            x
        );

    if (y.shape() != venla::Shape({3, 2})) {
        throw std::runtime_error(
            "transpose produced wrong shape"
        );
    }

    venla::Tensor loss =
        venla::sum(
            y
        );

    loss.backward();

    if (!x.has_grad()) {
        throw std::runtime_error(
            "x does not have gradient"
        );
    }

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < 6;
         ++i) {

        expect_close(
            grad[i],
            1.0f
        );
    }
}

void test_3d_transpose_backward() {

    venla::Tensor x =
        venla::Tensor::ones(
            {2, 3, 4},
            venla::DType::Float32,
            venla::Device::cpu()
        );

    x.requires_grad_(true);

    venla::Tensor y =
        venla::transpose(
            x,
            0,
            2
        );

    if (y.shape() !=
        venla::Shape({4, 3, 2})) {

        throw std::runtime_error(
            "3D transpose produced wrong shape"
        );
    }

    venla::Tensor loss =
        venla::sum(
            y
        );

    loss.backward();

    if (!x.has_grad()) {
        throw std::runtime_error(
            "3D transpose gradient missing"
        );
    }

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        expect_close(
            grad[i],
            1.0f
        );
    }
}

void test_transpose_chain() {

    venla::Tensor x =
        venla::Tensor::ones(
            {2, 3},
            venla::DType::Float32,
            venla::Device::cpu()
        );

    x.requires_grad_(true);

    venla::Tensor y =
        venla::transpose(
            x
        );

    venla::Tensor z =
        venla::transpose(
            y
        );

    venla::Tensor loss =
        venla::sum(
            z
        );

    loss.backward();

    if (!x.has_grad()) {
        throw std::runtime_error(
            "transpose chain gradient missing"
        );
    }

    const float* grad =
        x.grad().data_as<float>();

    for (std::size_t i = 0;
         i < x.numel();
         ++i) {

        expect_close(
            grad[i],
            1.0f
        );
    }
}

} // namespace

int main() {

    try {

        test_2d_transpose_backward();
        test_3d_transpose_backward();
        test_transpose_chain();

        std::cout
            << "TRANSPOSE AUTOGRAD TEST PASSED"
            << std::endl;

        return 0;

    }
    catch (const std::exception& error) {

        std::cerr
            << "TRANSPOSE AUTOGRAD TEST FAILED: "
            << error.what()
            << std::endl;

        return 1;
    }
}
