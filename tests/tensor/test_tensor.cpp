#include <cassert>
#include <cmath>
#include <iostream>

#include "venla/tensor/tensor.hpp"

int main() {
    // 1D
    {
        auto x = venla::Tensor::zeros(
            {10},
            venla::DType::Float32
        );

        assert(x.ndim() == 1);
        assert(x.shape()[0] == 10);
        assert(x.numel() == 10);
        assert(x.nbytes() == 40);
        assert(x.is_contiguous());
    }

    // 2D matrix
    {
        auto x = venla::Tensor::zeros(
            {3, 4},
            venla::DType::Float32
        );

        assert(x.ndim() == 2);
        assert(x.shape()[0] == 3);
        assert(x.shape()[1] == 4);
        assert(x.numel() == 12);
        assert(x.nbytes() == 48);

        assert(x.stride()[0] == 4);
        assert(x.stride()[1] == 1);
        assert(x.is_contiguous());
    }

    // 3D tensor
    {
        auto x = venla::Tensor::zeros(
            {2, 3, 4},
            venla::DType::Float32
        );

        assert(x.ndim() == 3);
        assert(x.numel() == 24);

        assert(x.stride()[0] == 12);
        assert(x.stride()[1] == 4);
        assert(x.stride()[2] == 1);
    }

    // 4D tensor
    {
        auto x = venla::Tensor::zeros(
            {2, 3, 4, 5},
            venla::DType::Float32
        );

        assert(x.ndim() == 4);
        assert(x.numel() == 120);
        assert(x.nbytes() == 480);

        assert(x.stride()[0] == 60);
        assert(x.stride()[1] == 20);
        assert(x.stride()[2] == 5);
        assert(x.stride()[3] == 1);
    }

    // Matrix data
    {
        auto matrix = venla::Tensor::zeros(
            {2, 3},
            venla::DType::Float32
        );

        float* data = matrix.data_as<float>();

        for (std::size_t i = 0; i < matrix.numel(); ++i) {
            data[i] = static_cast<float>(i + 1);
        }

        assert(data[0] == 1.0f);
        assert(data[1] == 2.0f);
        assert(data[5] == 6.0f);
    }

    // Ones
    {
        auto x = venla::Tensor::ones(
            {2, 2},
            venla::DType::Float32
        );

        const float* data = x.data_as<float>();

        for (std::size_t i = 0; i < x.numel(); ++i) {
            assert(std::fabs(data[i] - 1.0f) < 1e-6f);
        }
    }

    std::cout << "VENLACPU tensor tests passed\n";

    return 0;
}
