#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "venla/tensor/manipulation.hpp"

int main() {

    // ========================================================
    // BASE FLOAT32 TENSOR
    // ========================================================

    venla::Tensor x =
        venla::Tensor::empty(
            {2, 3},
            venla::DType::Float32
        );

    {
        float* data =
            x.data_as<float>();

        for (std::size_t i = 0;
             i < x.numel();
             ++i) {

            data[i] =
                static_cast<float>(i + 1);
        }
    }

    // ========================================================
    // RESHAPE
    // [2,3] -> [3,2]
    // ========================================================

    {
        auto y =
            venla::reshape(
                x,
                {3, 2}
            );

        assert(y.shape() == venla::Shape({3, 2}));
        assert(y.numel() == 6);

        const float* data =
            y.data_as<float>();

        assert(data[0] == 1.0f);
        assert(data[1] == 2.0f);
        assert(data[2] == 3.0f);
        assert(data[3] == 4.0f);
        assert(data[4] == 5.0f);
        assert(data[5] == 6.0f);
    }

    // ========================================================
    // FLATTEN
    // ========================================================

    {
        auto y =
            venla::flatten(x);

        assert(y.shape() == venla::Shape({6}));
        assert(y.numel() == 6);

        const float* data =
            y.data_as<float>();

        assert(data[0] == 1.0f);
        assert(data[5] == 6.0f);
    }

    // ========================================================
    // FLATTEN RANGE
    // [2,3,4] -> flatten dims 1..2
    // [2,12]
    // ========================================================

    {
        auto input =
            venla::Tensor::ones(
                {2, 3, 4},
                venla::DType::Float32
            );

        auto y =
            venla::flatten(
                input,
                1,
                2
            );

        assert(
            y.shape() ==
            venla::Shape({2, 12})
        );
    }

    // ========================================================
    // UNSQUEEZE
    // [2,3] -> [1,2,3]
    // ========================================================

    {
        auto y =
            venla::unsqueeze(
                x,
                0
            );

        assert(
            y.shape() ==
            venla::Shape({1, 2, 3})
        );
    }

    // ========================================================
    // UNSQUEEZE MIDDLE
    // [2,3] -> [2,1,3]
    // ========================================================

    {
        auto y =
            venla::unsqueeze(
                x,
                1
            );

        assert(
            y.shape() ==
            venla::Shape({2, 1, 3})
        );
    }

    // ========================================================
    // SQUEEZE
    // [1,2,1,3] -> [2,3]
    // ========================================================

    {
        auto input =
            venla::Tensor::ones(
                {1, 2, 1, 3},
                venla::DType::Float32
            );

        auto y =
            venla::squeeze(input);

        assert(
            y.shape() ==
            venla::Shape({2, 3})
        );
    }

    // ========================================================
    // SQUEEZE ONE DIMENSION
    // ========================================================

    {
        auto input =
            venla::Tensor::ones(
                {1, 2, 3},
                venla::DType::Float32
            );

        auto y =
            venla::squeeze(
                input,
                0
            );

        assert(
            y.shape() ==
            venla::Shape({2, 3})
        );
    }

    // ========================================================
    // INDEX 2D
    // ========================================================

    {
        auto y =
            venla::index(
                x,
                {1, 2}
            );

        assert(y.ndim() == 0);
        assert(y.numel() == 1);

        assert(
            std::fabs(
                y.data_as<float>()[0] -
                6.0f
            ) < 1e-6f
        );
    }

    // ========================================================
    // INDEX 1D
    // ========================================================

    {
        auto y =
            venla::index(
                venla::flatten(x),
                3
            );

        assert(
            y.data_as<float>()[0] ==
            4.0f
        );
    }

    // ========================================================
    // SLICE
    //
    // [1,2,3,4,5,6]
    // -> [2,3,4]
    // ========================================================

    {
        auto input =
            venla::flatten(x);

        auto y =
            venla::slice(
                input,
                0,
                1,
                4
            );

        assert(
            y.shape() ==
            venla::Shape({3})
        );

        const float* data =
            y.data_as<float>();

        assert(data[0] == 2.0f);
        assert(data[1] == 3.0f);
        assert(data[2] == 4.0f);
    }

    // ========================================================
    // SLICE WITH STEP
    //
    // [1,2,3,4,5,6]
    // -> [1,3,5]
    // ========================================================

    {
        auto input =
            venla::flatten(x);

        auto y =
            venla::slice(
                input,
                0,
                0,
                6,
                2
            );

        assert(
            y.shape() ==
            venla::Shape({3})
        );

        const float* data =
            y.data_as<float>();

        assert(data[0] == 1.0f);
        assert(data[1] == 3.0f);
        assert(data[2] == 5.0f);
    }

    // ========================================================
    // CONCATENATE
    //
    // [2,2] + [2,1] -> [2,3]
    // ========================================================

    {
        auto a =
            venla::Tensor::ones(
                {2, 2},
                venla::DType::Float32
            );

        auto b =
            venla::Tensor::ones(
                {2, 1},
                venla::DType::Float32
            );

        auto y =
            venla::concatenate(
                {a, b},
                1
            );

        assert(
            y.shape() ==
            venla::Shape({2, 3})
        );

        assert(y.numel() == 6);
    }

    // ========================================================
    // STACK
    //
    // [2,3] + [2,3] -> [2,2,3]
    // ========================================================

    {
        auto a =
            venla::Tensor::ones(
                {2, 3},
                venla::DType::Float32
            );

        auto b =
            venla::Tensor::ones(
                {2, 3},
                venla::DType::Float32
            );

        auto y =
            venla::stack(
                {a, b},
                0
            );

        assert(
            y.shape() ==
            venla::Shape({2, 2, 3})
        );

        assert(y.numel() == 12);
    }

    // ========================================================
    // INTEGER TENSOR
    //
    // Ini penting untuk tokenizer.
    // ========================================================

    {
        auto tokens =
            venla::Tensor::empty(
                {5},
                venla::DType::Int32
            );

        std::int32_t* data =
            tokens.data_as<std::int32_t>();

        data[0] = 10;
        data[1] = 20;
        data[2] = 30;
        data[3] = 40;
        data[4] = 50;

        assert(
            tokens.dtype() ==
            venla::DType::Int32
        );

        auto reshaped =
            venla::reshape(
                tokens,
                {1, 5}
            );

        assert(
            reshaped.shape() ==
            venla::Shape({1, 5})
        );

        auto selected =
            venla::index(
                tokens,
                2
            );

        assert(
            selected.dtype() ==
            venla::DType::Int32
        );

        assert(
            selected.data_as<std::int32_t>()[0] ==
            30
        );

        auto sliced =
            venla::slice(
                tokens,
                0,
                1,
                4
            );

        const std::int32_t* slice_data =
            sliced.data_as<std::int32_t>();

        assert(slice_data[0] == 20);
        assert(slice_data[1] == 30);
        assert(slice_data[2] == 40);
    }

    std::cout
        << "VENLACPU tensor manipulation tests passed\n";

    return 0;
}
