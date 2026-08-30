#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cstddef>

#include "venla/nn/linear.hpp"

namespace {

void assert_close(
    float actual,
    float expected,
    float tolerance = 1e-5f
) {
    assert(
        std::fabs(actual - expected) <= tolerance
    );
}

void fill_sequential(
    venla::Tensor& tensor
) {
    float* data = tensor.data_as<float>();

    for (std::size_t i = 0;
         i < tensor.numel();
         ++i) {

        data[i] =
            static_cast<float>(i + 1);
    }
}

} // namespace

int main() {

    // ========================================================
    // LINEAR METADATA
    // ========================================================

    {
        venla::Linear layer(
            3,
            2
        );

        assert(layer.in_features() == 3);
        assert(layer.out_features() == 2);
        assert(layer.has_bias());

        assert(
            layer.weight().shape().to_string()
            == "[3, 2]"
        );

        assert(
            layer.bias().shape().to_string()
            == "[2]"
        );
    }

    // ========================================================
    // 1D
    // ========================================================

    {
        venla::Linear layer(
            3,
            2
        );

        float* w =
            layer.weight().data_as<float>();

        w[0] = 1;
        w[1] = 2;
        w[2] = 3;
        w[3] = 4;
        w[4] = 5;
        w[5] = 6;

        float* b =
            layer.bias().data_as<float>();

        b[0] = 10;
        b[1] = 20;

        auto x =
            venla::Tensor::zeros(
                {3}
            );

        float* xv =
            x.data_as<float>();

        xv[0] = 1;
        xv[1] = 2;
        xv[2] = 3;

        auto y =
            layer.forward(x);

        assert(
            y.shape().to_string()
            == "[2]"
        );

        assert_close(
            y.data_as<float>()[0],
            29.0f
        );

        assert_close(
            y.data_as<float>()[1],
            48.0f
        );
    }

    // ========================================================
    // 2D
    // ========================================================

    {
        venla::Linear layer(
            3,
            2
        );

        float* w =
            layer.weight().data_as<float>();

        w[0] = 1;
        w[1] = 2;
        w[2] = 3;
        w[3] = 4;
        w[4] = 5;
        w[5] = 6;

        float* b =
            layer.bias().data_as<float>();

        b[0] = 10;
        b[1] = 20;

        auto x =
            venla::Tensor::zeros(
                {2, 3}
            );

        float* xv =
            x.data_as<float>();

        xv[0] = 1;
        xv[1] = 2;
        xv[2] = 3;

        xv[3] = 4;
        xv[4] = 5;
        xv[5] = 6;

        auto y =
            layer.forward(x);

        assert(
            y.shape().to_string()
            == "[2, 2]"
        );

        assert_close(
            y.data_as<float>()[0],
            29.0f
        );

        assert_close(
            y.data_as<float>()[1],
            48.0f
        );

        assert_close(
            y.data_as<float>()[2],
            59.0f
        );

        assert_close(
            y.data_as<float>()[3],
            84.0f
        );
    }

    // ========================================================
    // 3D
    // ========================================================

    {
        venla::Linear layer(
            3,
            2
        );

        float* w =
            layer.weight().data_as<float>();

        w[0] = 1;
        w[1] = 2;
        w[2] = 3;
        w[3] = 4;
        w[4] = 5;
        w[5] = 6;

        float* b =
            layer.bias().data_as<float>();

        b[0] = 0;
        b[1] = 0;

        auto x =
            venla::Tensor::zeros(
                {2, 2, 3}
            );

        fill_sequential(x);

        auto y =
            layer.forward(x);

        assert(
            y.shape().to_string()
            == "[2, 2, 2]"
        );

        const float* values =
            y.data_as<float>();

        assert_close(values[0], 22.0f);
        assert_close(values[1], 28.0f);

        assert_close(values[2], 49.0f);
        assert_close(values[3], 64.0f);

        assert_close(values[4], 76.0f);
        assert_close(values[5], 100.0f);

        assert_close(values[6], 103.0f);
        assert_close(values[7], 136.0f);
    }

    // ========================================================
    // 4D
    // ========================================================

    {
        venla::Linear layer(
            3,
            2
        );

        float* w =
            layer.weight().data_as<float>();

        w[0] = 1;
        w[1] = 2;
        w[2] = 3;
        w[3] = 4;
        w[4] = 5;
        w[5] = 6;

        float* b =
            layer.bias().data_as<float>();

        b[0] = 0;
        b[1] = 0;

        auto x =
            venla::Tensor::ones(
                {2, 2, 2, 3}
            );

        auto y =
            layer.forward(x);

        assert(
            y.shape().to_string()
            == "[2, 2, 2, 2]"
        );

        assert(y.numel() == 16);

        const float* values =
            y.data_as<float>();

        for (std::size_t i = 0;
             i < y.numel();
             ++i) {

            assert_close(
                values[i],
                9.0f
            );
        }
    }

    // ========================================================
    // LINEAR WITHOUT BIAS
    // ========================================================

    {
        venla::Linear layer(
            2,
            2,
            false
        );

        assert(!layer.has_bias());

        float* w =
            layer.weight().data_as<float>();

        w[0] = 1;
        w[1] = 2;
        w[2] = 3;
        w[3] = 4;

        auto x =
            venla::Tensor::zeros(
                {2}
            );

        float* xv =
            x.data_as<float>();

        xv[0] = 5;
        xv[1] = 6;

        auto y =
            layer.forward(x);

        assert(
            y.shape().to_string()
            == "[2]"
        );

        assert_close(
            y.data_as<float>()[0],
            23.0f
        );

        assert_close(
            y.data_as<float>()[1],
            34.0f
        );
    }

    // ========================================================
    // INVALID INPUT SHAPE
    // ========================================================

    {
        venla::Linear layer(
            3,
            2
        );

        auto invalid =
            venla::Tensor::zeros(
                {4}
            );

        bool threw = false;

        try {
            layer.forward(invalid);
        }
        catch (const std::runtime_error&) {
            threw = true;
        }

        assert(threw);
    }

    // ========================================================
    // INVALID ZERO IN FEATURES
    // ========================================================

    {
        bool threw = false;

        try {
            venla::Linear layer(
                0,
                2
            );
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    // ========================================================
    // INVALID ZERO OUT FEATURES
    // ========================================================

    {
        bool threw = false;

        try {
            venla::Linear layer(
                3,
                0
            );
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    // ========================================================
    // SCALAR INPUT
    //
    // Linear membutuhkan input minimal 1D.
    // ========================================================

    {
        venla::Linear layer(
            3,
            2
        );

        auto scalar =
            venla::Tensor::zeros(
                venla::Shape{}
            );

        bool threw = false;

        try {
            layer.forward(scalar);
        }
        catch (const std::runtime_error&) {
            threw = true;
        }

        assert(threw);
    }

    // ========================================================
    // FLOAT32 ONLY
    //
    // Jika Tensor mendukung dtype lain, Linear harus menolak.
    // ========================================================

    {
        venla::Linear layer(
            3,
            2
        );

        auto x =
            venla::Tensor::zeros(
                {3},
                venla::DType::Float32,
                venla::Device::cpu()
            );

        auto y =
            layer.forward(x);

        assert(
            y.dtype()
            == venla::DType::Float32
        );
    }

    // ========================================================
    // FINAL
    // ========================================================

    std::cout
        << "VENLACPU linear tests passed\n";

    return 0;
}
