#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "venla/nn/activation.hpp"

namespace {

void assert_close(
    float actual,
    float expected,
    float tolerance = 1e-5f
) {
    assert(
        std::fabs(actual - expected) <=
        tolerance
    );
}

venla::Tensor make_tensor(
    const std::vector<float>& values
) {
    venla::Tensor tensor =
        venla::Tensor::zeros(
            {values.size()}
        );

    float* data =
        tensor.data_as<float>();

    for (std::size_t i = 0;
         i < values.size();
         ++i) {

        data[i] =
            values[i];
    }

    return tensor;
}

} // namespace

int main() {

    // ========================================================
    // RELU 1D
    // ========================================================

    {
        auto x =
            make_tensor({
                -3.0f,
                -1.0f,
                0.0f,
                2.0f,
                5.0f
            });

        auto y =
            venla::relu(x);

        assert(
            y.shape().to_string()
            == "[5]"
        );

        const float* values =
            y.data_as<float>();

        assert_close(values[0], 0.0f);
        assert_close(values[1], 0.0f);
        assert_close(values[2], 0.0f);
        assert_close(values[3], 2.0f);
        assert_close(values[4], 5.0f);

        // Input must remain unchanged.
        const float* original =
            x.data_as<float>();

        assert_close(original[0], -3.0f);
        assert_close(original[3], 2.0f);
    }

    // ========================================================
    // RELU 2D
    // ========================================================

    {
        auto x =
            venla::Tensor::zeros(
                {2, 3}
            );

        float* data =
            x.data_as<float>();

        data[0] = -2.0f;
        data[1] = 1.0f;
        data[2] = -4.0f;
        data[3] = 3.0f;
        data[4] = 0.0f;
        data[5] = 7.0f;

        auto y =
            venla::relu(x);

        assert(
            y.shape().to_string()
            == "[2, 3]"
        );

        const float* values =
            y.data_as<float>();

        assert_close(values[0], 0.0f);
        assert_close(values[1], 1.0f);
        assert_close(values[2], 0.0f);
        assert_close(values[3], 3.0f);
        assert_close(values[4], 0.0f);
        assert_close(values[5], 7.0f);
    }

    // ========================================================
    // SIGMOID
    // ========================================================

    {
        auto x =
            make_tensor({
                -10.0f,
                -1.0f,
                0.0f,
                1.0f,
                10.0f
            });

        auto y =
            venla::sigmoid(x);

        const float* values =
            y.data_as<float>();

        assert_close(
            values[0],
            1.0f /
            (1.0f + std::exp(10.0f)),
            1e-5f
        );

        assert_close(
            values[1],
            1.0f /
            (1.0f + std::exp(1.0f)),
            1e-5f
        );

        assert_close(
            values[2],
            0.5f
        );

        assert_close(
            values[3],
            1.0f /
            (1.0f + std::exp(-1.0f)),
            1e-5f
        );

        assert_close(
            values[4],
            1.0f /
            (1.0f + std::exp(-10.0f)),
            1e-5f
        );
    }

    // ========================================================
    // SIGMOID NUMERICAL STABILITY
    // ========================================================

    {
        auto x =
            make_tensor({
                -1000.0f,
                1000.0f
            });

        auto y =
            venla::sigmoid(x);

        const float* values =
            y.data_as<float>();

        assert(
            std::isfinite(values[0])
        );

        assert(
            std::isfinite(values[1])
        );

        assert(
            values[0] >= 0.0f
        );

        assert(
            values[0] <= 1.0f
        );

        assert(
            values[1] >= 0.0f
        );

        assert(
            values[1] <= 1.0f
        );

        assert_close(
            values[0],
            0.0f
        );

        assert_close(
            values[1],
            1.0f
        );
    }

    // ========================================================
    // TANH
    // ========================================================

    {
        auto x =
            make_tensor({
                -2.0f,
                -1.0f,
                0.0f,
                1.0f,
                2.0f
            });

        auto y =
            venla::tanh(x);

        const float* values =
            y.data_as<float>();

        assert_close(
            values[0],
            std::tanh(-2.0f)
        );

        assert_close(
            values[1],
            std::tanh(-1.0f)
        );

        assert_close(
            values[2],
            0.0f
        );

        assert_close(
            values[3],
            std::tanh(1.0f)
        );

        assert_close(
            values[4],
            std::tanh(2.0f)
        );
    }

    // ========================================================
    // TANH 3D
    // ========================================================

    {
        auto x =
            venla::Tensor::zeros(
                {2, 2, 2}
            );

        float* data =
            x.data_as<float>();

        for (std::size_t i = 0;
             i < x.numel();
             ++i) {

            data[i] =
                static_cast<float>(
                    i
                ) - 4.0f;
        }

        auto y =
            venla::tanh(x);

        assert(
            y.shape().to_string()
            == "[2, 2, 2]"
        );

        assert(
            y.numel() == 8
        );

        const float* values =
            y.data_as<float>();

        for (std::size_t i = 0;
             i < y.numel();
             ++i) {

            assert_close(
                values[i],
                std::tanh(data[i])
            );
        }
    }

    // ========================================================
    // EMPTY / ZERO-ELEMENT TENSOR
    // ========================================================

    {
        auto x =
            venla::Tensor::zeros(
                {0}
            );

        auto r =
            venla::relu(x);

        auto s =
            venla::sigmoid(x);

        auto t =
            venla::tanh(x);

        assert(
            r.numel() == 0
        );

        assert(
            s.numel() == 0
        );

        assert(
            t.numel() == 0
        );
    }

    // ========================================================
    // OUTPUT MUST NOT ALIAS INPUT
    // ========================================================

    {
        auto x =
            make_tensor({
                -1.0f,
                2.0f
            });

        auto y =
            venla::relu(x);

        y.data_as<float>()[1] =
            100.0f;

        assert_close(
            x.data_as<float>()[1],
            2.0f
        );
    }

    // ========================================================
    // RESULT DTYPE
    // ========================================================

    {
        auto x =
            make_tensor({
                -1.0f,
                0.0f,
                1.0f
            });

        auto r =
            venla::relu(x);

        auto s =
            venla::sigmoid(x);

        auto t =
            venla::tanh(x);

        assert(
            r.dtype()
            == venla::DType::Float32
        );

        assert(
            s.dtype()
            == venla::DType::Float32
        );

        assert(
            t.dtype()
            == venla::DType::Float32
        );
    }

    std::cout
        << "VENLACPU activation tests passed\n";

    return 0;
}
