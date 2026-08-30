#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#include "venla/nn/sequential.hpp"

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

void set_identity_linear(
    venla::Linear& layer
) {
    assert(
        layer.in_features() ==
        layer.out_features()
    );

    const std::size_t features =
        layer.in_features();

    float* weight =
        layer.weight().data_as<float>();

    for (std::size_t in = 0;
         in < features;
         ++in) {

        for (std::size_t out = 0;
             out < features;
             ++out) {

            weight[
                in * features + out
            ] =
                in == out
                    ? 1.0f
                    : 0.0f;
        }
    }

    if (layer.has_bias()) {

        float* bias =
            layer.bias().data_as<float>();

        for (std::size_t i = 0;
             i < features;
             ++i) {

            bias[i] = 0.0f;
        }
    }
}

} // namespace

int main() {

    // ========================================================
    // EMPTY SEQUENTIAL
    // ========================================================

    {
        venla::Sequential model;

        assert(
            model.empty()
        );

        assert(
            model.size() == 0
        );
    }

    // ========================================================
    // SINGLE LINEAR
    // ========================================================

    {
        venla::Sequential model;

        model.add_linear(
            3,
            2
        );

        assert(
            !model.empty()
        );

        assert(
            model.size() == 1
        );

        assert(
            model.layer_type(0) ==
            venla::Sequential::LayerType::Linear
        );

        venla::Linear& layer =
            model.linear(0);

        assert(
            layer.in_features() == 3
        );

        assert(
            layer.out_features() == 2
        );
    }

    // ========================================================
    // LINEAR + RELU
    //
    // Input:
    //
    // [-1, 2, -3]
    //
    // Identity:
    //
    // [-1, 2, -3]
    //
    // ReLU:
    //
    // [0, 2, 0]
    // ========================================================

    {
        venla::Sequential model;

        model.add_linear(
            3,
            3,
            false
        );

        model.add_relu();

        assert(
            model.size() == 2
        );

        assert(
            model.layer_type(0) ==
            venla::Sequential::LayerType::Linear
        );

        assert(
            model.layer_type(1) ==
            venla::Sequential::LayerType::ReLU
        );

        set_identity_linear(
            model.linear(0)
        );

        auto input =
            venla::Tensor::zeros(
                {3}
            );

        float* values =
            input.data_as<float>();

        values[0] = -1.0f;
        values[1] = 2.0f;
        values[2] = -3.0f;

        auto output =
            model.forward(input);

        assert(
            output.shape().to_string()
            == "[3]"
        );

        const float* result =
            output.data_as<float>();

        assert_close(
            result[0],
            0.0f
        );

        assert_close(
            result[1],
            2.0f
        );

        assert_close(
            result[2],
            0.0f
        );
    }

    // ========================================================
    // LINEAR + SIGMOID
    //
    // Identity input:
    //
    // [0, 1, -1]
    //
    // sigmoid:
    //
    // [0.5, sigmoid(1), sigmoid(-1)]
    // ========================================================

    {
        venla::Sequential model;

        model.add_linear(
            3,
            3,
            false
        );

        model.add_sigmoid();

        set_identity_linear(
            model.linear(0)
        );

        auto input =
            venla::Tensor::zeros(
                {3}
            );

        float* values =
            input.data_as<float>();

        values[0] = 0.0f;
        values[1] = 1.0f;
        values[2] = -1.0f;

        auto output =
            model.forward(input);

        const float* result =
            output.data_as<float>();

        assert_close(
            result[0],
            0.5f
        );

        assert_close(
            result[1],
            1.0f /
            (1.0f + std::exp(-1.0f))
        );

        assert_close(
            result[2],
            1.0f /
            (1.0f + std::exp(1.0f))
        );
    }

    // ========================================================
    // LINEAR + TANH
    // ========================================================

    {
        venla::Sequential model;

        model.add_linear(
            2,
            2,
            false
        );

        model.add_tanh();

        set_identity_linear(
            model.linear(0)
        );

        auto input =
            venla::Tensor::zeros(
                {2}
            );

        float* values =
            input.data_as<float>();

        values[0] = 0.0f;
        values[1] = 1.0f;

        auto output =
            model.forward(input);

        const float* result =
            output.data_as<float>();

        assert_close(
            result[0],
            0.0f
        );

        assert_close(
            result[1],
            std::tanh(1.0f)
        );
    }

    // ========================================================
    // MULTI-LAYER NETWORK
    //
    // 3 -> 4 -> 2
    //
    // Linear
    // ReLU
    // Linear
    // ========================================================

    {
        venla::Sequential model;

        model
            .add_linear(3, 4, false)
            .add_relu()
            .add_linear(4, 2, false);

        assert(
            model.size() == 3
        );

        assert(
            model.layer_type(0) ==
            venla::Sequential::LayerType::Linear
        );

        assert(
            model.layer_type(1) ==
            venla::Sequential::LayerType::ReLU
        );

        assert(
            model.layer_type(2) ==
            venla::Sequential::LayerType::Linear
        );

        // First Linear:
        //
        // x [3]
        // ->
        // [x0, x1, x2, x0]
        //
        float* w1 =
            model.linear(0)
                .weight()
                .data_as<float>();

        w1[0] = 1.0f;
        w1[1] = 0.0f;
        w1[2] = 0.0f;
        w1[3] = 1.0f;

        w1[4] = 0.0f;
        w1[5] = 1.0f;
        w1[6] = 0.0f;
        w1[7] = 0.0f;

        w1[8] = 0.0f;
        w1[9] = 0.0f;
        w1[10] = 1.0f;
        w1[11] = 0.0f;

        // Second Linear:
        //
        // [a,b,c,d]
        //
        // ->
        //
        // [a+c, b+d]
        //
        float* w2 =
            model.linear(2)
                .weight()
                .data_as<float>();

        w2[0] = 1.0f;
        w2[1] = 0.0f;

        w2[2] = 0.0f;
        w2[3] = 1.0f;

        w2[4] = 1.0f;
        w2[5] = 0.0f;

        w2[6] = 0.0f;
        w2[7] = 1.0f;

        auto input =
            venla::Tensor::zeros(
                {3}
            );

        float* x =
            input.data_as<float>();

        x[0] = -1.0f;
        x[1] = 2.0f;
        x[2] = 3.0f;

        auto output =
            model.forward(input);

        assert(
            output.shape().to_string()
            == "[2]"
        );

        const float* y =
            output.data_as<float>();

        // First:
        //
        // [-1, 2, 3, -1]
        //
        // ReLU:
        //
        // [0, 2, 3, 0]
        //
        // Second:
        //
        // [3, 2]
        //

        assert_close(
            y[0],
            3.0f
        );

        assert_close(
            y[1],
            2.0f
        );
    }

    // ========================================================
    // 2D INPUT
    //
    // Sequential must preserve leading dimensions.
    //
    // [2,3]
    // ->
    // [2,2]
    // ========================================================

    {
        venla::Sequential model;

        model.add_linear(
            3,
            2,
            false
        );


        float* w =
            model.linear(0)
                .weight()
                .data_as<float>();

        w[0] = 1.0f;
        w[1] = 0.0f;

        w[2] = 0.0f;
        w[3] = 1.0f;

        w[4] = 1.0f;
        w[5] = 1.0f;

        auto input =
            venla::Tensor::zeros(
                {2, 3}
            );

        float* x =
            input.data_as<float>();

        x[0] = 1.0f;
        x[1] = 2.0f;
        x[2] = 3.0f;

        x[3] = 4.0f;
        x[4] = 5.0f;
        x[5] = 6.0f;

        auto output =
            model.forward(input);

        assert(
            output.shape().to_string()
            == "[2, 2]"
        );

        const float* y =
            output.data_as<float>();

        assert_close(
            y[0],
            4.0f
        );

        assert_close(
            y[1],
            5.0f
        );

        assert_close(
            y[2],
            10.0f
        );

        assert_close(
            y[3],
            11.0f
        );
    }

    // ========================================================
    // EMPTY MODEL FORWARD
    //
    // Should behave like identity.
    // ========================================================

    {
        venla::Sequential model;

        auto input =
            venla::Tensor::ones(
                {2, 3}
            );

        auto output =
            model.forward(input);

        assert(
            output.shape().to_string()
            == "[2, 3]"
        );

        assert(
            output.numel() == 6
        );

        const float* y =
            output.data_as<float>();

        for (std::size_t i = 0;
             i < output.numel();
             ++i) {

            assert_close(
                y[i],
                1.0f
            );
        }
    }

    // ========================================================
    // INVALID LAYER INDEX
    // ========================================================

    {
        venla::Sequential model;

        model.add_relu();

        bool threw = false;

        try {
            model.layer_type(5);
        }
        catch (const std::out_of_range&) {
            threw = true;
        }

        assert(threw);
    }

    // ========================================================
    // ACCESSING NON-LINEAR AS LINEAR
    // ========================================================

    {
        venla::Sequential model;

        model.add_relu();

        bool threw = false;

        try {
            model.linear(0);
        }
        catch (const std::runtime_error&) {
            threw = true;
        }

        assert(threw);
    }

    // ========================================================
    // CLEAR
    // ========================================================

    {
        venla::Sequential model;

        model
            .add_linear(3, 4)
            .add_relu()
            .add_linear(4, 2);

        assert(
            model.size() == 3
        );

        model.clear();

        assert(
            model.empty()
        );

        assert(
            model.size() == 0
        );
    }

    // ========================================================
    // FINAL
    // ========================================================

    std::cout
        << "VENLACPU sequential tests passed\n";

    return 0;
}

