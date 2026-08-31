#include "venla/nn/transformer_encoder.hpp"

#include "venla/math/operations.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(
            message
        );
    }
}

void fill_tensor(
    venla::Tensor& tensor
) {
    float* data =
        tensor.data_as<float>();

    for (std::size_t i = 0;
         i < tensor.numel();
         ++i) {

        data[i] =
            static_cast<float>(
                (static_cast<int>(i) % 11) -
                5
            ) *
            0.1f;
    }
}

// ============================================================
// TEST 1
// Metadata
// ============================================================

void test_metadata() {

    venla::TransformerEncoder
        encoder(
            8,
            2,
            32,
            4
        );

    expect(
        encoder.embed_dim() == 8,
        "embed_dim mismatch"
    );

    expect(
        encoder.num_heads() == 2,
        "num_heads mismatch"
    );

    expect(
        encoder.hidden_dim() == 32,
        "hidden_dim mismatch"
    );

    expect(
        encoder.num_layers() == 4,
        "num_layers mismatch"
    );

    expect(
        encoder.has_bias(),
        "bias should be enabled"
    );

    expect(
        !encoder.is_causal(),
        "encoder should not be causal"
    );
}

// ============================================================
// TEST 2
// Layer access
// ============================================================

void test_layer_access() {

    venla::TransformerEncoder
        encoder(
            8,
            2,
            32,
            3
        );

    for (std::size_t i = 0;
         i < encoder.num_layers();
         ++i) {

        const venla::TransformerEncoderLayer&
            layer =
                encoder.layer(i);

        expect(
            layer.embed_dim() == 8,
            "layer embed_dim mismatch"
        );

        expect(
            layer.num_heads() == 2,
            "layer num_heads mismatch"
        );

        expect(
            layer.hidden_dim() == 32,
            "layer hidden_dim mismatch"
        );
    }

    bool threw =
        false;

    try {

        encoder.layer(3);

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "out-of-range layer access must throw"
    );
}

// ============================================================
// TEST 3
// 2D forward
// ============================================================

void test_forward_2d() {

    venla::TransformerEncoder
        encoder(
            8,
            2,
            32,
            3
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {5, 8}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        encoder.forward(
            input
        );

    expect(
        output.shape() ==
            venla::Shape({5, 8}),
        "2D output shape mismatch"
    );

    expect(
        output.numel() == 40,
        "2D output numel mismatch"
    );

    const float* data =
        output.data_as<float>();

    for (std::size_t i = 0;
         i < output.numel();
         ++i) {

        expect(
            std::isfinite(data[i]),
            "2D output contains non-finite value"
        );
    }
}

// ============================================================
// TEST 4
// 3D forward
// ============================================================

void test_forward_3d() {

    venla::TransformerEncoder
        encoder(
            8,
            2,
            32,
            4
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 5, 8}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        encoder.forward(
            input
        );

    expect(
        output.shape() ==
            venla::Shape({2, 5, 8}),
        "3D output shape mismatch"
    );

    expect(
        output.numel() == 80,
        "3D output numel mismatch"
    );

    const float* data =
        output.data_as<float>();

    for (std::size_t i = 0;
         i < output.numel();
         ++i) {

        expect(
            std::isfinite(data[i]),
            "3D output contains non-finite value"
        );
    }
}

// ============================================================
// TEST 5
// Different layer depths
// ============================================================

void test_multiple_depths() {

    venla::Tensor input =
        venla::Tensor::zeros(
            {4, 8}
        );

    fill_tensor(
        input
    );

    for (std::size_t depth = 1;
         depth <= 5;
         ++depth) {

        venla::TransformerEncoder
            encoder(
                8,
                2,
                16,
                depth
            );

        expect(
            encoder.num_layers() == depth,
            "depth metadata mismatch"
        );

        venla::Tensor output =
            encoder.forward(
                input
            );

        expect(
            output.shape() ==
                venla::Shape({4, 8}),
            "depth output shape mismatch"
        );

        const float* data =
            output.data_as<float>();

        for (std::size_t i = 0;
             i < output.numel();
             ++i) {

            expect(
                std::isfinite(data[i]),
                "depth output contains non-finite value"
            );
        }
    }
}

// ============================================================
// TEST 6
// Causal propagation
// ============================================================

void test_causal() {

    venla::TransformerEncoder
        encoder(
            8,
            2,
            16,
            3,
            true,
            true
        );

    expect(
        encoder.is_causal(),
        "encoder should be causal"
    );

    for (std::size_t i = 0;
         i < encoder.num_layers();
         ++i) {

        expect(
            encoder.layer(i).is_causal(),
            "causal mode was not propagated"
        );
    }

    encoder.set_causal(
        false
    );

    expect(
        !encoder.is_causal(),
        "encoder causal flag should be false"
    );

    for (std::size_t i = 0;
         i < encoder.num_layers();
         ++i) {

        expect(
            !encoder.layer(i).is_causal(),
            "causal disable was not propagated"
        );
    }

    encoder.set_causal(
        true
    );

    for (std::size_t i = 0;
         i < encoder.num_layers();
         ++i) {

        expect(
            encoder.layer(i).is_causal(),
            "causal re-enable was not propagated"
        );
    }
}

// ============================================================
// TEST 7
// No bias
// ============================================================

void test_no_bias() {

    venla::TransformerEncoder
        encoder(
            8,
            2,
            16,
            2,
            false
        );

    expect(
        !encoder.has_bias(),
        "encoder bias should be disabled"
    );

    for (std::size_t i = 0;
         i < encoder.num_layers();
         ++i) {

        expect(
            !encoder.layer(i).has_bias(),
            "layer bias should be disabled"
        );
    }

    venla::Tensor input =
        venla::Tensor::zeros(
            {3, 8}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        encoder.forward(
            input
        );

    expect(
        output.shape() ==
            venla::Shape({3, 8}),
        "no-bias output shape mismatch"
    );
}

// ============================================================
// TEST 8
// Validation
// ============================================================

void test_validation() {

    bool threw =
        false;

    try {

        venla::TransformerEncoder
            encoder(
                8,
                2,
                16,
                0
            );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "zero layers must throw"
    );

    threw = false;

    try {

        venla::TransformerEncoder
            encoder(
                8,
                3,
                16,
                2
            );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "non-divisible embed/head configuration must throw"
    );

    venla::TransformerEncoder
        encoder(
            8,
            2,
            16,
            2
        );

    threw = false;

    try {

        venla::Tensor input =
            venla::Tensor::zeros(
                {8}
            );

        encoder.forward(
            input
        );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "1D input must throw"
    );

    threw = false;

    try {

        venla::Tensor input =
            venla::Tensor::zeros(
                {2, 4, 8, 1}
            );

        encoder.forward(
            input
        );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "4D input must throw"
    );

    threw = false;

    try {

        venla::Tensor input =
            venla::Tensor::zeros(
                {3, 7}
            );

        encoder.forward(
            input
        );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "wrong feature dimension must throw"
    );
}

// ============================================================
// TEST 9
// Input autograd
//
// loss = sum(output)
//
// The gradient must pass through ALL encoder layers.
// ============================================================

void test_input_autograd() {

    venla::TransformerEncoder
        encoder(
            4,
            2,
            8,
            3
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 4}
        );

    fill_tensor(
        input
    );

    input.requires_grad_(
        true
    );

    venla::Tensor output =
        encoder.forward(
            input
        );

    venla::Tensor loss =
        venla::sum(
            output
        );

    loss.backward();

    expect(
        input.has_grad(),
        "input gradient missing"
    );

    const float* gradient =
        input.grad()
            .data_as<float>();

    for (std::size_t i = 0;
         i < input.numel();
         ++i) {

        expect(
            std::isfinite(
                gradient[i]
            ),
            "input gradient is non-finite"
        );
    }
}

// ============================================================
// TEST 10
// First layer parameter autograd
// ============================================================

void test_first_layer_autograd() {

    venla::TransformerEncoder
        encoder(
            4,
            2,
            8,
            3
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 4}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        encoder.forward(
            input
        );

    venla::Tensor loss =
        venla::sum(
            output
        );

    loss.backward();

    const auto&
        first =
            encoder.layer(0);

    expect(
        first.self_attention()
            .q_weight()
            .has_grad(),
        "first layer Q weight gradient missing"
    );

    expect(
        first.feed_forward()
            .input_weight()
            .has_grad(),
        "first layer FFN gradient missing"
    );

    expect(
        first.norm1()
            .weight()
            .has_grad(),
        "first layer norm gradient missing"
    );
}

// ============================================================
// TEST 11
// Last layer parameter autograd
// ============================================================

void test_last_layer_autograd() {

    venla::TransformerEncoder
        encoder(
            4,
            2,
            8,
            3
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 4}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        encoder.forward(
            input
        );

    venla::Tensor loss =
        venla::sum(
            output
        );

    loss.backward();

    const auto&
        last =
            encoder.layer(
                encoder.num_layers() - 1
            );

    expect(
        last.self_attention()
            .q_weight()
            .has_grad(),
        "last layer Q weight gradient missing"
    );

    expect(
        last.feed_forward()
            .output_weight()
            .has_grad(),
        "last layer FFN output gradient missing"
    );

    expect(
        last.norm2()
            .weight()
            .has_grad(),
        "last layer norm gradient missing"
    );
}

// ============================================================
// TEST 12
// Deep stack
//
// Make sure a larger stack can execute.
// ============================================================

void test_deep_stack() {

    venla::TransformerEncoder
        encoder(
            8,
            2,
            32,
            8
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 6, 8}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        encoder.forward(
            input
        );

    expect(
        output.shape() ==
            venla::Shape({2, 6, 8}),
        "deep stack output shape mismatch"
    );

    const float* data =
        output.data_as<float>();

    for (std::size_t i = 0;
         i < output.numel();
         ++i) {

        expect(
            std::isfinite(data[i]),
            "deep stack produced non-finite value"
        );
    }
}

} // namespace

int main() {

    try {

        test_metadata();

        test_layer_access();

        test_forward_2d();

        test_forward_3d();

        test_multiple_depths();

        test_causal();

        test_no_bias();

        test_validation();

        test_input_autograd();

        test_first_layer_autograd();

        test_last_layer_autograd();

        test_deep_stack();

        std::cout
            << "VENLACPU TransformerEncoder Test: PASS"
            << std::endl;

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "VENLACPU TransformerEncoder Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
