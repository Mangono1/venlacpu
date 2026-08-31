#include "venla/nn/transformer_decoder_layer.hpp"

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
                (static_cast<int>(i) % 13) -
                6
            ) *
            0.1f;
    }
}

// ============================================================
// TEST 1
// Metadata
// ============================================================

void test_metadata() {

    venla::TransformerDecoderLayer
        layer(
            8,
            2,
            32
        );

    expect(
        layer.embed_dim() == 8,
        "embed_dim mismatch"
    );

    expect(
        layer.num_heads() == 2,
        "num_heads mismatch"
    );

    expect(
        layer.hidden_dim() == 32,
        "hidden_dim mismatch"
    );

    expect(
        layer.has_bias(),
        "bias should be enabled"
    );

    expect(
        layer.is_causal(),
        "decoder layer must always be causal"
    );
}

// ============================================================
// TEST 2
// Module shapes
// ============================================================

void test_module_shapes() {

    venla::TransformerDecoderLayer
        layer(
            8,
            2,
            32
        );

    expect(
        layer.self_attention()
            .q_weight()
            .shape() ==
            venla::Shape({8, 8}),
        "Q weight shape mismatch"
    );

    expect(
        layer.self_attention()
            .k_weight()
            .shape() ==
            venla::Shape({8, 8}),
        "K weight shape mismatch"
    );

    expect(
        layer.self_attention()
            .v_weight()
            .shape() ==
            venla::Shape({8, 8}),
        "V weight shape mismatch"
    );

    expect(
        layer.feed_forward()
            .input_weight()
            .shape() ==
            venla::Shape({8, 32}),
        "FFN input weight shape mismatch"
    );

    expect(
        layer.feed_forward()
            .output_weight()
            .shape() ==
            venla::Shape({32, 8}),
        "FFN output weight shape mismatch"
    );

    expect(
        layer.norm1()
            .weight()
            .shape() ==
            venla::Shape({8}),
        "norm1 weight shape mismatch"
    );

    expect(
        layer.norm2()
            .weight()
            .shape() ==
            venla::Shape({8}),
        "norm2 weight shape mismatch"
    );
}

// ============================================================
// TEST 3
// 2D forward
// ============================================================

void test_forward_2d() {

    venla::TransformerDecoderLayer
        layer(
            8,
            2,
            32
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {5, 8}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        layer.forward(
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

    venla::TransformerDecoderLayer
        layer(
            8,
            2,
            32
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 5, 8}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        layer.forward(
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
// Causal property
// ============================================================

void test_causal_property() {

    venla::TransformerDecoderLayer
        layer(
            8,
            2,
            16
        );

    expect(
        layer.is_causal(),
        "decoder must use causal attention"
    );

    expect(
        layer.self_attention().is_causal(),
        "self attention must be causal"
    );
}

// ============================================================
// TEST 6
// No bias
// ============================================================

void test_no_bias() {

    venla::TransformerDecoderLayer
        layer(
            8,
            2,
            16,
            false
        );

    expect(
        !layer.has_bias(),
        "bias should be disabled"
    );

    expect(
        !layer.self_attention()
            .has_bias(),
        "attention bias should be disabled"
    );

    expect(
        !layer.feed_forward()
            .has_bias(),
        "FFN bias should be disabled"
    );
}

// ============================================================
// TEST 7
// Validation
// ============================================================

void test_validation() {

    bool threw =
        false;

    try {

        venla::TransformerDecoderLayer
            layer(
                8,
                0,
                16
            );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "zero heads must throw"
    );

    threw = false;

    try {

        venla::TransformerDecoderLayer
            layer(
                8,
                3,
                16
            );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "non-divisible heads must throw"
    );

    venla::TransformerDecoderLayer
        layer(
            8,
            2,
            16
        );

    threw = false;

    try {

        venla::Tensor input =
            venla::Tensor::zeros(
                {8}
            );

        layer.forward(
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

        layer.forward(
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

        layer.forward(
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
// TEST 8
// Input autograd
// ============================================================

void test_input_autograd() {

    venla::TransformerDecoderLayer
        layer(
            4,
            2,
            8
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
        layer.forward(
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
// TEST 9
// Attention parameter autograd
// ============================================================

void test_attention_autograd() {

    venla::TransformerDecoderLayer
        layer(
            4,
            2,
            8
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 4}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        layer.forward(
            input
        );

    venla::Tensor loss =
        venla::sum(
            output
        );

    loss.backward();

    expect(
        layer.self_attention()
            .q_weight()
            .has_grad(),
        "Q gradient missing"
    );

    expect(
        layer.self_attention()
            .k_weight()
            .has_grad(),
        "K gradient missing"
    );

    expect(
        layer.self_attention()
            .v_weight()
            .has_grad(),
        "V gradient missing"
    );

    expect(
        layer.self_attention()
            .out_weight()
            .has_grad(),
        "output projection gradient missing"
    );
}

// ============================================================
// TEST 10
// FFN and LayerNorm autograd
// ============================================================

void test_ffn_norm_autograd() {

    venla::TransformerDecoderLayer
        layer(
            4,
            2,
            8
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 4}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        layer.forward(
            input
        );

    venla::Tensor loss =
        venla::sum(
            output
        );

    loss.backward();

    expect(
        layer.feed_forward()
            .input_weight()
            .has_grad(),
        "FFN input weight gradient missing"
    );

    expect(
        layer.feed_forward()
            .output_weight()
            .has_grad(),
        "FFN output weight gradient missing"
    );

    expect(
        layer.norm1()
            .weight()
            .has_grad(),
        "norm1 gradient missing"
    );

    expect(
        layer.norm2()
            .weight()
            .has_grad(),
        "norm2 gradient missing"
    );
}

// ============================================================
// TEST 11
// Longer sequence
// ============================================================

void test_long_sequence() {

    venla::TransformerDecoderLayer
        layer(
            8,
            2,
            32
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 16, 8}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        layer.forward(
            input
        );

    expect(
        output.shape() ==
            venla::Shape({2, 16, 8}),
        "long sequence output shape mismatch"
    );

    const float* data =
        output.data_as<float>();

    for (std::size_t i = 0;
         i < output.numel();
         ++i) {

        expect(
            std::isfinite(data[i]),
            "long sequence output is non-finite"
        );
    }
}

} // namespace

int main() {

    try {

        test_metadata();

        test_module_shapes();

        test_forward_2d();

        test_forward_3d();

        test_causal_property();

        test_no_bias();

        test_validation();

        test_input_autograd();

        test_attention_autograd();

        test_ffn_norm_autograd();

        test_long_sequence();

        std::cout
            << "VENLACPU TransformerDecoderLayer Test: PASS"
            << std::endl;

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "VENLACPU TransformerDecoderLayer Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
