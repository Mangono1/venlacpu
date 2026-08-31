#include "venla/nn/transformer_encoder_layer.hpp"

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
                (static_cast<int>(i) % 11) - 5
            ) *
            0.1f;
    }
}

// ============================================================
// TEST 1
// Metadata
// ============================================================

void test_metadata() {

    venla::TransformerEncoderLayer
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
        layer.self_attention().embed_dim() == 8,
        "attention embed_dim mismatch"
    );

    expect(
        layer.self_attention().num_heads() == 2,
        "attention num_heads mismatch"
    );

    expect(
        layer.feed_forward().embed_dim() == 8,
        "feed forward embed_dim mismatch"
    );

    expect(
        layer.feed_forward().hidden_dim() == 32,
        "feed forward hidden_dim mismatch"
    );

    expect(
        layer.norm1().normalized_shape() == 8,
        "norm1 normalized shape mismatch"
    );

    expect(
        layer.norm2().normalized_shape() == 8,
        "norm2 normalized shape mismatch"
    );
}

// ============================================================
// TEST 2
// No bias
// ============================================================

void test_no_bias() {

    venla::TransformerEncoderLayer
        layer(
            8,
            2,
            32,
            false
        );

    expect(
        !layer.has_bias(),
        "bias should be disabled"
    );

    expect(
        !layer.self_attention().has_bias(),
        "attention bias should be disabled"
    );

    expect(
        !layer.feed_forward().has_bias(),
        "feed forward bias should be disabled"
    );
}

// ============================================================
// TEST 3
// 2D forward
// ============================================================

void test_forward_2d() {

    venla::TransformerEncoderLayer
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
        output.dtype() ==
            venla::DType::Float32,
        "2D output dtype mismatch"
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

    venla::TransformerEncoderLayer
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
// Causal mode
// ============================================================

void test_causal() {

    venla::TransformerEncoderLayer
        layer(
            8,
            2,
            32,
            true,
            false
        );

    expect(
        !layer.is_causal(),
        "layer should initially be non-causal"
    );

    layer.set_causal(
        true
    );

    expect(
        layer.is_causal(),
        "layer should become causal"
    );

    layer.set_causal(
        false
    );

    expect(
        !layer.is_causal(),
        "layer should become non-causal"
    );
}

// ============================================================
// TEST 6
// Validation
// ============================================================

void test_validation() {

    venla::TransformerEncoderLayer
        layer(
            8,
            2,
            32
        );

    bool threw =
        false;

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
                {2, 5, 8, 1}
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
                {5, 7}
            );

        layer.forward(
            input
        );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "wrong embedding dimension must throw"
    );
}

// ============================================================
// TEST 7
// Input autograd
//
// Loss = sum(output)
//
// The complete chain must propagate:
//
// output
//   ↓
// LayerNorm
//   ↓
// residual add
//   ↓
// FeedForward + residual
//   ↓
// LayerNorm
//   ↓
// attention
//   ↓
// input
// ============================================================

void test_input_autograd() {

    venla::TransformerEncoderLayer
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

    const float* grad =
        input.grad()
            .data_as<float>();

    for (std::size_t i = 0;
         i < input.numel();
         ++i) {

        expect(
            std::isfinite(grad[i]),
            "input gradient is non-finite"
        );
    }
}

// ============================================================
// TEST 8
// Attention parameters receive gradient
// ============================================================

void test_attention_gradients() {

    venla::TransformerEncoderLayer
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
        "Q weight gradient missing"
    );

    expect(
        layer.self_attention()
            .k_weight()
            .has_grad(),
        "K weight gradient missing"
    );

    expect(
        layer.self_attention()
            .v_weight()
            .has_grad(),
        "V weight gradient missing"
    );

    expect(
        layer.self_attention()
            .out_weight()
            .has_grad(),
        "output projection gradient missing"
    );
}

// ============================================================
// TEST 9
// FeedForward parameters receive gradient
// ============================================================

void test_feed_forward_gradients() {

    venla::TransformerEncoderLayer
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
        "FF input weight gradient missing"
    );

    expect(
        layer.feed_forward()
            .output_weight()
            .has_grad(),
        "FF output weight gradient missing"
    );
}

// ============================================================
// TEST 10
// LayerNorm parameters receive gradient
// ============================================================

void test_layer_norm_gradients() {

    venla::TransformerEncoderLayer
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
        layer.norm1()
            .weight()
            .has_grad(),
        "norm1 weight gradient missing"
    );

    expect(
        layer.norm1()
            .bias()
            .has_grad(),
        "norm1 bias gradient missing"
    );

    expect(
        layer.norm2()
            .weight()
            .has_grad(),
        "norm2 weight gradient missing"
    );

    expect(
        layer.norm2()
            .bias()
            .has_grad(),
        "norm2 bias gradient missing"
    );
}

// ============================================================
// MAIN
// ============================================================

} // namespace

int main() {

    try {

        test_metadata();

        test_no_bias();

        test_forward_2d();

        test_forward_3d();

        test_causal();

        test_validation();

        test_input_autograd();

        test_attention_gradients();

        test_feed_forward_gradients();

        test_layer_norm_gradients();

        std::cout
            << "VENLACPU TransformerEncoderLayer Test: PASS"
            << std::endl;

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "VENLACPU TransformerEncoderLayer Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
