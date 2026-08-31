#include "venla/nn/transformer_decoder.hpp"

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
                (static_cast<int>(i) % 17) -
                8
            ) *
            0.05f;
    }
}

// ============================================================
// TEST 1
// Metadata
// ============================================================

void test_metadata() {

    venla::TransformerDecoder
        decoder(
            8,
            2,
            32,
            4
        );

    expect(
        decoder.embed_dim() == 8,
        "embed_dim mismatch"
    );

    expect(
        decoder.num_heads() == 2,
        "num_heads mismatch"
    );

    expect(
        decoder.hidden_dim() == 32,
        "hidden_dim mismatch"
    );

    expect(
        decoder.num_layers() == 4,
        "num_layers mismatch"
    );

    expect(
        decoder.has_bias(),
        "bias should be enabled"
    );
}

// ============================================================
// TEST 2
// Correct number of layers
// ============================================================

void test_layer_count() {

    venla::TransformerDecoder
        decoder(
            8,
            2,
            16,
            6
        );

    expect(
        decoder.num_layers() == 6,
        "decoder layer count mismatch"
    );

    for (std::size_t i = 0;
         i < decoder.num_layers();
         ++i) {

        expect(
            decoder.layer(i).embed_dim() == 8,
            "layer embed_dim mismatch"
        );

        expect(
            decoder.layer(i).num_heads() == 2,
            "layer num_heads mismatch"
        );

        expect(
            decoder.layer(i).hidden_dim() == 16,
            "layer hidden_dim mismatch"
        );

        expect(
            decoder.layer(i).is_causal(),
            "decoder layer must be causal"
        );
    }
}

// ============================================================
// TEST 3
// 2D forward
// ============================================================

void test_forward_2d() {

    venla::TransformerDecoder
        decoder(
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
        decoder.forward(
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

    venla::TransformerDecoder
        decoder(
            8,
            2,
            32,
            3
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 5, 8}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        decoder.forward(
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
// Causal guarantee
// ============================================================

void test_causal_guarantee() {

    venla::TransformerDecoder
        decoder(
            8,
            2,
            16,
            5
        );

    for (std::size_t i = 0;
         i < decoder.num_layers();
         ++i) {

        expect(
            decoder.layer(i)
                .self_attention()
                .is_causal(),
            "all decoder layers must use causal attention"
        );
    }
}

// ============================================================
// TEST 6
// No bias
// ============================================================

void test_no_bias() {

    venla::TransformerDecoder
        decoder(
            8,
            2,
            16,
            3,
            false
        );

    expect(
        !decoder.has_bias(),
        "decoder bias should be disabled"
    );

    for (std::size_t i = 0;
         i < decoder.num_layers();
         ++i) {

        expect(
            !decoder.layer(i)
                .has_bias(),
            "decoder layer bias should be disabled"
        );

        expect(
            !decoder.layer(i)
                .self_attention()
                .has_bias(),
            "attention bias should be disabled"
        );

        expect(
            !decoder.layer(i)
                .feed_forward()
                .has_bias(),
            "FFN bias should be disabled"
        );
    }
}

// ============================================================
// TEST 7
// Layer access
//
// TransformerDecoderLayer does not expose set_causal()
// directly. Causal state is controlled through its
// self_attention() module.
//
// ============================================================

void test_layer_access() {

    venla::TransformerDecoder
        decoder(
            4,
            2,
            8,
            2
        );

    decoder.layer(0)
        .self_attention()
        .set_causal(false);

    expect(
        !decoder.layer(0).is_causal(),
        "mutable self-attention causal access failed"
    );

    expect(
        decoder.layer(1).is_causal(),
        "layer 1 should remain causal"
    );

    decoder.layer(0)
        .self_attention()
        .set_causal(true);

    expect(
        decoder.layer(0).is_causal(),
        "causal state could not be restored"
    );
}

// ============================================================
// TEST 8
// Invalid layer index
// ============================================================

void test_invalid_layer_index() {

    venla::TransformerDecoder
        decoder(
            8,
            2,
            16,
            2
        );

    bool threw =
        false;

    try {

        decoder.layer(2);

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "out-of-range layer access must throw"
    );

    threw = false;

    try {

        decoder.layer(100);

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "far out-of-range layer access must throw"
    );
}

// ============================================================
// TEST 9
// Constructor validation
// ============================================================

void test_constructor_validation() {

    bool threw =
        false;

    try {

        venla::TransformerDecoder
            decoder(
                0,
                2,
                16,
                2
            );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "zero embed_dim must throw"
    );

    threw = false;

    try {

        venla::TransformerDecoder
            decoder(
                8,
                0,
                16,
                2
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

        venla::TransformerDecoder
            decoder(
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
        "non-divisible heads must throw"
    );

    threw = false;

    try {

        venla::TransformerDecoder
            decoder(
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
}

// ============================================================
// TEST 10
// Forward validation
// ============================================================

void test_forward_validation() {

    venla::TransformerDecoder
        decoder(
            8,
            2,
            16,
            2
        );

    bool threw =
        false;

    try {

        venla::Tensor input =
            venla::Tensor::zeros(
                {8}
            );

        decoder.forward(
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

        decoder.forward(
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
                {4, 7}
            );

        decoder.forward(
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
// TEST 11
// Input autograd through multiple layers
// ============================================================

void test_input_autograd() {

    venla::TransformerDecoder
        decoder(
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
        decoder.forward(
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
// TEST 12
// Parameter gradients across stack
// ============================================================

void test_parameter_gradients() {

    venla::TransformerDecoder
        decoder(
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
        decoder.forward(
            input
        );

    venla::Tensor loss =
        venla::sum(
            output
        );

    loss.backward();

    for (std::size_t i = 0;
         i < decoder.num_layers();
         ++i) {

        const auto& layer =
            decoder.layer(i);

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
}

// ============================================================
// TEST 13
// Deeper stack
// ============================================================

void test_deeper_stack() {

    venla::TransformerDecoder
        decoder(
            8,
            2,
            32,
            8
        );

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 12, 8}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        decoder.forward(
            input
        );

    expect(
        output.shape() ==
            venla::Shape({2, 12, 8}),
        "deep stack output shape mismatch"
    );

    const float* data =
        output.data_as<float>();

    for (std::size_t i = 0;
         i < output.numel();
         ++i) {

        expect(
            std::isfinite(data[i]),
            "deep stack output contains non-finite value"
        );
    }
}

} // namespace

int main() {

    try {

        test_metadata();

        test_layer_count();

        test_forward_2d();

        test_forward_3d();

        test_causal_guarantee();

        test_no_bias();

        test_layer_access();

        test_invalid_layer_index();

        test_constructor_validation();

        test_forward_validation();

        test_input_autograd();

        test_parameter_gradients();

        test_deeper_stack();

        std::cout
            << "VENLACPU TransformerDecoder Test: PASS"
            << std::endl;

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "VENLACPU TransformerDecoder Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
