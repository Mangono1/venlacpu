#include "venla/nn/feed_forward.hpp"

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

void expect_close(
    float actual,
    float expected,
    float tolerance,
    const std::string& message
) {
    if (std::fabs(
            actual -
            expected
        ) > tolerance) {

        throw std::runtime_error(
            message +
            ": expected " +
            std::to_string(expected) +
            ", got " +
            std::to_string(actual)
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
                (static_cast<int>(i) % 7) -
                3
            ) *
            0.25f;
    }
}

// ============================================================
// TEST 1
// Constructor / metadata
// ============================================================

void test_metadata() {

    venla::FeedForward
        ff(8, 32);

    expect(
        ff.embed_dim() == 8,
        "embed_dim mismatch"
    );

    expect(
        ff.hidden_dim() == 32,
        "hidden_dim mismatch"
    );

    expect(
        ff.has_bias(),
        "bias should be enabled"
    );

    expect(
        ff.input_weight().shape() ==
            venla::Shape({8, 32}),
        "input weight shape mismatch"
    );

    expect(
        ff.input_bias().shape() ==
            venla::Shape({32}),
        "input bias shape mismatch"
    );

    expect(
        ff.output_weight().shape() ==
            venla::Shape({32, 8}),
        "output weight shape mismatch"
    );

    expect(
        ff.output_bias().shape() ==
            venla::Shape({8}),
        "output bias shape mismatch"
    );

    expect(
        ff.input_weight().requires_grad(),
        "input weight must require grad"
    );

    expect(
        ff.input_bias().requires_grad(),
        "input bias must require grad"
    );

    expect(
        ff.output_weight().requires_grad(),
        "output weight must require grad"
    );

    expect(
        ff.output_bias().requires_grad(),
        "output bias must require grad"
    );
}

// ============================================================
// TEST 2
// No bias
// ============================================================

void test_no_bias() {

    venla::FeedForward
        ff(8, 32, false);

    expect(
        !ff.has_bias(),
        "bias should be disabled"
    );

    expect(
        !ff.input_bias().requires_grad(),
        "input bias must not require grad"
    );

    expect(
        !ff.output_bias().requires_grad(),
        "output bias must not require grad"
    );
}

// ============================================================
// TEST 3
// 2D forward
// ============================================================

void test_forward_2d() {

    venla::FeedForward
        ff(4, 16);

    venla::Tensor input =
        venla::Tensor::zeros(
            {3, 4}
        );

    float* data =
        input.data_as<float>();

    for (std::size_t i = 0;
         i < input.numel();
         ++i) {

        data[i] =
            static_cast<float>(
                i
            ) *
            0.1f;
    }

    venla::Tensor output =
        ff.forward(
            input
        );

    expect(
        output.shape() ==
            venla::Shape({3, 4}),
        "2D output shape mismatch"
    );

    expect(
        output.dtype() ==
            venla::DType::Float32,
        "2D output dtype mismatch"
    );

    expect(
        output.numel() == 12,
        "2D output numel mismatch"
    );
}

// ============================================================
// TEST 4
// 3D forward
// ============================================================

void test_forward_3d() {

    venla::FeedForward
        ff(8, 32);

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 5, 8}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        ff.forward(
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
            "output contains non-finite value"
        );
    }
}

// ============================================================
// TEST 5
// GELU sanity through controlled weights
//
// Set:
//
// W1 = 1
// b1 = 0
// W2 = 1
// b2 = 0
//
// For scalar feature:
//
// output = GELU(x)
//
// ============================================================

void test_gelu_forward() {

    venla::FeedForward
        ff(1, 1);

    float* w1 =
        ff.input_weight()
            .data_as<float>();

    float* b1 =
        ff.input_bias()
            .data_as<float>();

    float* w2 =
        ff.output_weight()
            .data_as<float>();

    float* b2 =
        ff.output_bias()
            .data_as<float>();

    w1[0] = 1.0f;
    b1[0] = 0.0f;
    w2[0] = 1.0f;
    b2[0] = 0.0f;

    venla::Tensor input =
        venla::Tensor::zeros(
            {3, 1}
        );

    float* x =
        input.data_as<float>();

    x[0] = -1.0f;
    x[1] = 0.0f;
    x[2] = 1.0f;

    venla::Tensor output =
        ff.forward(
            input
        );

    const float* y =
        output.data_as<float>();

    expect_close(
        y[0],
        -0.15865525f,
        1e-5f,
        "GELU(-1)"
    );

    expect_close(
        y[1],
        0.0f,
        1e-6f,
        "GELU(0)"
    );

    expect_close(
        y[2],
        0.84134475f,
        1e-5f,
        "GELU(1)"
    );
}

// ============================================================
// TEST 6
// Deterministic initialization
// ============================================================

void test_deterministic_initialization() {

    venla::FeedForward
        a(8, 32);

    venla::FeedForward
        b(8, 32);

    const float* aw =
        a.input_weight()
            .data_as<float>();

    const float* bw =
        b.input_weight()
            .data_as<float>();

    for (std::size_t i = 0;
         i < a.input_weight().numel();
         ++i) {

        expect_close(
            aw[i],
            bw[i],
            0.0f,
            "input weight initialization"
        );
    }

    const float* aow =
        a.output_weight()
            .data_as<float>();

    const float* bow =
        b.output_weight()
            .data_as<float>();

    for (std::size_t i = 0;
         i < a.output_weight().numel();
         ++i) {

        expect_close(
            aow[i],
            bow[i],
            0.0f,
            "output weight initialization"
        );
    }
}

// ============================================================
// TEST 7
// Validation
// ============================================================

void test_validation() {

    venla::FeedForward
        ff(4, 16);

    bool threw =
        false;

    try {

        venla::Tensor input =
            venla::Tensor::zeros(
                {4}
            );

        ff.forward(
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
                {2, 3, 4, 5}
            );

        ff.forward(
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
                {2, 8}
            );

        ff.forward(
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
//
// Loss = sum(output)
//
// Verify input gradient exists and is finite.
// ============================================================

void test_input_autograd() {

    venla::FeedForward
        ff(4, 8);

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
        ff.forward(
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
// TEST 9
// Parameter autograd
// ============================================================

void test_parameter_autograd() {

    venla::FeedForward
        ff(4, 8);

    venla::Tensor input =
        venla::Tensor::zeros(
            {2, 4}
        );

    fill_tensor(
        input
    );

    venla::Tensor output =
        ff.forward(
            input
        );

    venla::Tensor loss =
        venla::sum(
            output
        );

    loss.backward();

    expect(
        ff.input_weight().has_grad(),
        "input weight gradient missing"
    );

    expect(
        ff.input_bias().has_grad(),
        "input bias gradient missing"
    );

    expect(
        ff.output_weight().has_grad(),
        "output weight gradient missing"
    );

    expect(
        ff.output_bias().has_grad(),
        "output bias gradient missing"
    );
}

// ============================================================
// TEST 10
// Reset parameters
// ============================================================

void test_reset_parameters() {

    venla::FeedForward
        ff(4, 8);

    ff.reset_parameters();

    const float* b1 =
        ff.input_bias()
            .data_as<float>();

    const float* b2 =
        ff.output_bias()
            .data_as<float>();

    for (std::size_t i = 0;
         i < ff.input_bias().numel();
         ++i) {

        expect_close(
            b1[i],
            0.0f,
            0.0f,
            "input bias reset"
        );
    }

    for (std::size_t i = 0;
         i < ff.output_bias().numel();
         ++i) {

        expect_close(
            b2[i],
            0.0f,
            0.0f,
            "output bias reset"
        );
    }
}

} // namespace

int main() {

    try {

        test_metadata();

        test_no_bias();

        test_forward_2d();

        test_forward_3d();

        test_gelu_forward();

        test_deterministic_initialization();

        test_validation();

        test_input_autograd();

        test_parameter_autograd();

        test_reset_parameters();

        std::cout
            << "VENLACPU FeedForward Test: PASS"
            << std::endl;

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "VENLACPU FeedForward Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
