#include "venla/nn/cross_entropy_loss.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace venla;

namespace {

// ============================================================
// FLOAT32 LOGITS
// ============================================================

Tensor make_float32(
    const std::vector<std::size_t>& shape,
    const std::vector<float>& values
) {
    Tensor tensor =
        Tensor::empty(
            Shape(shape),
            DType::Float32,
            Device::cpu()
        );

    assert(
        tensor.numel() ==
        values.size()
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

// ============================================================
// INT32 TARGETS
// ============================================================

Tensor make_int32(
    const std::vector<std::size_t>& shape,
    const std::vector<std::int32_t>& values
) {
    Tensor tensor =
        Tensor::empty(
            Shape(shape),
            DType::Int32,
            Device::cpu()
        );

    assert(
        tensor.numel() ==
        values.size()
    );

    std::int32_t* data =
        tensor.data_as<std::int32_t>();

    for (std::size_t i = 0;
         i < values.size();
         ++i) {

        data[i] =
            values[i];
    }

    return tensor;
}

// ============================================================
// INT64 TARGETS
// ============================================================

Tensor make_int64(
    const std::vector<std::size_t>& shape,
    const std::vector<std::int64_t>& values
) {
    Tensor tensor =
        Tensor::empty(
            Shape(shape),
            DType::Int64,
            Device::cpu()
        );

    assert(
        tensor.numel() ==
        values.size()
    );

    std::int64_t* data =
        tensor.data_as<std::int64_t>();

    for (std::size_t i = 0;
         i < values.size();
         ++i) {

        data[i] =
            values[i];
    }

    return tensor;
}

// ============================================================
// APPROX
// ============================================================

void assert_close(
    float actual,
    float expected,
    float tolerance = 1e-4f
) {
    assert(
        std::fabs(
            actual - expected
        ) <= tolerance
    );
}

// ============================================================
// 1D / 2D
//
// logits:
//
//     [seq, vocab]
//
// targets:
//
//     [seq]
// ============================================================

void test_2d_forward() {

    Tensor logits =
        make_float32(
            {2, 3},
            {
                2.0f, 1.0f, 0.0f,
                0.0f, 2.0f, 1.0f
            }
        );

    Tensor targets =
        make_int32(
            {2},
            {
                0,
                1
            }
        );

    CrossEntropyLoss loss;

    Tensor result =
        loss.forward(
            logits,
            targets
        );

    assert(
        result.dtype() ==
        DType::Float32
    );

    assert(
        result.ndim() == 0
    );

    assert(
        result.numel() == 1
    );

    assert(
        std::isfinite(
            result.data_as<float>()[0]
        )
    );

    // --------------------------------------------------------
    // Expected:
    //
    // row 1:
    //
    // -2 + log(exp(2)+exp(1)+exp(0))
    //
    // row 2:
    //
    // -2 + log(exp(0)+exp(2)+exp(1))
    //
    // They are identical.
    // --------------------------------------------------------

    const float expected =
        -2.0f +
        std::log(
            std::exp(2.0f) +
            std::exp(1.0f) +
            std::exp(0.0f)
        );

    assert_close(
        result.data_as<float>()[0],
        expected
    );
}

// ============================================================
// 3D
//
// logits:
//
//     [batch, seq, vocab]
//
// targets:
//
//     [batch, seq]
// ============================================================

void test_3d_forward() {

    Tensor logits =
        make_float32(
            {2, 2, 3},
            {
                2.0f, 1.0f, 0.0f,
                0.0f, 2.0f, 1.0f,

                1.0f, 2.0f, 0.0f,
                2.0f, 0.0f, 1.0f
            }
        );

    Tensor targets =
        make_int64(
            {2, 2},
            {
                0, 1,
                1, 0
            }
        );

    CrossEntropyLoss loss;

    Tensor result =
        loss.forward(
            logits,
            targets
        );

    assert(
        result.ndim() == 0
    );

    assert(
        std::isfinite(
            result.data_as<float>()[0]
        )
    );
}

// ============================================================
// INT64
// ============================================================

void test_int64_targets() {

    Tensor logits =
        make_float32(
            {3, 4},
            {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f
            }
        );

    Tensor targets =
        make_int64(
            {3},
            {
                0,
                1,
                2
            }
        );

    CrossEntropyLoss loss;

    Tensor result =
        loss.forward(
            logits,
            targets
        );

    assert(
        std::isfinite(
            result.data_as<float>()[0]
        )
    );
}

// ============================================================
// IGNORE INDEX
// ============================================================

void test_ignore_index() {

    Tensor logits =
        make_float32(
            {3, 3},
            {
                10.0f, 0.0f, 0.0f,

                0.0f, 10.0f, 0.0f,

                0.0f, 0.0f, 10.0f
            }
        );

    Tensor targets =
        make_int32(
            {3},
            {
                0,
                -100,
                2
            }
        );

    CrossEntropyLoss loss(
        -100,
        true
    );

    Tensor result =
        loss.forward(
            logits,
            targets
        );

    // Both valid rows have the correct class with
    // dominant logits, therefore loss should be very small.

    assert(
        result.data_as<float>()[0] < 0.001f
    );
}

// ============================================================
// ALL TOKENS IGNORED
// ============================================================

void test_all_ignored() {

    Tensor logits =
        make_float32(
            {2, 3},
            {
                100.0f, 0.0f, 0.0f,
                0.0f, 100.0f, 0.0f
            }
        );

    Tensor targets =
        make_int32(
            {2},
            {
                -100,
                -100
            }
        );

    CrossEntropyLoss loss(
        -100,
        true
    );

    Tensor result =
        loss.forward(
            logits,
            targets
        );

    assert_close(
        result.data_as<float>()[0],
        0.0f
    );
}

// ============================================================
// STABLE LARGE LOGITS
// ============================================================

void test_numerical_stability() {

    Tensor logits =
        make_float32(
            {2, 3},
            {
                10000.0f,
                9999.0f,
                9998.0f,

                -10000.0f,
                -9999.0f,
                -9998.0f
            }
        );

    Tensor targets =
        make_int32(
            {2},
            {
                0,
                2
            }
        );

    CrossEntropyLoss loss;

    Tensor result =
        loss.forward(
            logits,
            targets
        );

    assert(
        std::isfinite(
            result.data_as<float>()[0]
        )
    );
}

// ============================================================
// AUTOGRAD
// ============================================================

void test_autograd() {

    Tensor logits =
        make_float32(
            {1, 3},
            {
                2.0f,
                1.0f,
                0.0f
            }
        );

    logits.requires_grad_();

    Tensor targets =
        make_int32(
            {1},
            {
                0
            }
        );

    CrossEntropyLoss loss;

    Tensor result =
        loss.forward(
            logits,
            targets
        );

    assert(
        result.requires_grad()
    );

    assert(
        !result.is_leaf()
    );

    assert(
        result.grad_state() != nullptr
    );

    assert(
        result.grad_state()->grad_fn != nullptr
    );

    result.backward();

    assert(
        logits.has_grad()
    );

    const Tensor& gradient =
        logits.grad();

    assert(
        gradient.shape() ==
        logits.shape()
    );

    const float* g =
        gradient.data_as<float>();

    // softmax([2,1,0])
    //
    // e2 / (e2+e1+e0)
    // e1 / (...)
    // e0 / (...)
    //
    // target 0 => softmax[0] - 1

    const float e2 =
        std::exp(2.0f);

    const float e1 =
        std::exp(1.0f);

    const float e0 =
        std::exp(0.0f);

    const float denominator =
        e2 + e1 + e0;

    const float expected0 =
        e2 / denominator - 1.0f;

    const float expected1 =
        e1 / denominator;

    const float expected2 =
        e0 / denominator;

    assert_close(
        g[0],
        expected0
    );

    assert_close(
        g[1],
        expected1
    );

    assert_close(
        g[2],
        expected2
    );

    // Gradient of softmax cross entropy must sum to zero.

    const float total =
        g[0] +
        g[1] +
        g[2];

    assert_close(
        total,
        0.0f
    );
}

// ============================================================
// AUTOGRAD MEAN
// ============================================================

void test_autograd_mean_scaling() {

    Tensor logits =
        make_float32(
            {2, 2},
            {
                1.0f, 0.0f,
                0.0f, 1.0f
            }
        );

    logits.requires_grad_();

    Tensor targets =
        make_int32(
            {2},
            {
                0,
                1
            }
        );

    CrossEntropyLoss loss;

    Tensor result =
        loss.forward(
            logits,
            targets
        );

    result.backward();

    assert(
        logits.has_grad()
    );

    const float* g =
        logits.grad().data_as<float>();

    // Each row is divided by 2 because reduction=mean.

    const float e =
        std::exp(1.0f);

    const float probability =
        e / (e + 1.0f);

    const float expected =
        (probability - 1.0f) /
        2.0f;

    assert_close(
        g[0],
        expected
    );

    assert_close(
        g[3],
        expected
    );
}

// ============================================================
// INVALID TARGET
// ============================================================

void test_invalid_target() {

    Tensor logits =
        make_float32(
            {2, 3},
            {
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f
            }
        );

    Tensor targets =
        make_int32(
            {2},
            {
                0,
                3
            }
        );

    CrossEntropyLoss loss;

    bool failed =
        false;

    try {

        (void)loss.forward(
            logits,
            targets
        );

    }
    catch (const std::out_of_range&) {

        failed =
            true;
    }

    assert(failed);
}

// ============================================================
// INVALID SHAPE
// ============================================================

void test_invalid_shape() {

    Tensor logits =
        make_float32(
            {2, 3},
            {
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f
            }
        );

    Tensor targets =
        make_int32(
            {1, 2},
            {
                0,
                1
            }
        );

    CrossEntropyLoss loss;

    bool failed =
        false;

    try {

        (void)loss.forward(
            logits,
            targets
        );

    }
    catch (const std::runtime_error&) {

        failed =
            true;
    }

    assert(failed);
}

// ============================================================
// INVALID DTYPE
// ============================================================

void test_invalid_dtype() {

    Tensor logits =
        Tensor::zeros(
            {2, 3},
            DType::Int32,
            Device::cpu()
        );

    Tensor targets =
        make_int32(
            {2},
            {
                0,
                1
            }
        );

    CrossEntropyLoss loss;

    bool failed =
        false;

    try {

        (void)loss.forward(
            logits,
            targets
        );

    }
    catch (const std::runtime_error&) {

        failed =
            true;
    }

    assert(failed);
}

// ============================================================
// METADATA
// ============================================================

void test_metadata() {

    CrossEntropyLoss loss(
        -100,
        true
    );

    assert(
        loss.ignore_index() ==
        -100
    );

    assert(
        loss.reduction_mean()
    );

    CrossEntropyLoss sum_loss(
        -1,
        false
    );

    assert(
        sum_loss.ignore_index() ==
        -1
    );

    assert(
        !sum_loss.reduction_mean()
    );
}

} // namespace

// ============================================================
// MAIN
// ============================================================

int main() {

    test_2d_forward();

    test_3d_forward();

    test_int64_targets();

    test_ignore_index();

    test_all_ignored();

    test_numerical_stability();

    test_autograd();

    test_autograd_mean_scaling();

    test_invalid_target();

    test_invalid_shape();

    test_invalid_dtype();

    test_metadata();

    std::cout
        << "CrossEntropyLoss tests passed."
        << std::endl;

    return 0;
}
