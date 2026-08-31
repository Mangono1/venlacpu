#include "venla/nn/kv_cache.hpp"

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
            actual - expected
        ) > tolerance) {

        throw std::runtime_error(
            message
        );
    }
}

void fill_tensor(
    venla::Tensor& tensor,
    float start
) {
    float* data =
        tensor.data_as<float>();

    for (std::size_t i = 0;
         i < tensor.numel();
         ++i) {

        data[i] =
            start +
            static_cast<float>(i);
    }
}

// ============================================================
// TEST 1
// Empty state
// ============================================================

void test_empty_state() {

    venla::KVCache cache;

    expect(
        cache.empty(),
        "new cache must be empty"
    );

    expect(
        !cache.initialized(),
        "new cache must not be initialized"
    );

    expect(
        cache.sequence_length() == 0,
        "initial sequence length must be zero"
    );

    expect(
        cache.embed_dim() == 0,
        "initial embed dimension must be zero"
    );

    expect(
        cache.batch_size() == 0,
        "initial batch size must be zero"
    );
}

// ============================================================
// TEST 2
// First 2D append
// ============================================================

void test_first_2d_append() {

    venla::KVCache cache;

    venla::Tensor key =
        venla::Tensor::zeros(
            {3, 4}
        );

    venla::Tensor value =
        venla::Tensor::zeros(
            {3, 4}
        );

    fill_tensor(
        key,
        1.0f
    );

    fill_tensor(
        value,
        101.0f
    );

    cache.append(
        key,
        value
    );

    expect(
        cache.initialized(),
        "cache should be initialized"
    );

    expect(
        !cache.empty(),
        "cache should not be empty"
    );

    expect(
        cache.sequence_length() == 3,
        "sequence length mismatch"
    );

    expect(
        cache.embed_dim() == 4,
        "embed dimension mismatch"
    );

    expect(
        cache.batch_size() == 1,
        "2D batch size should be one"
    );

    expect(
        cache.key().shape() ==
            venla::Shape({3, 4}),
        "key shape mismatch"
    );

    expect(
        cache.value().shape() ==
            venla::Shape({3, 4}),
        "value shape mismatch"
    );

    const float* k =
        cache.key()
            .data_as<float>();

    const float* v =
        cache.value()
            .data_as<float>();

    for (std::size_t i = 0;
         i < key.numel();
         ++i) {

        expect_close(
            k[i],
            1.0f +
            static_cast<float>(i),
            0.0f,
            "cached key mismatch"
        );

        expect_close(
            v[i],
            101.0f +
            static_cast<float>(i),
            0.0f,
            "cached value mismatch"
        );
    }
}

// ============================================================
// TEST 3
// Append second 2D block
// ============================================================

void test_second_2d_append() {

    venla::KVCache cache;

    venla::Tensor key1 =
        venla::Tensor::zeros(
            {2, 3}
        );

    venla::Tensor value1 =
        venla::Tensor::zeros(
            {2, 3}
        );

    venla::Tensor key2 =
        venla::Tensor::zeros(
            {1, 3}
        );

    venla::Tensor value2 =
        venla::Tensor::zeros(
            {1, 3}
        );

    fill_tensor(
        key1,
        10.0f
    );

    fill_tensor(
        value1,
        20.0f
    );

    fill_tensor(
        key2,
        30.0f
    );

    fill_tensor(
        value2,
        40.0f
    );

    cache.append(
        key1,
        value1
    );

    cache.append(
        key2,
        value2
    );

    expect(
        cache.sequence_length() == 3,
        "appended sequence length mismatch"
    );

    expect(
        cache.key().shape() ==
            venla::Shape({3, 3}),
        "appended key shape mismatch"
    );

    expect(
        cache.value().shape() ==
            venla::Shape({3, 3}),
        "appended value shape mismatch"
    );

    const float* k =
        cache.key()
            .data_as<float>();

    const float* v =
        cache.value()
            .data_as<float>();

    expect_close(
        k[0],
        10.0f,
        0.0f,
        "first key lost"
    );

    expect_close(
        k[5],
        15.0f,
        0.0f,
        "second first-block key lost"
    );

    expect_close(
        k[6],
        30.0f,
        0.0f,
        "new key not appended"
    );

    expect_close(
        v[0],
        20.0f,
        0.0f,
        "first value lost"
    );

    expect_close(
        v[6],
        40.0f,
        0.0f,
        "new value not appended"
    );
}

// ============================================================
// TEST 4
// Batched 3D cache
// ============================================================

void test_batched_cache() {

    venla::KVCache cache;

    venla::Tensor key1 =
        venla::Tensor::zeros(
            {2, 3, 4}
        );

    venla::Tensor value1 =
        venla::Tensor::zeros(
            {2, 3, 4}
        );

    venla::Tensor key2 =
        venla::Tensor::zeros(
            {2, 2, 4}
        );

    venla::Tensor value2 =
        venla::Tensor::zeros(
            {2, 2, 4}
        );

    fill_tensor(
        key1,
        1.0f
    );

    fill_tensor(
        value1,
        101.0f
    );

    fill_tensor(
        key2,
        201.0f
    );

    fill_tensor(
        value2,
        301.0f
    );

    cache.append(
        key1,
        value1
    );

    cache.append(
        key2,
        value2
    );

    expect(
        cache.batch_size() == 2,
        "batch size mismatch"
    );

    expect(
        cache.sequence_length() == 5,
        "batched sequence length mismatch"
    );

    expect(
        cache.embed_dim() == 4,
        "batched embed dimension mismatch"
    );

    expect(
        cache.key().shape() ==
            venla::Shape({2, 5, 4}),
        "batched key shape mismatch"
    );

    expect(
        cache.value().shape() ==
            venla::Shape({2, 5, 4}),
        "batched value shape mismatch"
    );

    const float* k =
        cache.key()
            .data_as<float>();

    const float* v =
        cache.value()
            .data_as<float>();

    const std::size_t
        batch_stride =
            5 * 4;

    expect_close(
        k[0],
        1.0f,
        0.0f,
        "batch 0 old key lost"
    );

    expect_close(
        k[batch_stride],
        13.0f,
        0.0f,
        "batch 1 old key lost"
    );

    expect_close(
        k[3 * 4],
        201.0f,
        0.0f,
        "batch 0 appended key missing"
    );

    expect_close(
        k[batch_stride + 3 * 4],
        209.0f,
        0.0f,
        "batch 1 appended key missing"
    );

    expect_close(
        v[0],
        101.0f,
        0.0f,
        "batch value lost"
    );

    expect_close(
        v[3 * 4],
        301.0f,
        0.0f,
        "batch appended value missing"
    );
}

// ============================================================
// TEST 5
// Shape mismatch
// ============================================================

void test_shape_mismatch() {

    venla::KVCache cache;

    venla::Tensor key =
        venla::Tensor::zeros(
            {2, 4}
        );

    venla::Tensor value =
        venla::Tensor::zeros(
            {2, 5}
        );

    bool threw =
        false;

    try {

        cache.append(
            key,
            value
        );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "key/value shape mismatch must throw"
    );
}

// ============================================================
// TEST 6
// Batch mismatch
// ============================================================

void test_batch_mismatch() {

    venla::KVCache cache;

    venla::Tensor key1 =
        venla::Tensor::zeros(
            {2, 2, 4}
        );

    venla::Tensor value1 =
        venla::Tensor::zeros(
            {2, 2, 4}
        );

    venla::Tensor key2 =
        venla::Tensor::zeros(
            {3, 1, 4}
        );

    venla::Tensor value2 =
        venla::Tensor::zeros(
            {3, 1, 4}
        );

    cache.append(
        key1,
        value1
    );

    bool threw =
        false;

    try {

        cache.append(
            key2,
            value2
        );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "batch mismatch must throw"
    );
}

// ============================================================
// TEST 7
// Embed dimension mismatch
// ============================================================

void test_embed_mismatch() {

    venla::KVCache cache;

    venla::Tensor key1 =
        venla::Tensor::zeros(
            {2, 4}
        );

    venla::Tensor value1 =
        venla::Tensor::zeros(
            {2, 4}
        );

    venla::Tensor key2 =
        venla::Tensor::zeros(
            {1, 5}
        );

    venla::Tensor value2 =
        venla::Tensor::zeros(
            {1, 5}
        );

    cache.append(
        key1,
        value1
    );

    bool threw =
        false;

    try {

        cache.append(
            key2,
            value2
        );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "embedding mismatch must throw"
    );
}

// ============================================================
// TEST 8
// Rank mismatch
// ============================================================

void test_rank_mismatch() {

    venla::KVCache cache;

    venla::Tensor key1 =
        venla::Tensor::zeros(
            {2, 4}
        );

    venla::Tensor value1 =
        venla::Tensor::zeros(
            {2, 4}
        );

    venla::Tensor key2 =
        venla::Tensor::zeros(
            {1, 2, 4}
        );

    venla::Tensor value2 =
        venla::Tensor::zeros(
            {1, 2, 4}
        );

    cache.append(
        key1,
        value1
    );

    bool threw =
        false;

    try {

        cache.append(
            key2,
            value2
        );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "rank mismatch must throw"
    );
}

// ============================================================
// TEST 9
// Clear
// ============================================================

void test_clear() {

    venla::KVCache cache;

    venla::Tensor key =
        venla::Tensor::zeros(
            {3, 4}
        );

    venla::Tensor value =
        venla::Tensor::zeros(
            {3, 4}
        );

    cache.append(
        key,
        value
    );

    cache.clear();

    expect(
        cache.empty(),
        "cache should be empty after clear"
    );

    expect(
        !cache.initialized(),
        "cache should not be initialized after clear"
    );

    expect(
        cache.sequence_length() == 0,
        "sequence length should reset"
    );

    expect(
        cache.embed_dim() == 0,
        "embed dimension should reset"
    );

    expect(
        cache.batch_size() == 0,
        "batch size should reset"
    );
}

// ============================================================
// TEST 10
// Access empty cache
// ============================================================

void test_empty_access() {

    venla::KVCache cache;

    bool threw =
        false;

    try {

        cache.key();

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "accessing empty key cache must throw"
    );

    threw = false;

    try {

        cache.value();

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "accessing empty value cache must throw"
    );
}

} // namespace

int main() {

    try {

        test_empty_state();

        test_first_2d_append();

        test_second_2d_append();

        test_batched_cache();

        test_shape_mismatch();

        test_batch_mismatch();

        test_embed_mismatch();

        test_rank_mismatch();

        test_clear();

        test_empty_access();

        std::cout
            << "VENLACPU KVCache Test: PASS"
            << std::endl;

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "VENLACPU KVCache Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
