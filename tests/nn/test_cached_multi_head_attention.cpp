#include "venla/nn/kv_cache.hpp"
#include "venla/nn/multi_head_attention.hpp"

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
        throw std::runtime_error(message);
    }
}

void expect_close(
    float actual,
    float expected,
    float tolerance,
    const std::string& message
) {
    if (std::fabs(actual - expected) > tolerance) {

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
                static_cast<int>(i % 11) - 5
            ) *
            0.1f;
    }
}

// ============================================================
// TEST 1
// Metadata
// ============================================================

void test_metadata() {

    venla::MultiHeadAttention mha(
        8,
        2,
        true,
        true
    );

    expect(
        mha.embed_dim() == 8,
        "embed dimension mismatch"
    );

    expect(
        mha.num_heads() == 2,
        "head count mismatch"
    );

    expect(
        mha.head_dim() == 4,
        "head dimension mismatch"
    );

    expect(
        mha.is_causal(),
        "MHA should be causal"
    );
}

// ============================================================
// TEST 2
// Single token cached inference
// ============================================================

void test_single_token_cache() {

    venla::MultiHeadAttention mha(
        4,
        2,
        true,
        true
    );

    venla::KVCache cache;

    venla::Tensor token =
        venla::Tensor::zeros(
            {1, 4}
        );

    fill_tensor(token);

    venla::Tensor output =
        mha.forward_cached(
            token,
            cache
        );

    expect(
        output.shape() ==
            venla::Shape({1, 4}),
        "cached output shape mismatch"
    );

    expect(
        cache.initialized(),
        "cache should be initialized"
    );

    expect(
        cache.sequence_length() == 1,
        "cache sequence should be one"
    );

    expect(
        cache.embed_dim() == 4,
        "cache embed dimension mismatch"
    );

    expect(
        cache.batch_size() == 1,
        "cache batch size mismatch"
    );

    const float* data =
        output.data_as<float>();

    for (std::size_t i = 0;
         i < output.numel();
         ++i) {

        expect(
            std::isfinite(data[i]),
            "cached output contains non-finite value"
        );
    }
}

// ============================================================
// TEST 3
// Incremental cache growth
// ============================================================

void test_incremental_growth() {

    venla::MultiHeadAttention mha(
        4,
        2,
        true,
        true
    );

    venla::KVCache cache;

    for (std::size_t step = 0;
         step < 5;
         ++step) {

        venla::Tensor token =
            venla::Tensor::zeros(
                {1, 4}
            );

        float* data =
            token.data_as<float>();

        for (std::size_t i = 0;
             i < 4;
             ++i) {

            data[i] =
                static_cast<float>(
                    step * 4 + i + 1
                ) *
                0.1f;
        }

        venla::Tensor output =
            mha.forward_cached(
                token,
                cache
            );

        expect(
            output.shape() ==
                venla::Shape({1, 4}),
            "incremental output shape mismatch"
        );

        expect(
            cache.sequence_length() ==
                step + 1,
            "cache did not grow correctly"
        );
    }
}

// ============================================================
// TEST 4
// Cached result vs normal causal attention
//
// Feed:
//
//   [t0, t1, t2, t3]
//
// Normal MHA:
//
//   output = MHA([t0,t1,t2,t3])
//
// Cached:
//
//   MHA(t0)
//   MHA(t1)
//   MHA(t2)
//   MHA(t3)
//
// The output of each cached step must equal the
// corresponding row of normal causal attention.
// ============================================================

void test_cached_matches_normal() {

    venla::MultiHeadAttention mha(
        4,
        2,
        true,
        true
    );

    venla::Tensor sequence =
        venla::Tensor::zeros(
            {4, 4}
        );

    fill_tensor(sequence);

    venla::Tensor normal =
        mha.forward(
            sequence
        );

    venla::KVCache cache;

    for (std::size_t step = 0;
         step < 4;
         ++step) {

        venla::Tensor token =
            venla::Tensor::zeros(
                {1, 4}
            );

        const float* source =
            sequence.data_as<float>();

        float* destination =
            token.data_as<float>();

        for (std::size_t i = 0;
             i < 4;
             ++i) {

            destination[i] =
                source[
                    step * 4 +
                    i
                ];
        }

        venla::Tensor cached =
            mha.forward_cached(
                token,
                cache
            );

        const float* normal_data =
            normal.data_as<float>();

        const float* cached_data =
            cached.data_as<float>();

        for (std::size_t i = 0;
             i < 4;
             ++i) {

            expect_close(
                cached_data[i],
                normal_data[
                    step * 4 +
                    i
                ],
                1e-5f,
                "cached output differs from normal output"
            );
        }
    }
}

// ============================================================
// TEST 5
// Batched cached inference
// ============================================================

void test_batched_cache() {

    venla::MultiHeadAttention mha(
        4,
        2,
        true,
        true
    );

    venla::KVCache cache;

    for (std::size_t step = 0;
         step < 3;
         ++step) {

        venla::Tensor input =
            venla::Tensor::zeros(
                {2, 1, 4}
            );

        float* data =
            input.data_as<float>();

        for (std::size_t i = 0;
             i < input.numel();
             ++i) {

            data[i] =
                static_cast<float>(
                    step * input.numel() +
                    i + 1
                ) *
                0.05f;
        }

        venla::Tensor output =
            mha.forward_cached(
                input,
                cache
            );

        expect(
            output.shape() ==
                venla::Shape({2, 1, 4}),
            "batched cached output shape mismatch"
        );

        expect(
            cache.batch_size() == 2,
            "batched cache batch size mismatch"
        );

        expect(
            cache.sequence_length() ==
                step + 1,
            "batched cache sequence mismatch"
        );

        const float* output_data =
            output.data_as<float>();

        for (std::size_t i = 0;
             i < output.numel();
             ++i) {

            expect(
                std::isfinite(
                    output_data[i]
                ),
                "batched cached output is non-finite"
            );
        }
    }
}

// ============================================================
// TEST 6
// Cache clear and restart
// ============================================================

void test_cache_clear_restart() {

    venla::MultiHeadAttention mha(
        4,
        2,
        true,
        true
    );

    venla::KVCache cache;

    venla::Tensor token =
        venla::Tensor::zeros(
            {1, 4}
        );

    fill_tensor(token);

    mha.forward_cached(
        token,
        cache
    );

    expect(
        cache.sequence_length() == 1,
        "initial cached token missing"
    );

    cache.clear();

    expect(
        cache.empty(),
        "cache should be empty"
    );

    mha.forward_cached(
        token,
        cache
    );

    expect(
        cache.sequence_length() == 1,
        "cache did not restart after clear"
    );
}

// ============================================================
// TEST 7
// Invalid feature dimension
// ============================================================

void test_invalid_dimension() {

    venla::MultiHeadAttention mha(
        4,
        2,
        true,
        true
    );

    venla::KVCache cache;

    venla::Tensor input =
        venla::Tensor::zeros(
            {1, 5}
        );

    bool threw = false;

    try {

        mha.forward_cached(
            input,
            cache
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
// Cache must preserve K/V
// ============================================================

void test_cache_preserves_history() {

    venla::MultiHeadAttention mha(
        4,
        2,
        true,
        true
    );

    venla::KVCache cache;

    venla::Tensor first =
        venla::Tensor::zeros(
            {1, 4}
        );

    venla::Tensor second =
        venla::Tensor::zeros(
            {1, 4}
        );

    float* first_data =
        first.data_as<float>();

    float* second_data =
        second.data_as<float>();

    for (std::size_t i = 0;
         i < 4;
         ++i) {

        first_data[i] =
            static_cast<float>(i + 1);

        second_data[i] =
            static_cast<float>(i + 10);
    }

    mha.forward_cached(
        first,
        cache
    );

    venla::Tensor first_key_copy =
        cache.key();

    mha.forward_cached(
        second,
        cache
    );

    expect(
        cache.sequence_length() == 2,
        "cache should contain two tokens"
    );

    const float* before =
        first_key_copy.data_as<float>();

    const float* after =
        cache.key().data_as<float>();

    for (std::size_t i = 0;
         i < 4;
         ++i) {

        expect_close(
            after[i],
            before[i],
            0.0f,
            "old cached key changed"
        );
    }
}

} // namespace

int main() {

    try {

        test_metadata();

        test_single_token_cache();

        test_incremental_growth();

        test_cached_matches_normal();

        test_batched_cache();

        test_cache_clear_restart();

        test_invalid_dimension();

        test_cache_preserves_history();

        std::cout
            << "VENLACPU Cached MultiHeadAttention Test: PASS"
            << std::endl;

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "VENLACPU Cached MultiHeadAttention Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
