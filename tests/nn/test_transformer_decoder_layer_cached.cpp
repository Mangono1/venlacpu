#include "venla/nn/kv_cache.hpp"
#include "venla/nn/transformer_decoder_layer.hpp"

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
                static_cast<int>(i % 17) - 8
            ) *
            0.05f;
    }
}

// ============================================================
// Metadata
// ============================================================

void test_metadata() {

    venla::TransformerDecoderLayer layer(
        8,
        2,
        16,
        true
    );

    expect(
        layer.embed_dim() == 8,
        "embed dimension mismatch"
    );

    expect(
        layer.num_heads() == 2,
        "head count mismatch"
    );

    expect(
        layer.hidden_dim() == 16,
        "hidden dimension mismatch"
    );

    expect(
        layer.is_causal(),
        "decoder layer must be causal"
    );
}

// ============================================================
// Single cached token
// ============================================================

void test_single_token() {

    venla::TransformerDecoderLayer layer(
        8,
        2,
        16,
        true
    );

    venla::KVCache cache;

    venla::Tensor input =
        venla::Tensor::zeros(
            {1, 8}
        );

    fill_tensor(input);

    venla::Tensor output =
        layer.forward_cached(
            input,
            cache
        );

    expect(
        output.shape() ==
            venla::Shape({1, 8}),
        "cached layer output shape mismatch"
    );

    expect(
        cache.initialized(),
        "cache must be initialized"
    );

    expect(
        cache.sequence_length() == 1,
        "cache sequence must be one"
    );

    const float* data =
        output.data_as<float>();

    for (std::size_t i = 0;
         i < output.numel();
         ++i) {

        expect(
            std::isfinite(data[i]),
            "cached layer output contains non-finite value"
        );
    }
}

// ============================================================
// Incremental growth
// ============================================================

void test_incremental_growth() {

    venla::TransformerDecoderLayer layer(
        8,
        2,
        16,
        true
    );

    venla::KVCache cache;

    for (std::size_t step = 0;
         step < 5;
         ++step) {

        venla::Tensor input =
            venla::Tensor::zeros(
                {1, 8}
            );

        float* data =
            input.data_as<float>();

        for (std::size_t i = 0;
             i < 8;
             ++i) {

            data[i] =
                static_cast<float>(
                    step * 8 +
                    i +
                    1
                ) *
                0.02f;
        }

        venla::Tensor output =
            layer.forward_cached(
                input,
                cache
            );

        expect(
            output.shape() ==
                venla::Shape({1, 8}),
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
// Cached vs normal
// ============================================================

void test_cached_matches_normal() {

    venla::TransformerDecoderLayer layer(
        8,
        2,
        16,
        true
    );

    venla::Tensor sequence =
        venla::Tensor::zeros(
            {4, 8}
        );

    fill_tensor(sequence);

    venla::Tensor normal =
        layer.forward(
            sequence
        );

    venla::KVCache cache;

    const float* sequence_data =
        sequence.data_as<float>();

    const float* normal_data =
        normal.data_as<float>();

    for (std::size_t step = 0;
         step < 4;
         ++step) {

        venla::Tensor token =
            venla::Tensor::zeros(
                {1, 8}
            );

        float* token_data =
            token.data_as<float>();

        for (std::size_t i = 0;
             i < 8;
             ++i) {

            token_data[i] =
                sequence_data[
                    step * 8 +
                    i
                ];
        }

        venla::Tensor cached =
            layer.forward_cached(
                token,
                cache
            );

        const float* cached_data =
            cached.data_as<float>();

        for (std::size_t i = 0;
             i < 8;
             ++i) {

            expect_close(
                cached_data[i],
                normal_data[
                    step * 8 +
                    i
                ],
                1e-4f,
                "cached decoder layer differs from normal"
            );
        }
    }
}

// ============================================================
// Batched cached inference
// ============================================================

void test_batched() {

    venla::TransformerDecoderLayer layer(
        8,
        2,
        16,
        true
    );

    venla::KVCache cache;

    for (std::size_t step = 0;
         step < 3;
         ++step) {

        venla::Tensor input =
            venla::Tensor::zeros(
                {2, 1, 8}
            );

        float* data =
            input.data_as<float>();

        for (std::size_t i = 0;
             i < input.numel();
             ++i) {

            data[i] =
                static_cast<float>(
                    step * input.numel() +
                    i +
                    1
                ) *
                0.01f;
        }

        venla::Tensor output =
            layer.forward_cached(
                input,
                cache
            );

        expect(
            output.shape() ==
                venla::Shape({2, 1, 8}),
            "batched decoder layer shape mismatch"
        );

        expect(
            cache.batch_size() == 2,
            "batched cache batch mismatch"
        );

        expect(
            cache.sequence_length() ==
                step + 1,
            "batched cache sequence mismatch"
        );
    }
}

// ============================================================
// Cache clear
// ============================================================

void test_clear() {

    venla::TransformerDecoderLayer layer(
        8,
        2,
        16,
        true
    );

    venla::KVCache cache;

    venla::Tensor input =
        venla::Tensor::zeros(
            {1, 8}
        );

    fill_tensor(input);

    layer.forward_cached(
        input,
        cache
    );

    expect(
        cache.sequence_length() == 1,
        "cache should contain token"
    );

    cache.clear();

    expect(
        cache.empty(),
        "cache should be empty after clear"
    );

    layer.forward_cached(
        input,
        cache
    );

    expect(
        cache.sequence_length() == 1,
        "cache did not restart"
    );
}

// ============================================================
// Invalid input
// ============================================================

void test_invalid_input() {

    venla::TransformerDecoderLayer layer(
        8,
        2,
        16,
        true
    );

    venla::KVCache cache;

    venla::Tensor input =
        venla::Tensor::zeros(
            {1, 7}
        );

    bool threw = false;

    try {

        layer.forward_cached(
            input,
            cache
        );

    } catch (const std::exception&) {

        threw = true;
    }

    expect(
        threw,
        "invalid feature dimension must throw"
    );
}

} // namespace

int main() {

    try {

        test_metadata();

        test_single_token();

        test_incremental_growth();

        test_cached_matches_normal();

        test_batched();

        test_clear();

        test_invalid_input();

        std::cout
            << "VENLACPU Cached TransformerDecoderLayer Test: PASS"
            << std::endl;

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "VENLACPU Cached TransformerDecoderLayer Test: FAIL"
            << std::endl;

        std::cerr
            << error.what()
            << std::endl;

        return 1;
    }
}
