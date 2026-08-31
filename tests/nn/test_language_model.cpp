#include "venla/nn/language_model.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace venla;

namespace {

Tensor make_int32_tokens(
    const std::vector<std::int32_t>& values
) {
    Tensor tensor =
        Tensor::empty(
            {values.size()},
            DType::Int32,
            Device::cpu()
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

Tensor make_int64_tokens_2d(
    std::size_t batch,
    std::size_t sequence_length,
    const std::vector<std::int64_t>& values
) {
    assert(
        values.size() ==
        batch * sequence_length
    );

    Tensor tensor =
        Tensor::empty(
            {
                batch,
                sequence_length
            },
            DType::Int64,
            Device::cpu()
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

void test_1d_forward() {

    const std::size_t vocab_size = 32;
    const std::size_t max_seq_len = 16;
    const std::size_t embed_dim = 8;
    const std::size_t num_heads = 2;
    const std::size_t hidden_dim = 16;
    const std::size_t num_layers = 2;

    LanguageModel model(
        vocab_size,
        max_seq_len,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers
    );

    Tensor input =
        make_int32_tokens(
            {
                1,
                4,
                7,
                2
            }
        );

    Tensor logits =
        model.forward(
            input
        );

    assert(
        logits.dtype() ==
        DType::Float32
    );

    assert(
        logits.shape().ndim() == 2
    );

    assert(
        logits.shape()[0] == 4
    );

    assert(
        logits.shape()[1] ==
        vocab_size
    );

    assert(
        logits.numel() ==
        4 * vocab_size
    );
}

void test_2d_forward() {

    const std::size_t vocab_size = 32;
    const std::size_t max_seq_len = 16;
    const std::size_t embed_dim = 8;
    const std::size_t num_heads = 2;
    const std::size_t hidden_dim = 16;
    const std::size_t num_layers = 2;

    LanguageModel model(
        vocab_size,
        max_seq_len,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers
    );

    Tensor input =
        make_int64_tokens_2d(
            2,
            4,
            {
                1, 2, 3, 4,
                5, 6, 7, 8
            }
        );

    Tensor logits =
        model.forward(
            input
        );

    assert(
        logits.dtype() ==
        DType::Float32
    );

    assert(
        logits.shape().ndim() == 3
    );

    assert(
        logits.shape()[0] == 2
    );

    assert(
        logits.shape()[1] == 4
    );

    assert(
        logits.shape()[2] ==
        vocab_size
    );

    assert(
        logits.numel() ==
        2 * 4 * vocab_size
    );
}

void test_metadata() {

    LanguageModel model(
        64,
        32,
        16,
        4,
        32,
        3,
        true
    );

    assert(
        model.vocab_size() == 64
    );

    assert(
        model.max_seq_len() == 32
    );

    assert(
        model.embed_dim() == 16
    );

    assert(
        model.num_heads() == 4
    );

    assert(
        model.hidden_dim() == 32
    );

    assert(
        model.num_layers() == 3
    );

    assert(
        model.has_bias()
    );

    assert(
        model.embedding().vocab_size() ==
        64
    );

    assert(
        model.embedding().embedding_dim() ==
        16
    );

    assert(
        model.positional_encoding().max_seq_len() ==
        32
    );

    assert(
        model.decoder().num_layers() ==
        3
    );

    assert(
        model.lm_head().in_features() ==
        16
    );

    assert(
        model.lm_head().out_features() ==
        64
    );
}

void test_sequence_limit() {

    LanguageModel model(
        32,
        4,
        8,
        2,
        16,
        2
    );

    Tensor input =
        make_int32_tokens(
            {
                1,
                2,
                3,
                4,
                5
            }
        );

    bool failed =
        false;

    try {

        (void)model.forward(
            input
        );

    }
    catch (const std::out_of_range&) {

        failed =
            true;
    }

    assert(failed);
}

void test_invalid_token_dtype() {

    LanguageModel model(
        32,
        8,
        8,
        2,
        16,
        2
    );

    Tensor input =
        Tensor::zeros(
            {4},
            DType::Float32,
            Device::cpu()
        );

    bool failed =
        false;

    try {

        (void)model.forward(
            input
        );

    }
    catch (const std::runtime_error&) {

        failed =
            true;
    }

    assert(failed);
}

void test_autograd_path() {

    LanguageModel model(
        16,
        8,
        8,
        2,
        16,
        1
    );

    Tensor input =
        make_int32_tokens(
            {
                1,
                2,
                3
            }
        );

    Tensor logits =
        model.forward(
            input
        );

    assert(
        logits.requires_grad()
    );

    assert(
        !logits.is_leaf()
    );

    assert(
        logits.grad_state() != nullptr
    );

    assert(
        logits.grad_state()->grad_fn != nullptr
    );
}

} // namespace

int main() {

    test_1d_forward();

    test_2d_forward();

    test_metadata();

    test_sequence_limit();

    test_invalid_token_dtype();

    test_autograd_path();

    std::cout
        << "LanguageModel tests passed."
        << std::endl;

    return 0;
}
