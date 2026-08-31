#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"

#include <cassert>
#include <cstddef>
#include <iostream>
#include <unordered_set>
#include <vector>

using namespace venla;

namespace {

void test_parameter_count_with_bias() {

    LanguageModel model(
        32,
        16,
        8,
        2,
        16,
        2,
        true
    );

    std::vector<Tensor*> parameters =
        model.parameters();

    // Per decoder layer:
    //
    // MHA:
    //   4 weights
    //   4 biases
    //
    // FFN:
    //   2 weights
    //   2 biases
    //
    // LayerNorm:
    //   norm1 weight/bias
    //   norm2 weight/bias
    //
    // Total = 16 parameters/layer.
    //
    // 2 layers = 32
    // Embedding = 1
    // LM head = 2
    //
    // Total = 35.

    assert(
        parameters.size() == 35
    );
}

void test_parameter_count_without_bias() {

    LanguageModel model(
        32,
        16,
        8,
        2,
        16,
        2,
        false
    );

    std::vector<Tensor*> parameters =
        model.parameters();

    // Per decoder layer:
    //
    // MHA weights = 4
    // FFN weights = 2
    // LayerNorm = 4
    //
    // Total = 10/layer.
    //
    // 2 layers = 20
    // Embedding = 1
    // LM head = 1
    //
    // Total = 22.

    assert(
        parameters.size() == 22
    );
}

void test_all_parameters_require_grad() {

    LanguageModel model(
        32,
        16,
        8,
        2,
        16,
        2,
        true
    );

    std::vector<Tensor*> parameters =
        model.parameters();

    assert(
        !parameters.empty()
    );

    for (Tensor* parameter : parameters) {

        assert(
            parameter != nullptr
        );

        assert(
            parameter->requires_grad()
        );

        assert(
            parameter->is_leaf()
        );

        assert(
            parameter->dtype() ==
            DType::Float32
        );

        assert(
            parameter->device().is_cpu()
        );

        assert(
            !parameter->empty()
        );
    }
}

void test_parameters_are_unique() {

    LanguageModel model(
        32,
        16,
        8,
        2,
        16,
        3,
        true
    );

    std::vector<Tensor*> parameters =
        model.parameters();

    std::unordered_set<Tensor*> unique;

    for (Tensor* parameter : parameters) {

        assert(
            parameter != nullptr
        );

        auto inserted =
            unique.insert(parameter);

        assert(
            inserted.second
        );
    }

    assert(
        unique.size() ==
        parameters.size()
    );
}

void test_optimizer_registration() {

    LanguageModel model(
        32,
        16,
        8,
        2,
        16,
        2,
        true
    );

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        model.parameters()
    );

    assert(
        optimizer.parameter_count() ==
        model.parameters().size()
    );
}

void test_parameter_shapes() {

    const std::size_t vocab_size = 32;
    const std::size_t embed_dim = 8;

    LanguageModel model(
        vocab_size,
        16,
        embed_dim,
        2,
        16,
        1,
        true
    );

    std::vector<Tensor*> parameters =
        model.parameters();

    assert(
        parameters.size() == 19
    );

    // Embedding
    assert(
        parameters[0]->shape() ==
        Shape({vocab_size, embed_dim})
    );

    // MHA weights
    assert(
        parameters[1]->shape() ==
        Shape({embed_dim, embed_dim})
    );

    assert(
        parameters[2]->shape() ==
        Shape({embed_dim, embed_dim})
    );

    assert(
        parameters[3]->shape() ==
        Shape({embed_dim, embed_dim})
    );

    assert(
        parameters[4]->shape() ==
        Shape({embed_dim, embed_dim})
    );

    // MHA biases
    assert(
        parameters[5]->shape() ==
        Shape({embed_dim})
    );

    assert(
        parameters[6]->shape() ==
        Shape({embed_dim})
    );

    assert(
        parameters[7]->shape() ==
        Shape({embed_dim})
    );

    assert(
        parameters[8]->shape() ==
        Shape({embed_dim})
    );

    // FFN weights
    assert(
        parameters[9]->shape() ==
        Shape({embed_dim, 16})
    );

    assert(
        parameters[10]->shape() ==
        Shape({16, embed_dim})
    );

    // FFN biases
    assert(
        parameters[11]->shape() ==
        Shape({16})
    );

    assert(
        parameters[12]->shape() ==
        Shape({embed_dim})
    );

    // LayerNorm 1
    assert(
        parameters[13]->shape() ==
        Shape({embed_dim})
    );

    assert(
        parameters[14]->shape() ==
        Shape({embed_dim})
    );

    // LayerNorm 2
    assert(
        parameters[15]->shape() ==
        Shape({embed_dim})
    );

    assert(
        parameters[16]->shape() ==
        Shape({embed_dim})
    );

    // LM Head
    assert(
        parameters[17]->shape() ==
        Shape({embed_dim, vocab_size})
    );

    assert(
        parameters[18]->shape() ==
        Shape({vocab_size})
    );
}

} // namespace

int main() {

    test_parameter_count_with_bias();

    test_parameter_count_without_bias();

    test_all_parameters_require_grad();

    test_parameters_are_unique();

    test_optimizer_registration();

    test_parameter_shapes();

    std::cout
        << "LanguageModel parameter tests passed."
        << std::endl;

    return 0;
}
