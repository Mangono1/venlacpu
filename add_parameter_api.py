from pathlib import Path

ROOT = Path.home() / "venlacpu"


def replace_once(path, old, new):
    text = path.read_text()

    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"Pattern di {path} ditemukan {count} kali, "
            "perubahan dibatalkan."
        )

    path.write_text(text.replace(old, new))


# ============================================================
# TRANSFORMER DECODER LAYER HPP
# ============================================================

path = ROOT / "include/venla/nn/transformer_decoder_layer.hpp"

old = """    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t embed_dim() const;
"""

new = """    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    //
    // Mengembalikan seluruh parameter trainable layer:
    //
    //   MultiHeadAttention
    //   FeedForward
    //   LayerNorm 1
    //   LayerNorm 2
    //
    // Bias MHA dan FFN hanya dikembalikan jika menggunakan bias.
    // --------------------------------------------------------

    std::vector<Tensor*> parameters();

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t embed_dim() const;
"""

replace_once(path, old, new)

text = path.read_text()

if "#include <vector>" not in text:
    text = text.replace(
        "#include <cstddef>\n",
        "#include <cstddef>\n#include <vector>\n"
    )
    path.write_text(text)


# ============================================================
# TRANSFORMER DECODER LAYER CPP
# ============================================================

path = ROOT / "src/nn/transformer_decoder_layer.cpp"

old = """// ============================================================
// METADATA
// ============================================================

std::size_t
TransformerDecoderLayer::embed_dim() const {
"""

new = """// ============================================================
// PARAMETERS
// ============================================================

std::vector<Tensor*>
TransformerDecoderLayer::parameters() {

    std::vector<Tensor*> result;

    // --------------------------------------------------------
    // Multi-Head Attention weights
    // --------------------------------------------------------

    result.push_back(
        &self_attention_.q_weight()
    );

    result.push_back(
        &self_attention_.k_weight()
    );

    result.push_back(
        &self_attention_.v_weight()
    );

    result.push_back(
        &self_attention_.out_weight()
    );

    // --------------------------------------------------------
    // Multi-Head Attention biases
    // --------------------------------------------------------

    if (use_bias_) {

        result.push_back(
            &self_attention_.q_bias()
        );

        result.push_back(
            &self_attention_.k_bias()
        );

        result.push_back(
            &self_attention_.v_bias()
        );

        result.push_back(
            &self_attention_.out_bias()
        );
    }

    // --------------------------------------------------------
    // Feed Forward weights
    // --------------------------------------------------------

    result.push_back(
        &feed_forward_.input_weight()
    );

    result.push_back(
        &feed_forward_.output_weight()
    );

    // --------------------------------------------------------
    // Feed Forward biases
    // --------------------------------------------------------

    if (use_bias_) {

        result.push_back(
            &feed_forward_.input_bias()
        );

        result.push_back(
            &feed_forward_.output_bias()
        );
    }

    // --------------------------------------------------------
    // LayerNorm 1
    // --------------------------------------------------------

    result.push_back(
        &norm1_.weight()
    );

    result.push_back(
        &norm1_.bias()
    );

    // --------------------------------------------------------
    // LayerNorm 2
    // --------------------------------------------------------

    result.push_back(
        &norm2_.weight()
    );

    result.push_back(
        &norm2_.bias()
    );

    return result;
}

// ============================================================
// METADATA
// ============================================================

std::size_t
TransformerDecoderLayer::embed_dim() const {
"""

replace_once(path, old, new)

text = path.read_text()

if "#include <vector>" not in text:
    text = text.replace(
        "#include <sstream>\n",
        "#include <sstream>\n#include <vector>\n"
    )
    path.write_text(text)


# ============================================================
# TRANSFORMER DECODER HPP
# ============================================================

path = ROOT / "include/venla/nn/transformer_decoder.hpp"

old = """    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t embed_dim() const;
"""

new = """    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    //
    // Mengembalikan seluruh parameter trainable dari semua
    // decoder layer secara berurutan.
    // --------------------------------------------------------

    std::vector<Tensor*> parameters();

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t embed_dim() const;
"""

replace_once(path, old, new)


# ============================================================
# TRANSFORMER DECODER CPP
# ============================================================

path = ROOT / "src/nn/transformer_decoder.cpp"

old = """// ============================================================
// METADATA
// ============================================================

std::size_t
TransformerDecoder::embed_dim() const {
"""

new = """// ============================================================
// PARAMETERS
// ============================================================

std::vector<Tensor*>
TransformerDecoder::parameters() {

    std::vector<Tensor*> result;

    for (TransformerDecoderLayer& decoder_layer :
         layers_) {

        std::vector<Tensor*> layer_parameters =
            decoder_layer.parameters();

        result.insert(
            result.end(),
            layer_parameters.begin(),
            layer_parameters.end()
        );
    }

    return result;
}

// ============================================================
// METADATA
// ============================================================

std::size_t
TransformerDecoder::embed_dim() const {
"""

replace_once(path, old, new)


# ============================================================
# LANGUAGE MODEL HPP
# ============================================================

path = ROOT / "include/venla/nn/language_model.hpp"

old = """    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t vocab_size() const;
"""

new = """    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    //
    // Mengembalikan seluruh parameter trainable LanguageModel:
    //
    //   Embedding
    //   Transformer Decoder
    //   LM Head
    //
    // Positional encoding tidak termasuk karena bukan parameter
    // trainable.
    // --------------------------------------------------------

    std::vector<Tensor*> parameters();

    // --------------------------------------------------------
    // Metadata
    // --------------------------------------------------------

    std::size_t vocab_size() const;
"""

replace_once(path, old, new)

text = path.read_text()

if "#include <vector>" not in text:
    text = text.replace(
        "#include <cstddef>\n",
        "#include <cstddef>\n#include <vector>\n"
    )
    path.write_text(text)


# ============================================================
# LANGUAGE MODEL CPP
# ============================================================

path = ROOT / "src/nn/language_model.cpp"

old = """// ============================================================
// METADATA
// ============================================================

std::size_t
LanguageModel::vocab_size() const {
"""

new = """// ============================================================
// PARAMETERS
// ============================================================

std::vector<Tensor*>
LanguageModel::parameters() {

    std::vector<Tensor*> result;

    // --------------------------------------------------------
    // Embedding
    // --------------------------------------------------------

    result.push_back(
        &embedding_.weight()
    );

    // --------------------------------------------------------
    // Transformer Decoder
    // --------------------------------------------------------

    std::vector<Tensor*> decoder_parameters =
        decoder_.parameters();

    result.insert(
        result.end(),
        decoder_parameters.begin(),
        decoder_parameters.end()
    );

    // --------------------------------------------------------
    // LM Head
    // --------------------------------------------------------

    result.push_back(
        &lm_head_.weight()
    );

    if (use_bias_) {

        result.push_back(
            &lm_head_.bias()
        );
    }

    return result;
}

// ============================================================
// METADATA
// ============================================================

std::size_t
LanguageModel::vocab_size() const {
"""

replace_once(path, old, new)


# ============================================================
# LANGUAGE MODEL PARAMETER TEST
# ============================================================

test_path = ROOT / "tests/nn/test_language_model_parameters.cpp"

test_path.write_text(r'''#include "venla/nn/language_model.hpp"
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
''')


# ============================================================
# CMAKE
# ============================================================

cmake = ROOT / "CMakeLists.txt"
text = cmake.read_text()

if "venlacpu_language_model_parameters_test" not in text:

    marker = """add_executable(
    venlacpu_language_model_test
    tests/nn/test_language_model.cpp
)
"""

    if marker not in text:
        raise RuntimeError(
            "Blok venlacpu_language_model_test tidak ditemukan "
            "di CMakeLists.txt"
        )

    addition = marker + """
add_executable(
    venlacpu_language_model_parameters_test
    tests/nn/test_language_model_parameters.cpp
)

target_link_libraries(
    venlacpu_language_model_parameters_test
    PRIVATE
    venlacpu_core
)

add_test(
    NAME venlacpu_language_model_parameters_test
    COMMAND venlacpu_language_model_parameters_test
)
"""

    text = text.replace(
        marker,
        addition,
        1
    )

    cmake.write_text(text)


print()
print("============================================================")
print("VENLACPU PARAMETER API")
print("============================================================")
print()
print("Berhasil diperbarui:")
print()
print("  include/venla/nn/transformer_decoder_layer.hpp")
print("  src/nn/transformer_decoder_layer.cpp")
print("  include/venla/nn/transformer_decoder.hpp")
print("  src/nn/transformer_decoder.cpp")
print("  include/venla/nn/language_model.hpp")
print("  src/nn/language_model.cpp")
print("  tests/nn/test_language_model_parameters.cpp")
print("  CMakeLists.txt")
print()
print("Parameter traversal sekarang:")
print()
print("  LanguageModel")
print("    -> Embedding")
print("    -> TransformerDecoder")
print("       -> DecoderLayer")
print("          -> MultiHeadAttention")
print("          -> FeedForward")
print("          -> LayerNorm")
print("          -> LayerNorm")
print("    -> LM Head")
print()
print("============================================================")
