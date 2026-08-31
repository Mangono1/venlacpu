#include "venla/tokenizer/tokenizer.hpp"
#include "venla/nn/language_model.hpp"
#include "venla/tensor/tensor.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "VENLACPU NATIVE API SMOKE TEST" << std::endl;
    std::cout << "==============================================" << std::endl;

    // ========================================================
    // TOKENIZER
    // ========================================================

    venla::BPETokenizer tokenizer;

    tokenizer.train(
        std::string(
            "halo dunia halo dunia "
            "ini adalah venlacpu"
        )
    );

    assert(tokenizer.trained());

    std::vector<std::size_t> ids =
        tokenizer.encode("halo dunia");

    assert(!ids.empty());

    std::string decoded =
        tokenizer.decode(ids);

    assert(!decoded.empty());

    std::cout << "[OK] BPETokenizer train" << std::endl;
    std::cout << "[OK] BPETokenizer encode" << std::endl;
    std::cout << "[OK] BPETokenizer decode" << std::endl;

    // ========================================================
    // TOKENIZER PERSISTENCE
    // ========================================================

    const std::string tokenizer_path =
        "venlacpu_tokenizer_test.bin";

    tokenizer.save(tokenizer_path);

    venla::BPETokenizer loaded_tokenizer =
        venla::BPETokenizer::load(
            tokenizer_path
        );

    assert(loaded_tokenizer.trained());

    std::vector<std::size_t> loaded_ids =
        loaded_tokenizer.encode("halo dunia");

    assert(!loaded_ids.empty());

    std::cout << "[OK] BPETokenizer save" << std::endl;
    std::cout << "[OK] BPETokenizer load" << std::endl;

    // ========================================================
    // TENSOR
    // ========================================================

    venla::Tensor tensor =
        venla::Tensor::zeros(
            venla::Shape({2, 3}),
            venla::DType::Float32,
            venla::Device::cpu()
        );

    assert(
        tensor.ndim() == 2
    );

    assert(
        tensor.numel() == 6
    );

    assert(
        tensor.nbytes() > 0
    );

    std::string tensor_info =
        tensor.info();

    assert(!tensor_info.empty());

    std::cout << "[OK] Tensor creation" << std::endl;
    std::cout << "[OK] Tensor metadata" << std::endl;
    std::cout << "[OK] Tensor info()" << std::endl;

    // ========================================================
    // LANGUAGE MODEL
    // ========================================================

    venla::LanguageModel model(
        32,   // vocab size
        16,   // max sequence length
        16,   // embedding dimension
        4,    // attention heads
        32,   // hidden dimension
        1,    // transformer layers
        true  // bias
    );

    std::vector<venla::Tensor*> parameters =
        model.parameters();

    assert(!parameters.empty());

    std::cout
        << "[OK] LanguageModel construction"
        << std::endl;

    std::cout
        << "[OK] LanguageModel parameters(): "
        << parameters.size()
        << std::endl;

    // ========================================================
    // CLEANUP
    // ========================================================

    std::remove(
        tokenizer_path.c_str()
    );

    std::cout << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "ALL NATIVE API TESTS PASSED" << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
