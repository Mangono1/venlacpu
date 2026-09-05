#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "venla/core/dtype.hpp"
#include "venla/core/device.hpp"
#include "venla/core/shape.hpp"
#include "venla/tensor/tensor.hpp"
#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"
#include "venla/training/trainer.hpp"

namespace {

using namespace venla;

void require(
    bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

LanguageModel make_model() {
    return LanguageModel(
        16,  // vocab_size
        8,   // max_seq_len
        8,   // embed_dim
        2,   // num_heads
        16,  // hidden_dim
        1    // num_layers
    );
}

CausalLMDataset make_dataset() {
    CausalLMDataset dataset(
        0,
        -100
    );

    dataset.add_sequence(
        std::vector<std::int64_t>{
            1, 2, 3, 4, 5, 6
        }
    );

    dataset.add_sequence(
        std::vector<std::int64_t>{
            2, 3, 4, 5, 6, 7
        }
    );

    return dataset;
}

void poison_model_with_nan(
    LanguageModel& model
) {
    const std::vector<Tensor*> parameters =
        model.parameters();

    require(
        !parameters.empty(),
        "LanguageModel has no parameters"
    );

    Tensor* parameter =
        parameters.front();

    require(
        parameter != nullptr,
        "First model parameter is null"
    );

    require(
        parameter->dtype() == DType::Float32,
        "Expected Float32 model parameter"
    );

    require(
        parameter->numel() > 0,
        "First model parameter is empty"
    );

    float* data =
        parameter->data_as<float>();

    /*
     * parameters().front() adalah embedding weight.
     *
     * Jangan merusak row 0 karena dataset tidak menggunakan
     * token ID 0. Kita sengaja merusak row token ID 1 agar
     * parameter NaN pasti masuk ke jalur LanguageModel::forward().
     *
     * embed_dim = 8 pada model test.
     */
    constexpr std::size_t embed_dim = 8;
    constexpr std::size_t token_id = 1;

    const std::size_t offset =
        token_id * embed_dim;

    require(
        offset < parameter->numel(),
        "Embedding token offset is out of range"
    );

    data[offset] =
        std::numeric_limits<float>::quiet_NaN();

    require(
        !std::isfinite(data[offset]),
        "Failed to inject NaN into active embedding parameter"
    );
}

void test_nonfinite_loss_is_rejected_before_optimizer() {
    LanguageModel model =
        make_model();

    Adam optimizer(
        0.001f
    );

    TrainerConfig config;

    config.epochs = 1;
    config.batch_size = 2;
    config.gradient_accumulation_steps = 1;

    Trainer trainer(
        model,
        optimizer,
        config
    );

    CausalLMDataset dataset =
        make_dataset();

    poison_model_with_nan(model);

    bool threw = false;
    std::string error_message;

    try {
        (void)trainer.fit(dataset);

    } catch (const std::exception& error) {
        threw = true;
        error_message = error.what();
    }

    require(
        threw,
        "Training accepted non-finite loss"
    );

    require(
        error_message.find("loss") !=
            std::string::npos,
        "Non-finite loss was not rejected as a loss error: " +
        error_message
    );

    require(
        error_message.find("NaN") !=
            std::string::npos ||
        error_message.find("Inf") !=
            std::string::npos ||
        error_message.find("finite") !=
            std::string::npos,
        "Error does not identify non-finite loss: " +
        error_message
    );

    require(
        optimizer.step_count() == 0,
        "Optimizer stepped despite non-finite loss"
    );

    std::cout
        << "[PASS] non-finite loss rejected before optimizer\n";

    std::cout
        << "       error: "
        << error_message
        << "\n";

    std::cout
        << "       optimizer steps: "
        << optimizer.step_count()
        << "\n";
}

void test_finite_training_still_works() {
    LanguageModel model =
        make_model();

    Adam optimizer(
        0.001f
    );

    TrainerConfig config;

    config.epochs = 1;
    config.batch_size = 2;
    config.gradient_accumulation_steps = 1;

    Trainer trainer(
        model,
        optimizer,
        config
    );

    CausalLMDataset dataset =
        make_dataset();

    TrainingMetrics metrics =
        trainer.fit(dataset);

    require(
        metrics.loss_finite,
        "Normal finite training produced non-finite loss"
    );

    require(
        std::isfinite(metrics.loss),
        "Normal training loss is not finite"
    );

    require(
        metrics.optimizer_steps == 1,
        "Expected exactly one optimizer step"
    );

    require(
        optimizer.step_count() == 1,
        "Adam step count should be exactly one"
    );

    std::cout
        << "[PASS] finite training remains valid\n";
}

} // namespace

int main() {
    std::cout
        << "============================================================\n"
        << " VENLACPU 3.5.1 — NON-FINITE LOSS REGRESSION TEST\n"
        << "============================================================\n";

    try {
        test_nonfinite_loss_is_rejected_before_optimizer();
        test_finite_training_still_works();

        std::cout
            << "\n============================================================\n"
            << " NON-FINITE LOSS REGRESSION TEST PASSED\n"
            << "============================================================\n";

        return EXIT_SUCCESS;

    } catch (const std::exception& error) {
        std::cerr
            << "\n[FAIL] "
            << error.what()
            << "\n";

        return EXIT_FAILURE;
    }
}
