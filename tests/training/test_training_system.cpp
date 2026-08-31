#include "venla/nn/cross_entropy_loss.hpp"
#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"
#include "venla/training/causal_lm.hpp"
#include "venla/training/trainer.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace venla;

namespace {

Tensor make_tokens(
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

void test_causal_shift() {

    Tensor tokens =
        make_tokens(
            {
                1,
                2,
                3,
                4,
                5
            }
        );

    CausalLMBatch batch =
        make_causal_lm_batch(
            tokens
        );

    assert(
        batch.input.shape() ==
        Shape({4})
    );

    assert(
        batch.targets.shape() ==
        Shape({4})
    );

    const std::int32_t* input =
        batch.input.data_as<std::int32_t>();

    const std::int32_t* targets =
        batch.targets.data_as<std::int32_t>();

    assert(input[0] == 1);
    assert(input[1] == 2);
    assert(input[2] == 3);
    assert(input[3] == 4);

    assert(targets[0] == 2);
    assert(targets[1] == 3);
    assert(targets[2] == 4);
    assert(targets[3] == 5);
}

void test_dataset_batching() {

    CausalLMDataset dataset;

    dataset.add_sequence(
        {
            1,
            2,
            3,
            4
        }
    );

    dataset.add_sequence(
        {
            5,
            6,
            7
        }
    );

    dataset.add_sequence(
        {
            8,
            9,
            10,
            11,
            12
        }
    );

    assert(
        dataset.size() == 3
    );

    assert(
        dataset.max_sequence_length() == 5
    );

    assert(
        dataset.num_batches(2) == 2
    );

    CausalLMBatch batch =
        dataset.batch(
            0,
            2
        );

    assert(
        batch.batch_size == 2
    );

    assert(
        batch.sequence_length == 3
    );

    assert(
        batch.input.ndim() == 2
    );

    assert(
        batch.targets.ndim() == 2
    );

    assert(
        batch.valid_tokens == 5
    );
}

void test_training() {

    LanguageModel model(
        16,
        8,
        8,
        2,
        16,
        1
    );

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        model.parameters()
    );

    CausalLMDataset dataset;

    dataset.add_sequence(
        {
            1,
            2,
            3,
            4,
            5
        }
    );

    dataset.add_sequence(
        {
            2,
            3,
            4,
            5,
            6
        }
    );

    dataset.add_sequence(
        {
            3,
            4,
            5,
            6,
            7
        }
    );

    TrainerConfig config;

    config.epochs = 2;
    config.batch_size = 2;
    config.gradient_accumulation_steps = 2;

    Trainer trainer(
        model,
        optimizer,
        config
    );

    TrainingMetrics metrics =
        trainer.fit(
            dataset
        );

    assert(
        metrics.tokens > 0
    );

    assert(
        metrics.batches > 0
    );

    assert(
        metrics.optimizer_steps > 0
    );

    assert(
        trainer.current_epoch() == 2
    );

    assert(
        trainer.global_step() ==
        metrics.global_step
    );

    assert(
        std::isfinite(
            metrics.loss
        )
    );
}

void test_evaluation() {

    LanguageModel model(
        16,
        8,
        8,
        2,
        16,
        1
    );

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        model.parameters()
    );

    CausalLMDataset dataset;

    dataset.add_sequence(
        {
            1,
            2,
            3,
            4
        }
    );

    dataset.add_sequence(
        {
            2,
            3,
            4,
            5
        }
    );

    TrainerConfig config;

    config.batch_size = 2;

    Trainer trainer(
        model,
        optimizer,
        config
    );

    TrainingMetrics metrics =
        trainer.evaluate(
            dataset
        );

    assert(
        metrics.tokens > 0
    );

    assert(
        std::isfinite(
            metrics.loss
        )
    );

    assert(
        metrics.optimizer_steps == 0
    );
}

void test_generation() {

    LanguageModel model(
        16,
        16,
        8,
        2,
        16,
        1
    );

    Tensor prompt =
        make_tokens(
            {
                1,
                2,
                3
            }
        );

    GenerationConfig config;

    config.max_new_tokens = 4;

    config.temperature = 0.0f;

    config.eos_token_id = -1;

    Tensor generated =
        generate(
            model,
            prompt,
            config
        );

    assert(
        generated.ndim() == 1
    );

    assert(
        generated.shape()[0] >= 3
    );

    assert(
        generated.shape()[0] <= 7
    );

    assert(
        generated.dtype() ==
        DType::Int64
    );

    const std::int64_t* data =
        generated.data_as<std::int64_t>();

    assert(data[0] == 1);
    assert(data[1] == 2);
    assert(data[2] == 3);
}

} // namespace

int main() {

    test_causal_shift();

    test_dataset_batching();

    test_training();

    test_evaluation();

    test_generation();

    std::cout
        << "VENLACPU training system tests passed."
        << std::endl;

    return 0;
}
