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

    // v3.0.0 evaluation metrics.
    //
    // For causal LM evaluation, every non-ignored target
    // contributes exactly one evaluated token.
    assert(
        metrics.evaluated_tokens ==
        metrics.tokens
    );

    assert(
        metrics.correct_tokens <=
        metrics.evaluated_tokens
    );

    assert(
        metrics.evaluated_tokens > 0
    );

    assert(
        std::isfinite(
            metrics.next_token_accuracy
        )
    );

    assert(
        metrics.next_token_accuracy >= 0.0f
    );

    assert(
        metrics.next_token_accuracy <= 1.0f
    );

    const float expected_accuracy =
        static_cast<float>(
            static_cast<double>(
                metrics.correct_tokens
            ) /
            static_cast<double>(
                metrics.evaluated_tokens
            )
        );

    assert(
        std::fabs(
            metrics.next_token_accuracy -
            expected_accuracy
        ) < 1e-6f
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

void test_evaluation_ignore_index() {

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

    // Different sequence lengths force padding in the batch.
    //
    // Sequence 1: 4 tokens -> 3 valid targets
    // Sequence 2: 2 tokens -> 1 valid target
    //
    // Total valid target tokens = 4.
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
            3
        }
    );

    TrainerConfig config;

    config.batch_size = 2;
    config.ignore_index = -100;

    Trainer trainer(
        model,
        optimizer,
        config
    );

    CausalLMBatch batch =
        dataset.batch(
            0,
            2
        );

    // The batch contains:
    //
    // row 0: 3 valid targets
    // row 1: 1 valid target + 2 ignored targets
    assert(
        batch.valid_tokens == 4
    );

    assert(
        batch.targets.numel() == 6
    );

    const std::int64_t* targets =
        batch.targets.data_as<std::int64_t>();

    std::size_t ignored_targets = 0;

    for (std::size_t i = 0;
         i < batch.targets.numel();
         ++i) {

        if (targets[i] ==
            config.ignore_index) {

            ++ignored_targets;
        }
    }

    assert(
        ignored_targets == 2
    );

    TrainingMetrics metrics =
        trainer.evaluate(
            dataset
        );

    // Accuracy must ignore padded target positions.
    assert(
        metrics.tokens == 4
    );

    assert(
        metrics.evaluated_tokens == 4
    );

    assert(
        metrics.correct_tokens <=
        metrics.evaluated_tokens
    );

    assert(
        std::isfinite(
            metrics.next_token_accuracy
        )
    );

    assert(
        metrics.next_token_accuracy >= 0.0f
    );

    assert(
        metrics.next_token_accuracy <= 1.0f
    );

    const float expected_accuracy =
        static_cast<float>(
            static_cast<double>(
                metrics.correct_tokens
            ) /
            static_cast<double>(
                metrics.evaluated_tokens
            )
        );

    assert(
        std::fabs(
            metrics.next_token_accuracy -
            expected_accuracy
        ) < 1e-6f
    );
}

void test_unseen_text_evaluation() {

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

    // --------------------------------------------------
    // TRAIN DATA
    //
    // This dataset is the ONLY dataset passed to fit().
    // --------------------------------------------------
    CausalLMDataset train_dataset;

    train_dataset.add_sequence(
        {
            1,
            2,
            3,
            4,
            5
        }
    );

    train_dataset.add_sequence(
        {
            2,
            3,
            4,
            5,
            6
        }
    );

    train_dataset.add_sequence(
        {
            3,
            4,
            5,
            6,
            7
        }
    );

    // --------------------------------------------------
    // UNSEEN DATA
    //
    // These sequences are intentionally different from
    // the training sequences and are never passed to fit().
    // --------------------------------------------------
    CausalLMDataset unseen_dataset;

    unseen_dataset.add_sequence(
        {
            8,
            9,
            10,
            11
        }
    );

    unseen_dataset.add_sequence(
        {
            12,
            13
        }
    );

    TrainerConfig config;

    config.epochs = 2;
    config.batch_size = 2;
    config.gradient_accumulation_steps = 1;
    config.ignore_index = -100;

    Trainer trainer(
        model,
        optimizer,
        config
    );

    // Train ONLY on train_dataset.
    TrainingMetrics train_metrics =
        trainer.fit(
            train_dataset
        );

    assert(
        train_metrics.tokens > 0
    );

    assert(
        train_metrics.batches > 0
    );

    assert(
        trainer.current_epoch() == 2
    );

    // --------------------------------------------------
    // EVALUATE UNSEEN DATA
    // --------------------------------------------------
    TrainingMetrics unseen_metrics =
        trainer.evaluate(
            unseen_dataset
        );

    // Unseen evaluation must produce real metrics.
    assert(
        unseen_metrics.tokens > 0
    );

    assert(
        unseen_metrics.evaluated_tokens ==
        unseen_metrics.tokens
    );

    assert(
        unseen_metrics.correct_tokens <=
        unseen_metrics.evaluated_tokens
    );

    assert(
        unseen_metrics.evaluated_tokens > 0
    );

    assert(
        std::isfinite(
            unseen_metrics.loss
        )
    );

    assert(
        unseen_metrics.loss >= 0.0f
    );

    assert(
        std::isfinite(
            unseen_metrics.perplexity
        )
    );

    assert(
        unseen_metrics.perplexity > 0.0f
    );

    assert(
        std::isfinite(
            unseen_metrics.next_token_accuracy
        )
    );

    assert(
        unseen_metrics.next_token_accuracy >= 0.0f
    );

    assert(
        unseen_metrics.next_token_accuracy <= 1.0f
    );

    const float expected_accuracy =
        static_cast<float>(
            static_cast<double>(
                unseen_metrics.correct_tokens
            ) /
            static_cast<double>(
                unseen_metrics.evaluated_tokens
            )
        );

    assert(
        std::fabs(
            unseen_metrics.next_token_accuracy -
            expected_accuracy
        ) < 1e-6f
    );

    // Evaluation must not perform optimizer steps.
    assert(
        unseen_metrics.optimizer_steps == 0
    );

    // Evaluation must not advance the training epoch.
    assert(
        trainer.current_epoch() == 2
    );
}


void test_best_model_and_early_stopping() {

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

    CausalLMDataset train_dataset;

    train_dataset.add_sequence(
        {
            1,
            2,
            3,
            4,
            5
        }
    );

    train_dataset.add_sequence(
        {
            2,
            3,
            4,
            5,
            6
        }
    );

    CausalLMDataset validation_dataset;

    validation_dataset.add_sequence(
        {
            1,
            2,
            3,
            4,
            5
        }
    );

    validation_dataset.add_sequence(
        {
            2,
            3,
            4,
            5,
            6
        }
    );

    TrainerConfig config;

    config.epochs = 5;
    config.batch_size = 2;
    config.gradient_accumulation_steps = 1;

    config.evaluate_each_epoch = true;

    config.early_stopping = true;
    config.early_stopping_patience = 2;

    // Deliberately enormous so that only the first epoch
    // can qualify as an improvement.
    config.early_stopping_min_delta = 1.0e9f;

    config.keep_best_model = true;

    Trainer trainer(
        model,
        optimizer,
        config
    );

    // --------------------------------------------------------
    // CAPTURE BEST-EPOCH PARAMETERS
    // --------------------------------------------------------

    class BestEpochCapture
        : public TrainerCallback {

    public:

        std::vector<std::vector<float>>
        parameters;

        bool captured = false;

        void on_epoch_end(
            const TrainingMetrics& metrics
        ) override {

            if (!captured &&
                metrics.is_best) {

                // The callback runs after the epoch has been
                // evaluated and after snapshot_best_model().
                //
                // Capture the actual model parameters at the
                // best epoch for later comparison.

                captured = true;

                // The model pointer is supplied separately
                // by the test after construction.
            }
        }
    };

    BestEpochCapture callback;

    // We need direct access to the model from the callback,
    // so use a small local callback implementation that stores
    // the parameter snapshot.
    class ParameterCaptureCallback
        : public TrainerCallback {

    public:

        explicit ParameterCaptureCallback(
            LanguageModel& model
        )
            : model_(model) {
        }

        LanguageModel& model_;

        std::vector<std::vector<float>>
        best_parameters;

        bool captured = false;

        void on_epoch_end(
            const TrainingMetrics& metrics
        ) override {

            if (!metrics.is_best ||
                captured) {
                return;
            }

            const std::vector<Tensor*> parameters =
                model_.parameters();

            best_parameters.clear();

            best_parameters.reserve(
                parameters.size()
            );

            for (const Tensor* parameter :
                 parameters) {

                assert(
                    parameter != nullptr
                );

                assert(
                    parameter->dtype() ==
                    DType::Float32
                );

                const float* data =
                    parameter->data_as<float>();

                best_parameters.emplace_back(
                    data,
                    data + parameter->numel()
                );
            }

            captured = true;
        }
    };

    ParameterCaptureCallback capture(
        model
    );

    trainer.add_callback(
        capture
    );

    // --------------------------------------------------------
    // TRAIN
    // --------------------------------------------------------

    TrainingMetrics metrics =
        trainer.fit(
            train_dataset,
            validation_dataset
        );

    // --------------------------------------------------------
    // BEST MODEL TRACKING
    // --------------------------------------------------------

    assert(
        trainer.has_best_model()
    );

    assert(
        std::isfinite(
            trainer.best_eval_loss()
        )
    );

    assert(
        capture.captured
    );

    assert(
        !capture.best_parameters.empty()
    );

    // --------------------------------------------------------
    // EARLY STOPPING
    // --------------------------------------------------------

    assert(
        trainer.history().size() == 3
    );

    const TrainingMetrics& epoch1 =
        trainer.history().at(0);

    const TrainingMetrics& epoch2 =
        trainer.history().at(1);

    const TrainingMetrics& epoch3 =
        trainer.history().at(2);

    assert(
        epoch1.is_best
    );

    assert(
        !epoch2.is_best
    );

    assert(
        !epoch3.is_best
    );

    assert(
        epoch2.bad_epochs == 1
    );

    assert(
        epoch3.bad_epochs == 2
    );

    assert(
        epoch3.early_stopped
    );

    assert(
        metrics.early_stopped
    );

    assert(
        trainer.current_epoch() == 3
    );

    assert(
        trainer.current_epoch() <
        config.epochs
    );

    // --------------------------------------------------------
    // BEST LOSS
    // --------------------------------------------------------

    assert(
        std::fabs(
            trainer.best_eval_loss() -
            epoch1.eval_loss
        ) < 1e-6f
    );

    // --------------------------------------------------------
    // VERIFY AUTOMATIC RESTORE
    // --------------------------------------------------------

    const std::vector<Tensor*> parameters =
        model.parameters();

    assert(
        parameters.size() ==
        capture.best_parameters.size()
    );

    for (std::size_t i = 0;
         i < parameters.size();
         ++i) {

        assert(
            parameters[i] != nullptr
        );

        const float* data =
            parameters[i]->data_as<float>();

        assert(
            parameters[i]->numel() ==
            capture.best_parameters[i].size()
        );

        for (std::size_t j = 0;
             j < parameters[i]->numel();
             ++j) {

            assert(
                std::fabs(
                    data[j] -
                    capture.best_parameters[i][j]
                ) < 1e-6f
            );
        }
    }

    // --------------------------------------------------------
    // VERIFY EXPLICIT RESTORE
    // --------------------------------------------------------

    trainer.restore_best_model();

    const std::vector<Tensor*> parameters_after =
        model.parameters();

    assert(
        parameters_after.size() ==
        capture.best_parameters.size()
    );

    for (std::size_t i = 0;
         i < parameters_after.size();
         ++i) {

        assert(
            parameters_after[i] != nullptr
        );

        const float* data =
            parameters_after[i]->data_as<float>();

        assert(
            parameters_after[i]->numel() ==
            capture.best_parameters[i].size()
        );

        for (std::size_t j = 0;
             j < parameters_after[i]->numel();
             ++j) {

            assert(
                std::fabs(
                    data[j] -
                    capture.best_parameters[i][j]
                ) < 1e-6f
            );
        }
    }
}

void test_generation_regression() {

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

    // --------------------------------------------------
    // DETERMINISTIC GENERATION
    // --------------------------------------------------
    GenerationConfig config;

    config.max_new_tokens = 5;
    config.temperature = 0.0f;
    config.eos_token_id = -1;
    config.stop_on_eos = true;

    Tensor generated_a =
        generate(
            model,
            prompt,
            config
        );

    Tensor generated_b =
        generate(
            model,
            prompt,
            config
        );

    // Output must remain one-dimensional Int64.
    assert(
        generated_a.ndim() == 1
    );

    assert(
        generated_a.dtype() ==
        DType::Int64
    );

    assert(
        generated_b.ndim() == 1
    );

    assert(
        generated_b.dtype() ==
        DType::Int64
    );

    // Prompt must always be preserved.
    assert(
        generated_a.shape()[0] >=
        prompt.shape()[0]
    );

    assert(
        generated_b.shape()[0] >=
        prompt.shape()[0]
    );

    const std::int32_t* prompt_data =
        prompt.data_as<std::int32_t>();

    const std::int64_t* data_a =
        generated_a.data_as<std::int64_t>();

    const std::int64_t* data_b =
        generated_b.data_as<std::int64_t>();

    // Prompt tokens must remain unchanged.
    assert(
        data_a[0] ==
        static_cast<std::int64_t>(
            prompt_data[0]
        )
    );

    assert(
        data_a[1] ==
        static_cast<std::int64_t>(
            prompt_data[1]
        )
    );

    assert(
        data_a[2] ==
        static_cast<std::int64_t>(
            prompt_data[2]
        )
    );

    assert(
        data_b[0] ==
        static_cast<std::int64_t>(
            prompt_data[0]
        )
    );

    assert(
        data_b[1] ==
        static_cast<std::int64_t>(
            prompt_data[1]
        )
    );

    assert(
        data_b[2] ==
        static_cast<std::int64_t>(
            prompt_data[2]
        )
    );

    // At most max_new_tokens may be appended.
    assert(
        generated_a.shape()[0] <=
        prompt.shape()[0] +
        config.max_new_tokens
    );

    assert(
        generated_b.shape()[0] <=
        prompt.shape()[0] +
        config.max_new_tokens
    );

    // --------------------------------------------------
    // DETERMINISM
    //
    // temperature = 0 means greedy deterministic
    // generation. Repeated calls must produce identical
    // token sequences.
    // --------------------------------------------------
    assert(
        generated_a.numel() ==
        generated_b.numel()
    );

    for (std::size_t i = 0;
         i < generated_a.numel();
         ++i) {

        assert(
            data_a[i] ==
            data_b[i]
        );
    }

    // --------------------------------------------------
    // TOKEN VALIDITY
    // --------------------------------------------------
    for (std::size_t i = 0;
         i < generated_a.numel();
         ++i) {

        assert(
            data_a[i] >= 0
        );

        assert(
            data_a[i] <
            static_cast<std::int64_t>(
                model.vocab_size()
            )
        );
    }

    // --------------------------------------------------
    // PROMPT IMMUTABILITY
    // --------------------------------------------------
    const std::int32_t* prompt_after =
        prompt.data_as<std::int32_t>();

    assert(
        prompt_after[0] == 1
    );

    assert(
        prompt_after[1] == 2
    );

    assert(
        prompt_after[2] == 3
    );

    // --------------------------------------------------
    // ZERO NEW TOKENS
    // --------------------------------------------------
    GenerationConfig zero_config;

    zero_config.max_new_tokens = 0;
    zero_config.temperature = 0.0f;
    zero_config.eos_token_id = -1;
    zero_config.stop_on_eos = true;

    Tensor zero_generated =
        generate(
            model,
            prompt,
            zero_config
        );

    assert(
        zero_generated.ndim() == 1
    );

    assert(
        zero_generated.dtype() ==
        DType::Int64
    );

    assert(
        zero_generated.numel() ==
        prompt.numel()
    );

    const std::int64_t* zero_data =
        zero_generated.data_as<std::int64_t>();

    for (std::size_t i = 0;
         i < prompt.numel();
         ++i) {

        assert(
            zero_data[i] ==
            static_cast<std::int64_t>(
                prompt_after[i]
            )
        );
    }
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

    test_evaluation_ignore_index();

    test_unseen_text_evaluation();

    test_best_model_and_early_stopping();

    test_generation_regression();

    test_generation();

    std::cout
        << "VENLACPU training system tests passed."
        << std::endl;

    return 0;
}
