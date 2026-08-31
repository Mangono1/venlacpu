#include "venla/training/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace venla {

namespace {

void validate_config(
    const TrainerConfig& config
) {
    if (config.epochs == 0) {

        throw std::invalid_argument(
            "Trainer: epochs must be greater than zero"
        );
    }

    if (config.batch_size == 0) {

        throw std::invalid_argument(
            "Trainer: batch_size must be greater than zero"
        );
    }

    if (config.gradient_accumulation_steps == 0) {

        throw std::invalid_argument(
            "Trainer: gradient_accumulation_steps "
            "must be greater than zero"
        );
    }
}

float read_loss(
    const Tensor& loss
) {
    if (loss.numel() != 1) {

        throw std::runtime_error(
            "Trainer: loss must be scalar"
        );
    }

    const float value =
        loss.data_as<float>()[0];

    if (!std::isfinite(value)) {

        throw std::runtime_error(
            "Trainer: loss is not finite"
        );
    }

    return value;
}

void validate_prompt(
    const Tensor& prompt
) {
    if (prompt.ndim() != 1 &&
        prompt.ndim() != 2) {

        throw std::runtime_error(
            "generate: prompt must be 1D or 2D"
        );
    }

    if (prompt.dtype() != DType::Int32 &&
        prompt.dtype() != DType::Int64) {

        throw std::runtime_error(
            "generate: prompt must use Int32 or Int64"
        );
    }

    if (!prompt.device().is_cpu()) {

        throw std::runtime_error(
            "generate: prompt must be on CPU"
        );
    }

    if (prompt.numel() == 0) {

        throw std::runtime_error(
            "generate: prompt cannot be empty"
        );
    }

    if (prompt.ndim() == 2 &&
        prompt.shape()[0] != 1) {

        throw std::runtime_error(
            "generate: batch size must be one"
        );
    }
}

std::int64_t read_token(
    const Tensor& tensor,
    std::size_t index
) {
    if (tensor.dtype() == DType::Int32) {

        return static_cast<std::int64_t>(
            tensor.data_as<std::int32_t>()[index]
        );
    }

    return tensor.data_as<std::int64_t>()[index];
}

std::size_t argmax(
    const float* data,
    std::size_t count
) {
    if (count == 0) {

        throw std::runtime_error(
            "generate: empty vocabulary"
        );
    }

    std::size_t best = 0;

    for (std::size_t i = 1;
         i < count;
         ++i) {

        if (data[i] > data[best]) {
            best = i;
        }
    }

    return best;
}

std::size_t sample_token(
    const float* logits,
    std::size_t vocab_size,
    float temperature,
    std::mt19937& generator
) {
    if (temperature <= 0.0f) {

        return argmax(
            logits,
            vocab_size
        );
    }

    float maximum =
        logits[0] /
        temperature;

    for (std::size_t i = 1;
         i < vocab_size;
         ++i) {

        maximum =
            std::max(
                maximum,
                logits[i] /
                temperature
            );
    }

    std::vector<float> probabilities(
        vocab_size,
        0.0f
    );

    double sum = 0.0;

    for (std::size_t i = 0;
         i < vocab_size;
         ++i) {

        const double value =
            std::exp(
                static_cast<double>(
                    logits[i] /
                    temperature -
                    maximum
                )
            );

        probabilities[i] =
            static_cast<float>(value);

        sum += value;
    }

    if (!std::isfinite(sum) ||
        sum <= 0.0) {

        return argmax(
            logits,
            vocab_size
        );
    }

    std::uniform_real_distribution<float>
        distribution(
            0.0f,
            static_cast<float>(sum)
        );

    const float random_value =
        distribution(generator);

    float cumulative = 0.0f;

    for (std::size_t i = 0;
         i < vocab_size;
         ++i) {

        cumulative +=
            probabilities[i];

        if (random_value <= cumulative) {
            return i;
        }
    }

    return vocab_size - 1;
}

} // namespace

Trainer::Trainer(
    LanguageModel& model,
    Optimizer& optimizer,
    const TrainerConfig& config
)
    : model_(&model),
      optimizer_(&optimizer),
      config_(config),
      current_epoch_(0),
      global_step_(0),
      last_metrics_() {

    validate_config(config_);
}

TrainingMetrics Trainer::train_epoch(
    const CausalLMDataset& dataset
) {
    if (dataset.empty()) {

        throw std::invalid_argument(
            "Trainer::train_epoch: dataset is empty"
        );
    }

    optimizer_->zero_grad();

    TrainingMetrics metrics;

    metrics.epoch =
        current_epoch_;

    const std::size_t batch_count =
        dataset.num_batches(
            config_.batch_size,
            config_.drop_last
        );

    if (batch_count == 0) {

        throw std::runtime_error(
            "Trainer::train_epoch: zero batches"
        );
    }

    CrossEntropyLoss loss_function(
        config_.ignore_index,
        true
    );

    double weighted_loss = 0.0;

    std::size_t accumulated_batches = 0;

    for (std::size_t batch_index = 0;
         batch_index < batch_count;
         ++batch_index) {

        CausalLMBatch batch =
            dataset.batch(
                batch_index,
                config_.batch_size,
                config_.drop_last
            );

        Tensor logits =
            model_->forward(
                batch.input
            );

        Tensor loss =
            loss_function.forward(
                logits,
                batch.targets
            );

        const float loss_value =
            read_loss(loss);

        weighted_loss +=
            static_cast<double>(
                loss_value
            ) *
            static_cast<double>(
                batch.valid_tokens
            );

        metrics.tokens +=
            batch.valid_tokens;

        ++metrics.batches;

        ++accumulated_batches;

        const bool last_batch =
            batch_index + 1 ==
            batch_count;

        const bool should_step =
            accumulated_batches >=
                config_.gradient_accumulation_steps ||
            last_batch;

        std::size_t divisor =
            accumulated_batches;

        if (!should_step) {

            divisor =
                config_.gradient_accumulation_steps;
        }

        Tensor upstream =
            Tensor::ones(
                Shape{},
                DType::Float32,
                Device::cpu()
            );

        upstream.data_as<float>()[0] =
            1.0f /
            static_cast<float>(
                divisor
            );

        loss.backward(
            upstream
        );

        if (should_step) {

            optimizer_->step();

            ++global_step_;

            ++metrics.optimizer_steps;

            optimizer_->zero_grad();

            accumulated_batches = 0;
        }
    }

    if (metrics.tokens != 0) {

        metrics.loss =
            static_cast<float>(
                weighted_loss /
                static_cast<double>(
                    metrics.tokens
                )
            );
    }

    metrics.global_step =
        global_step_;

    last_metrics_ =
        metrics;

    return metrics;
}

TrainingMetrics Trainer::fit(
    const CausalLMDataset& dataset
) {
    if (dataset.empty()) {

        throw std::invalid_argument(
            "Trainer::fit: dataset is empty"
        );
    }

    TrainingMetrics metrics;

    for (std::size_t epoch = 0;
         epoch < config_.epochs;
         ++epoch) {

        current_epoch_ =
            epoch + 1;

        metrics =
            train_epoch(
                dataset
            );
    }

    return metrics;
}

TrainingMetrics Trainer::evaluate(
    const CausalLMDataset& dataset
) {
    if (dataset.empty()) {

        throw std::invalid_argument(
            "Trainer::evaluate: dataset is empty"
        );
    }

    TrainingMetrics metrics;

    metrics.epoch =
        current_epoch_;

    const std::size_t batch_count =
        dataset.num_batches(
            config_.batch_size,
            config_.drop_last
        );

    CrossEntropyLoss loss_function(
        config_.ignore_index,
        true
    );

    double weighted_loss = 0.0;

    for (std::size_t batch_index = 0;
         batch_index < batch_count;
         ++batch_index) {

        CausalLMBatch batch =
            dataset.batch(
                batch_index,
                config_.batch_size,
                config_.drop_last
            );

        Tensor logits =
            model_->forward(
                batch.input
            );

        Tensor loss =
            loss_function.forward(
                logits,
                batch.targets
            );

        const float loss_value =
            read_loss(loss);

        weighted_loss +=
            static_cast<double>(
                loss_value
            ) *
            static_cast<double>(
                batch.valid_tokens
            );

        metrics.tokens +=
            batch.valid_tokens;

        ++metrics.batches;
    }

    if (metrics.tokens != 0) {

        metrics.loss =
            static_cast<float>(
                weighted_loss /
                static_cast<double>(
                    metrics.tokens
                )
            );
    }

    metrics.global_step =
        global_step_;

    last_metrics_ =
        metrics;

    return metrics;
}

std::size_t Trainer::current_epoch() const {
    return current_epoch_;
}

std::size_t Trainer::global_step() const {
    return global_step_;
}

const TrainingMetrics&
Trainer::last_metrics() const {
    return last_metrics_;
}

LanguageModel& Trainer::model() {
    return *model_;
}

Optimizer& Trainer::optimizer() {
    return *optimizer_;
}

const TrainerConfig&
Trainer::config() const {
    return config_;
}

Tensor generate(
    LanguageModel& model,
    const Tensor& prompt,
    const GenerationConfig& config
) {
    validate_prompt(prompt);

    if (!std::isfinite(config.temperature) ||
        config.temperature < 0.0f) {

        throw std::invalid_argument(
            "generate: invalid temperature"
        );
    }

    const std::size_t initial_length =
        prompt.shape()[
            prompt.ndim() - 1
        ];

    if (initial_length >
        model.max_seq_len()) {

        throw std::out_of_range(
            "generate: prompt exceeds max_seq_len"
        );
    }

    std::vector<std::int64_t> tokens;

    tokens.reserve(
        std::min(
            model.max_seq_len(),
            initial_length +
            config.max_new_tokens
        )
    );

    for (std::size_t i = 0;
         i < initial_length;
         ++i) {

        tokens.push_back(
            read_token(
                prompt,
                i
            )
        );
    }

    std::mt19937 generator(
        0x56454E4CUL
    );

    for (std::size_t step = 0;
         step < config.max_new_tokens;
         ++step) {

        if (tokens.size() >=
            model.max_seq_len()) {

            break;
        }

        Tensor current =
            Tensor::empty(
                {tokens.size()},
                DType::Int64,
                Device::cpu()
            );

        std::int64_t* current_data =
            current.data_as<std::int64_t>();

        for (std::size_t i = 0;
             i < tokens.size();
             ++i) {

            current_data[i] =
                tokens[i];
        }

        Tensor logits =
            model.forward(
                current
            );

        const std::size_t sequence_length =
            logits.shape()[0];

        const std::size_t vocab_size =
            logits.shape()[1];

        const float* row =
            logits.data_as<float>() +
            (
                sequence_length - 1
            ) *
            vocab_size;

        const std::size_t next_token =
            sample_token(
                row,
                vocab_size,
                config.temperature,
                generator
            );

        const std::int64_t token =
            static_cast<std::int64_t>(
                next_token
            );

        tokens.push_back(
            token
        );

        if (config.stop_on_eos &&
            config.eos_token_id >= 0 &&
            token == config.eos_token_id) {

            break;
        }
    }

    Tensor result =
        Tensor::empty(
            {tokens.size()},
            DType::Int64,
            Device::cpu()
        );

    std::int64_t* result_data =
        result.data_as<std::int64_t>();

    for (std::size_t i = 0;
         i < tokens.size();
         ++i) {

        result_data[i] =
            tokens[i];
    }

    return result;
}

} // namespace venla
