#include "venla/training/trainer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
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
            "Trainer: gradient_accumulation_steps must be greater than zero"
        );
    }

    if (!std::isfinite(config.max_grad_norm) ||
        config.max_grad_norm < 0.0f) {
        throw std::invalid_argument(
            "Trainer: max_grad_norm must be finite and >= 0"
        );
    }

    if (config.early_stopping_min_delta < 0.0f ||
        !std::isfinite(config.early_stopping_min_delta)) {
        throw std::invalid_argument(
            "Trainer: invalid early_stopping_min_delta"
        );
    }

    if (config.early_stopping &&
        config.early_stopping_patience == 0) {
        throw std::invalid_argument(
            "Trainer: early_stopping_patience must be > 0"
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

    const float value = loss.data_as<float>()[0];

    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "Trainer: loss is not finite"
        );
    }

    return value;
}

float safe_perplexity(
    float loss
) {
    if (!std::isfinite(loss)) {
        return 0.0f;
    }

    // Avoid floating point overflow.
    if (loss > 88.0f) {
        return HUGE_VALF;
    }

    return std::exp(loss);
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

    for (std::size_t i = 1; i < count; ++i) {
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
        return argmax(logits, vocab_size);
    }

    float maximum =
        logits[0] / temperature;

    for (std::size_t i = 1;
         i < vocab_size;
         ++i) {
        maximum = std::max(
            maximum,
            logits[i] / temperature
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
                    logits[i] / temperature -
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

        cumulative += probabilities[i];

        if (random_value <= cumulative) {
            return i;
        }
    }

    return vocab_size - 1;
}

} // namespace

// ============================================================
// TRAINING HISTORY
// ============================================================

void TrainingHistory::clear() {
    records_.clear();
}

void TrainingHistory::add(
    const TrainingMetrics& metrics
) {
    records_.push_back(metrics);
}

std::size_t TrainingHistory::size() const {
    return records_.size();
}

bool TrainingHistory::empty() const {
    return records_.empty();
}

const TrainingMetrics&
TrainingHistory::at(
    std::size_t index
) const {
    return records_.at(index);
}

const std::vector<TrainingMetrics>&
TrainingHistory::records() const {
    return records_;
}

// ============================================================
// TRAINER
// ============================================================

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
      last_metrics_(),
      history_(),
      callbacks_(),
      has_best_model_(false),
      best_eval_loss_(HUGE_VALF),
      bad_epochs_(0),
      best_parameters_() {

    validate_config(config_);
}

void Trainer::add_callback(
    TrainerCallback& callback
) {
    callbacks_.push_back(&callback);
}

void Trainer::clear_callbacks() {
    callbacks_.clear();
}

const TrainingHistory&
Trainer::history() const {
    return history_;
}

bool Trainer::has_best_model() const {
    return has_best_model_;
}

float Trainer::best_eval_loss() const {
    return best_eval_loss_;
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

// ============================================================
// GRADIENT NORM
// ============================================================

float Trainer::calculate_gradient_norm() const {

    const std::vector<Tensor*> parameters =
        model_->parameters();

    double sum = 0.0;

    for (const Tensor* parameter : parameters) {

        if (parameter == nullptr ||
            !parameter->has_grad()) {
            continue;
        }

        const Tensor& gradient =
            parameter->grad();

        if (gradient.dtype() != DType::Float32) {
            throw std::runtime_error(
                "Trainer: gradients must be Float32"
            );
        }

        const float* data =
            gradient.data_as<float>();

        for (std::size_t i = 0;
             i < gradient.numel();
             ++i) {

            const float value = data[i];

            if (!std::isfinite(value)) {
                return HUGE_VALF;
            }

            sum +=
                static_cast<double>(value) *
                static_cast<double>(value);
        }
    }

    return static_cast<float>(
        std::sqrt(sum)
    );
}

void Trainer::clip_gradients(
    float max_norm,
    float current_norm
) {
    if (max_norm <= 0.0f ||
        !std::isfinite(current_norm) ||
        current_norm <= max_norm) {
        return;
    }

    const float scale =
        max_norm / current_norm;

    const std::vector<Tensor*> parameters =
        model_->parameters();

    for (Tensor* parameter : parameters) {

        if (parameter == nullptr ||
            !parameter->has_grad()) {
            continue;
        }

        Tensor& gradient =
            parameter->grad();

        if (gradient.dtype() != DType::Float32) {
            throw std::runtime_error(
                "Trainer: gradients must be Float32"
            );
        }

        float* data =
            gradient.data_as<float>();

        for (std::size_t i = 0;
             i < gradient.numel();
             ++i) {
            data[i] *= scale;
        }
    }
}

// ============================================================
// BEST MODEL
// ============================================================

void Trainer::snapshot_best_model() {

    best_parameters_.clear();

    const std::vector<Tensor*> parameters =
        model_->parameters();

    best_parameters_.reserve(
        parameters.size()
    );

    for (const Tensor* parameter : parameters) {

        if (parameter == nullptr) {
            best_parameters_.push_back({});
            continue;
        }

        if (parameter->dtype() != DType::Float32) {
            throw std::runtime_error(
                "Trainer: best model supports Float32 parameters only"
            );
        }

        const float* data =
            parameter->data_as<float>();

        best_parameters_.emplace_back(
            data,
            data + parameter->numel()
        );
    }

    has_best_model_ = true;
}

void Trainer::restore_best_model() {

    if (!has_best_model_) {
        throw std::runtime_error(
            "Trainer: no best model available"
        );
    }

    const std::vector<Tensor*> parameters =
        model_->parameters();

    if (parameters.size() !=
        best_parameters_.size()) {
        throw std::runtime_error(
            "Trainer: best model parameter count mismatch"
        );
    }

    for (std::size_t i = 0;
         i < parameters.size();
         ++i) {

        Tensor* parameter =
            parameters[i];

        if (parameter == nullptr) {
            continue;
        }

        if (parameter->dtype() != DType::Float32) {
            throw std::runtime_error(
                "Trainer: best model supports Float32 parameters only"
            );
        }

        if (parameter->numel() !=
            best_parameters_[i].size()) {
            throw std::runtime_error(
                "Trainer: best model parameter shape mismatch"
            );
        }

        float* data =
            parameter->data_as<float>();

        std::copy(
            best_parameters_[i].begin(),
            best_parameters_[i].end(),
            data
        );
    }
}

// ============================================================
// CHECKPOINT
// ============================================================

void Trainer::save_checkpoint(
    const std::string& path
) const {

    CheckpointMetadata metadata;

    metadata.epoch =
        current_epoch_;

    metadata.global_step =
        global_step_;

    metadata.best_eval_loss =
        best_eval_loss_;

    metadata.bad_epochs =
        bad_epochs_;

    metadata.has_best_model =
        has_best_model_;

    CheckpointEngine::save(
        path,
        *model_,
        *optimizer_,
        metadata
    );
}

void Trainer::load_checkpoint(
    const std::string& path
) {

    CheckpointMetadata metadata =
        CheckpointEngine::load(
            path,
            *model_,
            *optimizer_
        );

    current_epoch_ =
        metadata.epoch;

    global_step_ =
        metadata.global_step;

    best_eval_loss_ =
        metadata.best_eval_loss;

    bad_epochs_ =
        metadata.bad_epochs;

    // The checkpoint stores the active model parameters, but the
    // current checkpoint format does not store the separate
    // best-model parameter snapshot.
    //
    // Do not advertise a best model after loading because
    // restore_best_model() would otherwise have no parameters
    // to restore.
    has_best_model_ = false;
    best_parameters_.clear();

    last_metrics_.epoch =
        current_epoch_;

    last_metrics_.global_step =
        global_step_;

    last_metrics_.learning_rate =
        optimizer_->learning_rate();
}


// ============================================================
// EARLY STOPPING
// ============================================================

bool Trainer::check_early_stopping(
    float eval_loss
) {
    if (!config_.early_stopping) {
        return false;
    }

    if (!std::isfinite(eval_loss)) {
        ++bad_epochs_;
        return bad_epochs_ >=
            config_.early_stopping_patience;
    }

    if (!has_best_model_) {
        bad_epochs_ = 0;
        return false;
    }

    const float improvement =
        best_eval_loss_ - eval_loss;

    if (improvement >
        config_.early_stopping_min_delta) {
        bad_epochs_ = 0;
        return false;
    }

    ++bad_epochs_;

    return bad_epochs_ >=
        config_.early_stopping_patience;
}

// ============================================================
// TRAIN EPOCH
// ============================================================

TrainingMetrics Trainer::train_epoch(
    const CausalLMDataset& dataset
) {
    if (dataset.empty()) {
        throw std::invalid_argument(
            "Trainer::train_epoch: dataset is empty"
        );
    }

    const auto start =
        std::chrono::steady_clock::now();

    optimizer_->zero_grad();

    TrainingMetrics metrics;

    metrics.epoch =
        current_epoch_;

    metrics.learning_rate =
        optimizer_->learning_rate();

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

            const float gradient_norm =
                calculate_gradient_norm();

            metrics.gradient_norm =
                gradient_norm;

            if (!std::isfinite(gradient_norm)) {
                metrics.gradient_finite = false;

                optimizer_->zero_grad();

                throw std::runtime_error(
                    "Trainer: gradient contains NaN or Inf"
                );
            }

            if (config_.max_grad_norm > 0.0f &&
                gradient_norm >
                    config_.max_grad_norm) {

                clip_gradients(
                    config_.max_grad_norm,
                    gradient_norm
                );

                metrics.gradient_clipped = true;

                metrics.clipped_gradient_norm =
                    config_.max_grad_norm;
            } else {
                metrics.clipped_gradient_norm =
                    gradient_norm;
            }

            optimizer_->step();

            ++global_step_;

            ++metrics.optimizer_steps;

            optimizer_->zero_grad();

            accumulated_batches = 0;

            TrainingMetrics batch_metrics =
                metrics;

            batch_metrics.global_step =
                global_step_;

            for (TrainerCallback* callback :
                 callbacks_) {

                if (callback != nullptr) {
                    callback->on_batch_end(
                        batch_metrics
                    );
                }
            }
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

    metrics.perplexity =
        safe_perplexity(
            metrics.loss
        );

    metrics.global_step =
        global_step_;

    const auto finish =
        std::chrono::steady_clock::now();

    metrics.epoch_seconds =
        std::chrono::duration<double>(
            finish - start
        ).count();

    if (metrics.epoch_seconds > 0.0) {

        metrics.tokens_per_second =
            static_cast<double>(
                metrics.tokens
            ) /
            metrics.epoch_seconds;

        metrics.batches_per_second =
            static_cast<double>(
                metrics.batches
            ) /
            metrics.epoch_seconds;
    }

    metrics.loss_finite =
        std::isfinite(metrics.loss);

    last_metrics_ =
        metrics;

    return metrics;
}

// ============================================================
// EVALUATE
// ============================================================

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

    metrics.learning_rate =
        optimizer_->learning_rate();

    const std::size_t batch_count =
        dataset.num_batches(
            config_.batch_size,
            config_.drop_last
        );

    if (batch_count == 0) {
        throw std::runtime_error(
            "Trainer::evaluate: zero batches"
        );
    }

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

    metrics.perplexity =
        safe_perplexity(
            metrics.loss
        );

    metrics.global_step =
        global_step_;

    metrics.loss_finite =
        std::isfinite(metrics.loss);

    last_metrics_ =
        metrics;

    return metrics;
}

// ============================================================
// FIT WITHOUT VALIDATION
// ============================================================

TrainingMetrics Trainer::fit(
    const CausalLMDataset& dataset
) {
    if (dataset.empty()) {
        throw std::invalid_argument(
            "Trainer::fit: dataset is empty"
        );
    }

    for (TrainerCallback* callback :
         callbacks_) {

        if (callback != nullptr) {
            callback->on_train_begin();
        }
    }

    TrainingMetrics metrics;

    float previous_loss = 0.0f;

    bool has_previous_loss = false;

    const std::size_t start_epoch =
        current_epoch_;

    if (start_epoch >= config_.epochs) {
        last_metrics_.epoch =
            current_epoch_;

        last_metrics_.global_step =
            global_step_;

        last_metrics_.learning_rate =
            optimizer_->learning_rate();

        for (TrainerCallback* callback :
             callbacks_) {

            if (callback != nullptr) {
                callback->on_train_end();
            }
        }

        return last_metrics_;
    }

    for (std::size_t epoch = start_epoch;
         epoch < config_.epochs;
         ++epoch) {

        current_epoch_ =
            epoch + 1;

        for (TrainerCallback* callback :
             callbacks_) {

            if (callback != nullptr) {
                callback->on_epoch_begin(
                    current_epoch_
                );
            }
        }

        metrics =
            train_epoch(
                dataset
            );

        if (has_previous_loss) {
            metrics.loss_reduction =
                previous_loss -
                metrics.loss;
        }

        previous_loss =
            metrics.loss;

        has_previous_loss = true;

        history_.add(metrics);

        for (TrainerCallback* callback :
             callbacks_) {

            if (callback != nullptr) {
                callback->on_epoch_end(
                    metrics
                );
            }
        }
    }

    for (TrainerCallback* callback :
         callbacks_) {

        if (callback != nullptr) {
            callback->on_train_end();
        }
    }

    last_metrics_ =
        metrics;

    return metrics;
}

// ============================================================
// FIT WITH VALIDATION
// ============================================================

TrainingMetrics Trainer::fit(
    const CausalLMDataset& dataset,
    const CausalLMDataset& validation_dataset
) {
    if (dataset.empty()) {
        throw std::invalid_argument(
            "Trainer::fit: dataset is empty"
        );
    }

    if (validation_dataset.empty()) {
        throw std::invalid_argument(
            "Trainer::fit: validation dataset is empty"
        );
    }

    for (TrainerCallback* callback :
         callbacks_) {

        if (callback != nullptr) {
            callback->on_train_begin();
        }
    }

    TrainingMetrics metrics;

    float previous_loss = 0.0f;

    bool has_previous_loss = false;

    const std::size_t start_epoch =
        current_epoch_;

    if (start_epoch >= config_.epochs) {
        last_metrics_.epoch =
            current_epoch_;

        last_metrics_.global_step =
            global_step_;

        last_metrics_.learning_rate =
            optimizer_->learning_rate();

        for (TrainerCallback* callback :
             callbacks_) {

            if (callback != nullptr) {
                callback->on_train_end();
            }
        }

        return last_metrics_;
    }

    for (std::size_t epoch = start_epoch;
         epoch < config_.epochs;
         ++epoch) {

        current_epoch_ =
            epoch + 1;

        for (TrainerCallback* callback :
             callbacks_) {

            if (callback != nullptr) {
                callback->on_epoch_begin(
                    current_epoch_
                );
            }
        }

        metrics =
            train_epoch(
                dataset
            );

        if (has_previous_loss) {
            metrics.loss_reduction =
                previous_loss -
                metrics.loss;
        }

        previous_loss =
            metrics.loss;

        has_previous_loss = true;

        TrainingMetrics evaluation =
            evaluate(
                validation_dataset
            );

        metrics.eval_loss =
            evaluation.loss;

        metrics.eval_perplexity =
            evaluation.perplexity;

        metrics.global_step =
            global_step_;

        metrics.learning_rate =
            optimizer_->learning_rate();

        const bool improved =
            !has_best_model_ ||
            evaluation.loss <
                best_eval_loss_;

        if (improved) {

            best_eval_loss_ =
                evaluation.loss;

            metrics.is_best = true;

            bad_epochs_ = 0;

            if (config_.keep_best_model) {
                snapshot_best_model();
            }
        } else {

            metrics.is_best = false;

            ++bad_epochs_;
        }

        metrics.bad_epochs =
            bad_epochs_;

        history_.add(metrics);

        for (TrainerCallback* callback :
             callbacks_) {

            if (callback != nullptr) {
                callback->on_epoch_end(
                    metrics
                );
            }
        }

        if (config_.early_stopping &&
            bad_epochs_ >=
                config_.early_stopping_patience) {

            metrics.early_stopped = true;

            last_metrics_ =
                metrics;

            if (config_.keep_best_model &&
                has_best_model_) {
                restore_best_model();
            }

            break;
        }

        last_metrics_ =
            metrics;
    }

    for (TrainerCallback* callback :
         callbacks_) {

        if (callback != nullptr) {
            callback->on_train_end();
        }
    }

    return last_metrics_;
}

// ============================================================
// GENERATE
// ============================================================

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

        tokens.push_back(token);

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
