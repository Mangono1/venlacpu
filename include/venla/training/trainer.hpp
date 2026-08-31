#pragma once

#include "venla/nn/cross_entropy_loss.hpp"
#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"
#include "venla/training/causal_lm.hpp"

#include <cstddef>
#include <cstdint>

namespace venla {

struct TrainerConfig {

    std::size_t epochs = 1;

    std::size_t batch_size = 1;

    std::size_t gradient_accumulation_steps = 1;

    bool drop_last = false;

    std::size_t log_every = 1;

    std::int64_t ignore_index = -100;
};

struct TrainingMetrics {

    float loss = 0.0f;

    std::size_t tokens = 0;

    std::size_t batches = 0;

    std::size_t optimizer_steps = 0;

    std::size_t epoch = 0;

    std::size_t global_step = 0;
};

class Trainer {
public:

    Trainer(
        LanguageModel& model,
        Optimizer& optimizer,
        const TrainerConfig& config = TrainerConfig()
    );

    TrainingMetrics train_epoch(
        const CausalLMDataset& dataset
    );

    TrainingMetrics fit(
        const CausalLMDataset& dataset
    );

    TrainingMetrics evaluate(
        const CausalLMDataset& dataset
    );

    std::size_t current_epoch() const;

    std::size_t global_step() const;

    const TrainingMetrics& last_metrics() const;

    LanguageModel& model();

    Optimizer& optimizer();

    const TrainerConfig& config() const;

private:

    LanguageModel* model_;

    Optimizer* optimizer_;

    TrainerConfig config_;

    std::size_t current_epoch_;

    std::size_t global_step_;

    TrainingMetrics last_metrics_;
};

struct GenerationConfig {

    std::size_t max_new_tokens = 32;

    float temperature = 0.0f;

    std::int64_t eos_token_id = -1;

    bool stop_on_eos = true;
};

Tensor generate(
    LanguageModel& model,
    const Tensor& prompt,
    const GenerationConfig& config = GenerationConfig()
);

} // namespace venla
