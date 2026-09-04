#pragma once

#include "venla/nn/cross_entropy_loss.hpp"
#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"
#include "venla/training/causal_lm.hpp"
#include "venla/training/checkpoint.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace venla {

// ============================================================
// TRAINING CONFIG
// ============================================================

struct TrainerConfig {

    std::size_t epochs = 1;

    std::size_t batch_size = 1;

    std::size_t gradient_accumulation_steps = 1;

    bool drop_last = false;

    std::size_t log_every = 1;

    std::int64_t ignore_index = -100;

    // Gradient clipping.
    // 0 = disabled.
    float max_grad_norm = 0.0f;

    // Validation.
    bool evaluate_each_epoch = false;

    // Early stopping.
    bool early_stopping = false;

    std::size_t early_stopping_patience = 3;

    float early_stopping_min_delta = 0.0f;

    // Save best model in memory.
    bool keep_best_model = false;
};

// ============================================================
// TRAINING METRICS
// ============================================================

struct TrainingMetrics {

    // Core loss.
    float loss = 0.0f;

    float eval_loss = 0.0f;

    float perplexity = 0.0f;

    float eval_perplexity = 0.0f;

    float loss_reduction = 0.0f;

    // Learning.
    float learning_rate = 0.0f;

    // Counters.
    std::size_t tokens = 0;

    std::size_t batches = 0;

    // Evaluation metrics.
    //
    // evaluated_tokens:
    //     Number of non-ignored target tokens evaluated.
    //
    // correct_tokens:
    //     Number of evaluated tokens whose predicted class
    //     matches the target token.
    //
    // next_token_accuracy:
    //     correct_tokens / evaluated_tokens.
    std::size_t evaluated_tokens = 0;

    std::size_t correct_tokens = 0;

    float next_token_accuracy = 0.0f;

    std::size_t optimizer_steps = 0;

    std::size_t epoch = 0;

    std::size_t global_step = 0;

    // Performance.
    double epoch_seconds = 0.0;

    double tokens_per_second = 0.0;

    double batches_per_second = 0.0;

    // Gradient diagnostics.
    float gradient_norm = 0.0f;

    float clipped_gradient_norm = 0.0f;

    bool gradient_clipped = false;

    bool gradient_finite = true;

    bool loss_finite = true;

    // Training control.
    bool is_best = false;

    bool early_stopped = false;

    std::size_t bad_epochs = 0;
};

// ============================================================
// TRAINING HISTORY
// ============================================================

class TrainingHistory {

public:

    void clear();

    void add(const TrainingMetrics& metrics);

    std::size_t size() const;

    bool empty() const;

    const TrainingMetrics& at(
        std::size_t index
    ) const;

    const std::vector<TrainingMetrics>& records() const;

private:

    std::vector<TrainingMetrics> records_;

};

// ============================================================
// CALLBACK
// ============================================================

class TrainerCallback {

public:

    virtual ~TrainerCallback() = default;

    virtual void on_train_begin() {}

    virtual void on_epoch_begin(
        std::size_t
    ) {}

    virtual void on_batch_end(
        const TrainingMetrics&
    ) {}

    virtual void on_epoch_end(
        const TrainingMetrics&
    ) {}

    virtual void on_train_end() {}
};

// ============================================================
// TRAINER
// ============================================================

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

    TrainingMetrics fit(
        const CausalLMDataset& dataset,
        const CausalLMDataset& validation_dataset
    );

    TrainingMetrics evaluate(
        const CausalLMDataset& dataset
    );

    // --------------------------------------------------------
    // CALLBACK
    // --------------------------------------------------------

    void add_callback(
        TrainerCallback& callback
    );

    // Adds a callback whose lifetime is owned by Trainer.
    //
    // This is primarily used by language bindings where the
    // callback object is created dynamically (for example,
    // a Python callback adapter).
    void add_owned_callback(
        std::unique_ptr<TrainerCallback> callback
    );

    void clear_callbacks();

    // --------------------------------------------------------
    // HISTORY
    // --------------------------------------------------------

    const TrainingHistory& history() const;

    // --------------------------------------------------------
    // BEST MODEL
    // --------------------------------------------------------

    bool has_best_model() const;

    float best_eval_loss() const;

    // Restores best model parameters if available.
    void restore_best_model();

    // --------------------------------------------------------
    // CHECKPOINT
    // --------------------------------------------------------

    // Save model, optimizer and trainer state.
    void save_checkpoint(
        const std::string& path
    ) const;

    // Load model, optimizer and trainer state.
    //
    // Training will continue from the restored epoch/global
    // step when fit() is called again.
    void load_checkpoint(
        const std::string& path
    );

    // --------------------------------------------------------
    // STATE
    // --------------------------------------------------------

    std::size_t current_epoch() const;

    std::size_t global_step() const;

    const TrainingMetrics& last_metrics() const;

    LanguageModel& model();

    Optimizer& optimizer();

    const TrainerConfig& config() const;

private:

    float calculate_gradient_norm() const;

    void clip_gradients(
        float max_norm,
        float current_norm
    );

    void snapshot_best_model();

    bool check_early_stopping(
        float eval_loss,
        bool improved
    );

private:

    LanguageModel* model_;

    Optimizer* optimizer_;

    TrainerConfig config_;

    std::size_t current_epoch_;

    std::size_t global_step_;

    TrainingMetrics last_metrics_;

    TrainingHistory history_;

    // Non-owning callbacks registered directly by C++ callers.
    std::vector<TrainerCallback*> callbacks_;

    // Owned callbacks used by language bindings and other
    // dynamically-created callback adapters.
    std::vector<std::unique_ptr<TrainerCallback>> owned_callbacks_;

    // Tracks whether a validation baseline has been
    // established independently from parameter snapshotting.
    bool has_best_eval_;

    // Indicates whether a restorable best-parameter snapshot
    // is currently available.
    bool has_best_model_;

    float best_eval_loss_;

    std::size_t bad_epochs_;

    std::vector<std::vector<float>> best_parameters_;

};

// ============================================================
// GENERATION
// ============================================================

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
