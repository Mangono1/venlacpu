#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"
#include "venla/training/causal_lm.hpp"
#include "venla/training/trainer.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace venla;

namespace {

CausalLMDataset make_dataset() {

    CausalLMDataset dataset;

    dataset.add_sequence({
        1, 2, 3, 4, 5, 6
    });

    dataset.add_sequence({
        2, 3, 4, 5, 6, 7
    });

    dataset.add_sequence({
        3, 4, 5, 6, 7, 8
    });

    dataset.add_sequence({
        4, 5, 6, 7, 8, 9
    });

    return dataset;
}

LanguageModel make_model() {

    return LanguageModel(
        16,
        8,
        8,
        2,
        16,
        1
    );
}

// ============================================================
// EXACT GRADIENT ACCUMULATION
// ============================================================

void test_gradient_accumulation() {

    LanguageModel model =
        make_model();

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        model.parameters()
    );

    CausalLMDataset dataset =
        make_dataset();

    TrainerConfig config;

    config.epochs = 1;
    config.batch_size = 2;

    // Dataset contains 4 sequences.
    // batch_size = 2 => exactly 2 batches.
    //
    // accumulation = 2:
    //
    // batch 1 \
    //          -> optimizer step #1
    // batch 2 /
    //
    // Therefore exactly one optimizer step.

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
        metrics.batches == 2
    );

    assert(
        metrics.optimizer_steps == 1
    );

    assert(
        trainer.global_step() == 1
    );

    assert(
        metrics.global_step == 1
    );

    assert(
        optimizer.step_count() == 1
    );

    assert(
        std::isfinite(metrics.loss)
    );

    std::cout
        << "[PASS] exact gradient accumulation"
        << std::endl;
}

// ============================================================
// PARTIAL ACCUMULATION
// ============================================================

void test_partial_accumulation() {

    LanguageModel model =
        make_model();

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        model.parameters()
    );

    CausalLMDataset dataset;

    dataset.add_sequence({
        1, 2, 3, 4, 5, 6
    });

    dataset.add_sequence({
        2, 3, 4, 5, 6, 7
    });

    dataset.add_sequence({
        3, 4, 5, 6, 7, 8
    });

    // 3 sequences / batch_size 2 = 2 batches.
    //
    // accumulation = 3
    //
    // batch 1 \
    // batch 2  > never reaches 3
    //
    // But batch 2 is the final batch, therefore
    // Trainer must still perform one optimizer step.

    TrainerConfig config;

    config.epochs = 1;
    config.batch_size = 2;
    config.gradient_accumulation_steps = 3;

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
        metrics.batches == 2
    );

    assert(
        metrics.optimizer_steps == 1
    );

    assert(
        trainer.global_step() == 1
    );

    assert(
        optimizer.step_count() == 1
    );

    std::cout
        << "[PASS] partial accumulation final step"
        << std::endl;
}

// ============================================================
// GRADIENT CLIPPING ENABLED
// ============================================================

void test_gradient_clipping_enabled() {

    LanguageModel model =
        make_model();

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        model.parameters()
    );

    CausalLMDataset dataset =
        make_dataset();

    TrainerConfig config;

    config.epochs = 1;
    config.batch_size = 2;

    // Extremely small threshold.
    // This should force clipping for the normal
    // non-zero model gradient.

    config.max_grad_norm = 1.0e-12f;

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
        metrics.optimizer_steps > 0
    );

    assert(
        metrics.gradient_finite
    );

    assert(
        metrics.gradient_clipped
    );

    assert(
        std::isfinite(
            metrics.gradient_norm
        )
    );

    assert(
        std::isfinite(
            metrics.clipped_gradient_norm
        )
    );

    assert(
        std::fabs(
            metrics.clipped_gradient_norm -
            config.max_grad_norm
        ) < 1.0e-18f
    );

    std::cout
        << "[PASS] gradient clipping enabled"
        << std::endl;
}

// ============================================================
// GRADIENT CLIPPING DISABLED
// ============================================================

void test_gradient_clipping_disabled() {

    LanguageModel model =
        make_model();

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        model.parameters()
    );

    CausalLMDataset dataset =
        make_dataset();

    TrainerConfig config;

    config.epochs = 1;
    config.batch_size = 2;

    // Contract:
    // 0 means clipping disabled.

    config.max_grad_norm = 0.0f;

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
        metrics.optimizer_steps > 0
    );

    assert(
        metrics.gradient_finite
    );

    assert(
        !metrics.gradient_clipped
    );

    assert(
        std::isfinite(
            metrics.gradient_norm
        )
    );

    assert(
        std::fabs(
            metrics.clipped_gradient_norm -
            metrics.gradient_norm
        ) < 1.0e-5f
    );

    std::cout
        << "[PASS] gradient clipping disabled"
        << std::endl;
}

// ============================================================
// METRICS CONSISTENCY
// ============================================================

void test_metrics_consistency() {

    LanguageModel model =
        make_model();

    Adam optimizer(
        0.001f
    );

    optimizer.add_parameters(
        model.parameters()
    );

    CausalLMDataset dataset =
        make_dataset();

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

    // 2 batches / epoch
    // accumulation = 2
    // => 1 optimizer step / epoch
    // 2 epochs => 2 optimizer steps.

    assert(
        metrics.batches == 2
    );

    assert(
        metrics.optimizer_steps == 1
    );

    assert(
        trainer.current_epoch() == 2
    );

    assert(
        trainer.global_step() == 2
    );

    assert(
        metrics.global_step == 2
    );

    assert(
        optimizer.step_count() == 2
    );

    assert(
        std::isfinite(metrics.loss)
    );

    assert(
        metrics.loss_finite
    );

    assert(
        metrics.gradient_finite
    );

    std::cout
        << "[PASS] trainer metrics consistency"
        << std::endl;
}

} // namespace

int main() {

    std::cout
        << "============================================================"
        << std::endl;

    std::cout
        << " VENLACPU 3.5.1 — TRAINER RELIABILITY TEST"
        << std::endl;

    std::cout
        << "============================================================"
        << std::endl;

    test_gradient_accumulation();

    test_partial_accumulation();

    test_gradient_clipping_enabled();

    test_gradient_clipping_disabled();

    test_metrics_consistency();

    std::cout
        << std::endl
        << "============================================================"
        << std::endl;

    std::cout
        << " TRAINER RELIABILITY TEST PASSED"
        << std::endl;

    std::cout
        << "============================================================"
        << std::endl;

    return 0;
}
