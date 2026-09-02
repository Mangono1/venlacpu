#include "venla/nn/language_model.hpp"
#include "venla/optim/optimizer.hpp"
#include "venla/training/causal_lm.hpp"
#include "venla/training/trainer.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <iostream>

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

void test_trainer_resume() {

    const char* checkpoint_path =
        "venlacpu_trainer_resume_test.bin";

    CausalLMDataset dataset =
        make_dataset();

    // ========================================================
    // TRAINER A
    // ========================================================

    LanguageModel model_a(
        16,
        8,
        8,
        2,
        16,
        1
    );

    Adam optimizer_a(
        0.001f
    );

    optimizer_a.add_parameters(
        model_a.parameters()
    );

    TrainerConfig config_a;

    config_a.epochs = 1;
    config_a.batch_size = 2;
    config_a.gradient_accumulation_steps = 1;

    Trainer trainer_a(
        model_a,
        optimizer_a,
        config_a
    );

    TrainingMetrics metrics_a =
        trainer_a.fit(
            dataset
        );

    assert(
        trainer_a.current_epoch() == 1
    );

    assert(
        trainer_a.global_step() > 0
    );

    assert(
        metrics_a.global_step ==
        trainer_a.global_step()
    );

    assert(
        std::isfinite(
            metrics_a.loss
        )
    );

    const std::size_t saved_epoch =
        trainer_a.current_epoch();

    const std::size_t saved_global_step =
        trainer_a.global_step();

    const std::size_t saved_optimizer_steps =
        optimizer_a.step_count();

    assert(
        saved_optimizer_steps > 0
    );

    // ========================================================
    // SAVE
    // ========================================================

    trainer_a.save_checkpoint(
        checkpoint_path
    );

    // ========================================================
    // TRAINER B
    // ========================================================

    LanguageModel model_b(
        16,
        8,
        8,
        2,
        16,
        1
    );

    Adam optimizer_b(
        0.001f
    );

    optimizer_b.add_parameters(
        model_b.parameters()
    );

    TrainerConfig config_b;

    config_b.epochs = 2;
    config_b.batch_size = 2;
    config_b.gradient_accumulation_steps = 1;

    Trainer trainer_b(
        model_b,
        optimizer_b,
        config_b
    );

    // ========================================================
    // LOAD
    // ========================================================

    trainer_b.load_checkpoint(
        checkpoint_path
    );

    assert(
        trainer_b.current_epoch() ==
        saved_epoch
    );

    assert(
        trainer_b.global_step() ==
        saved_global_step
    );

    assert(
        optimizer_b.step_count() ==
        saved_optimizer_steps
    );

    assert(
        std::fabs(
            optimizer_b.learning_rate() -
            optimizer_a.learning_rate()
        ) < 1e-12f
    );

    // ========================================================
    // RESUME
    // ========================================================

    TrainingMetrics resumed_metrics =
        trainer_b.fit(
            dataset
        );

    // Trainer B must continue from epoch 1
    // and finish at epoch 2.
    assert(
        trainer_b.current_epoch() == 2
    );

    // Global step must continue increasing,
    // never reset to zero.
    assert(
        trainer_b.global_step() >
        saved_global_step
    );

    assert(
        resumed_metrics.global_step ==
        trainer_b.global_step()
    );

    // Optimizer state must also continue.
    assert(
        optimizer_b.step_count() >
        saved_optimizer_steps
    );

    assert(
        std::isfinite(
            resumed_metrics.loss
        )
    );

    // ========================================================
    // CLEANUP
    // ========================================================

    std::remove(
        checkpoint_path
    );

    std::cout
        << "[OK] Trainer A completed epoch 1"
        << std::endl;

    std::cout
        << "[OK] checkpoint saved"
        << std::endl;

    std::cout
        << "[OK] Trainer B restored epoch = "
        << saved_epoch
        << std::endl;

    std::cout
        << "[OK] global_step restored = "
        << saved_global_step
        << std::endl;

    std::cout
        << "[OK] optimizer steps restored = "
        << saved_optimizer_steps
        << std::endl;

    std::cout
        << "[OK] training resumed"
        << std::endl;

    std::cout
        << "[OK] resumed training reached epoch = "
        << trainer_b.current_epoch()
        << std::endl;

    std::cout
        << "[OK] global_step continued = "
        << trainer_b.global_step()
        << std::endl;

    std::cout
        << "[OK] optimizer steps continued = "
        << optimizer_b.step_count()
        << std::endl;
}

} // namespace

int main() {

    std::cout
        << "============================================================"
        << std::endl;

    std::cout
        << " VENLACPU 2.3.0 — TRAINER RESUME TEST"
        << std::endl;

    std::cout
        << "============================================================"
        << std::endl;

    test_trainer_resume();

    std::cout
        << std::endl
        << "============================================================"
        << std::endl;

    std::cout
        << " TRAINER RESUME TEST PASSED"
        << std::endl;

    std::cout
        << "============================================================"
        << std::endl;

    return 0;
}
