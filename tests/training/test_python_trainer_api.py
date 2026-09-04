import gc
import math

import venlacpu


def make_model_optimizer():
    model = venlacpu.LanguageModel(
        16, 8, 4, 2, 8, 1, True
    )

    optimizer = venlacpu.Adam(0.001)
    optimizer.add_parameters(model.parameters())

    return model, optimizer


def make_dataset():
    dataset = venlacpu.CausalLMDataset(0, -100)

    dataset.add_sequence([1, 2, 3, 4, 5])
    dataset.add_sequence([2, 3, 4, 5, 6])

    return dataset


def test_trainer_keeps_dependencies_alive():
    model, optimizer = make_model_optimizer()

    config = venlacpu.TrainerConfig()
    config.epochs = 1
    config.batch_size = 1

    trainer = venlacpu.Trainer(model, optimizer, config)

    del model
    del optimizer
    gc.collect()

    # Trainer must still own the lifetime of its dependencies.
    assert trainer.current_epoch() == 0
    assert trainer.global_step() == 0

    model_from_trainer = trainer.model()
    optimizer_from_trainer = trainer.optimizer()

    assert model_from_trainer is not None
    assert optimizer_from_trainer is not None

    print("[OK] Trainer keeps model alive")
    print("[OK] Trainer keeps optimizer alive")
    print("[OK] trainer.model() remains valid")
    print("[OK] trainer.optimizer() remains valid")


def test_trainer_config_binding():
    config = venlacpu.TrainerConfig()

    config.epochs = 3
    config.batch_size = 2
    config.gradient_accumulation_steps = 4
    config.drop_last = True
    config.log_every = 5
    config.ignore_index = -100
    config.max_grad_norm = 1.5
    config.evaluate_each_epoch = True
    config.early_stopping = True
    config.early_stopping_patience = 7
    config.early_stopping_min_delta = 0.01
    config.keep_best_model = True

    assert config.epochs == 3
    assert config.batch_size == 2
    assert config.gradient_accumulation_steps == 4
    assert config.drop_last is True
    assert config.log_every == 5
    assert config.ignore_index == -100
    assert math.isclose(config.max_grad_norm, 1.5)
    assert config.evaluate_each_epoch is True
    assert config.early_stopping is True
    assert config.early_stopping_patience == 7
    assert math.isclose(
        config.early_stopping_min_delta,
        0.01,
        rel_tol=1e-6,
        abs_tol=1e-7,
    )
    assert config.keep_best_model is True

    print("[OK] TrainerConfig exposes all 12 fields")


def test_training_history_binding():
    model, optimizer = make_model_optimizer()

    config = venlacpu.TrainerConfig()
    config.epochs = 2
    config.batch_size = 1
    config.gradient_accumulation_steps = 1
    config.log_every = 1

    trainer = venlacpu.Trainer(model, optimizer, config)
    dataset = make_dataset()

    history = trainer.history()

    assert isinstance(history, venlacpu.TrainingHistory)
    assert history.empty()
    assert history.size() == 0

    trainer.fit(dataset)

    assert history.size() == 2
    assert not history.empty()

    records = history.records()

    assert len(records) == 2

    first = history.at(0)
    second = history.at(1)

    assert first.epoch == 1
    assert second.epoch == 2

    # Verify every TrainingMetrics field exposed by Python.
    metric_fields = [
        "loss",
        "eval_loss",
        "perplexity",
        "eval_perplexity",
        "loss_reduction",
        "learning_rate",
        "tokens",
        "batches",
        "evaluated_tokens",
        "correct_tokens",
        "next_token_accuracy",
        "optimizer_steps",
        "epoch",
        "global_step",
        "epoch_seconds",
        "tokens_per_second",
        "batches_per_second",
        "gradient_norm",
        "clipped_gradient_norm",
        "gradient_clipped",
        "gradient_finite",
        "loss_finite",
        "is_best",
        "early_stopped",
        "bad_epochs",
    ]

    for field in metric_fields:
        getattr(first, field)
        getattr(second, field)

    print("[OK] TrainingHistory binding")
    print("[OK] history.size() =", history.size())
    print("[OK] history.records() =", len(records))
    print("[OK] history.at() works")
    print("[OK] TrainingMetrics exposes all 25 fields")


def main():
    print()
    print("=" * 72)
    print(" VENLACPU 3.0.0 — PYTHON TRAINER API REGRESSION TEST")
    print("=" * 72)

    test_trainer_keeps_dependencies_alive()
    test_trainer_config_binding()
    test_training_history_binding()

    print()
    print("PYTHON TRAINER API REGRESSION TEST PASSED")
    print()


if __name__ == "__main__":
    main()
