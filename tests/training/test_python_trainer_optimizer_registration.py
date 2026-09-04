import math

import venlacpu


def make_model():
    return venlacpu.LanguageModel(
        16, 8, 4, 2, 8, 1, True
    )


def make_dataset():
    dataset = venlacpu.CausalLMDataset(0, -100)

    dataset.add_sequence([1, 2, 3, 4, 5])
    dataset.add_sequence([2, 3, 4, 5, 6])

    return dataset


def parameter_values(model):
    values = []

    for parameter in model.parameters():
        values.append(parameter.tolist())

    return values


def test_trainer_automatically_registers_model_parameters():
    model = make_model()
    optimizer = venlacpu.Adam(0.001)

    model_parameters = model.parameters()

    assert len(model_parameters) > 0

    trainer = venlacpu.Trainer(model, optimizer)

    assert optimizer.parameter_count() == len(model_parameters)

    print(
        "[OK] Automatic parameter registration:",
        optimizer.parameter_count(),
    )


def test_trainer_does_not_duplicate_existing_registration():
    model = make_model()
    optimizer = venlacpu.Adam(0.001)

    model_parameters = model.parameters()

    optimizer.add_parameters(model_parameters)

    before = optimizer.parameter_count()

    trainer = venlacpu.Trainer(model, optimizer)

    after = optimizer.parameter_count()

    assert before == len(model_parameters)
    assert after == before

    assert trainer.optimizer().parameter_count() == before
    assert len(trainer.model().parameters()) == len(model_parameters)

    print(
        "[OK] Existing registration not duplicated:",
        after,
    )


def test_trainer_updates_model_parameters_without_manual_registration():
    model = make_model()
    optimizer = venlacpu.Adam(0.001)

    trainer = venlacpu.Trainer(model, optimizer)

    dataset = make_dataset()

    before = parameter_values(model)

    metrics = trainer.train_epoch(dataset)

    after = parameter_values(model)

    assert metrics.optimizer_steps > 0
    assert optimizer.parameter_count() == len(model.parameters())

    changed = False
    max_difference = 0.0

    for before_parameter, after_parameter in zip(
        before,
        after,
    ):
        assert len(before_parameter) == len(after_parameter)

        for before_value, after_value in zip(
            before_parameter,
            after_parameter,
        ):
            difference = abs(
                float(after_value) - float(before_value)
            )

            if difference > max_difference:
                max_difference = difference

            if not math.isclose(
                before_value,
                after_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                changed = True
                break

        if changed:
            break

    assert changed, (
        "Optimizer steps occurred, but no model parameter changed. "
        "Trainer -> Optimizer automatic registration is broken."
    )

    print("[OK] Parameters update without manual registration")
    print("[OK] Optimizer steps =", metrics.optimizer_steps)
    print("[OK] Max parameter difference =", max_difference)
    print("[OK] Parameters changed = True")


def main():
    print()
    print("=" * 72)
    print(" VENLACPU 3.0.0 — TRAINER/OPTIMIZER REGRESSION TEST")
    print("=" * 72)

    test_trainer_automatically_registers_model_parameters()
    test_trainer_does_not_duplicate_existing_registration()
    test_trainer_updates_model_parameters_without_manual_registration()

    print()
    print("TRAINER/OPTIMIZER REGRESSION TEST PASSED")
    print()


if __name__ == "__main__":
    main()
