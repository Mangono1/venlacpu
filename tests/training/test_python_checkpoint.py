import os
import sys
import tempfile

import venlacpu


def make_dataset():
    dataset = venlacpu.CausalLMDataset()

    dataset.add_sequence([1, 2, 3, 4, 5, 6])
    dataset.add_sequence([2, 3, 4, 5, 6, 7])
    dataset.add_sequence([3, 4, 5, 6, 7, 8])
    dataset.add_sequence([4, 5, 6, 7, 8, 9])

    return dataset


def main():
    print("=" * 60)
    print(" VENLACPU 2.3.1 — PYTHON CHECKPOINT API TEST")
    print("=" * 60)

    dataset = make_dataset()

    # --------------------------------------------------------
    # Trainer A
    # --------------------------------------------------------

    model_a = venlacpu.LanguageModel(
        16,
        8,
        8,
        2,
        16,
        1
    )

    optimizer_a = venlacpu.Adam(
        0.001
    )

    optimizer_a.add_parameters(
        model_a.parameters()
    )

    config_a = venlacpu.TrainerConfig()
    config_a.epochs = 1
    config_a.batch_size = 2

    trainer_a = venlacpu.Trainer(
        model_a,
        optimizer_a,
        config_a
    )

    metrics_a = trainer_a.fit(dataset)

    assert trainer_a.current_epoch() == 1
    assert trainer_a.global_step() > 0

    saved_epoch = trainer_a.current_epoch()
    saved_step = trainer_a.global_step()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fd, checkpoint = tempfile.mkstemp(
        prefix="venlacpu_",
        suffix=".checkpoint"
    )

    os.close(fd)

    try:
        trainer_a.save_checkpoint(checkpoint)

        assert os.path.exists(checkpoint)

        print("[OK] Python save_checkpoint()")

        # ----------------------------------------------------
        # Trainer B
        # ----------------------------------------------------

        model_b = venlacpu.LanguageModel(
            16,
            8,
            8,
            2,
            16,
            1
        )

        optimizer_b = venlacpu.Adam(
            0.001
        )

        optimizer_b.add_parameters(
            model_b.parameters()
        )

        config_b = venlacpu.TrainerConfig()
        config_b.epochs = 2
        config_b.batch_size = 2

        trainer_b = venlacpu.Trainer(
            model_b,
            optimizer_b,
            config_b
        )

        trainer_b.load_checkpoint(checkpoint)

        assert trainer_b.current_epoch() == saved_epoch
        assert trainer_b.global_step() == saved_step

        print("[OK] Python load_checkpoint()")
        print("[OK] Python epoch restored =", trainer_b.current_epoch())
        print("[OK] Python global_step restored =", trainer_b.global_step())

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        resumed = trainer_b.fit(dataset)

        assert trainer_b.current_epoch() == 2
        assert trainer_b.global_step() > saved_step
        assert resumed.global_step == trainer_b.global_step()

        print("[OK] Python resume training")
        print("[OK] resumed epoch =", trainer_b.current_epoch())
        print("[OK] resumed global_step =", trainer_b.global_step())

    finally:
        if os.path.exists(checkpoint):
            os.remove(checkpoint)

    print()
    print("=" * 60)
    print(" PYTHON CHECKPOINT API TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
