import os
import tempfile

import venlacpu


def test_tensor_repr():
    tensor = venlacpu.Tensor.zeros(
        venlacpu.Shape([2, 3]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    text = repr(tensor)

    assert "Tensor" in text


def test_tokenizer_roundtrip():
    tokenizer = venlacpu.BPETokenizer()

    tokenizer.train(
        "halo dunia halo dunia"
    )

    ids = tokenizer.encode(
        "halo dunia"
    )

    assert len(ids) > 0

    decoded = tokenizer.decode(ids)

    assert isinstance(decoded, str)


def test_tokenizer_persistence():
    tokenizer = venlacpu.BPETokenizer()

    tokenizer.train(
        "ini adalah contoh tokenizer"
    )

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(
            directory,
            "tokenizer.bin",
        )

        tokenizer.save(path)

        loaded = (
            venlacpu.BPETokenizer.load(path)
        )

        assert loaded.trained()
        assert (
            loaded.vocab_size()
            == tokenizer.vocab_size()
        )


def test_model_weights_persistence():
    model = venlacpu.LanguageModel(
        32,
        16,
        16,
        4,
        32,
        1,
        True,
    )

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(
            directory,
            "model.bin",
        )

        model.save_weights(path)

        assert os.path.exists(path)

        model.load_weights(path)


def test_generate_text_api_exists():
    assert hasattr(
        venlacpu,
        "generate_text",
    )


# v3.0.0 persistence regression
def test_weights_persistence_regression():
    import os
    import tempfile

    model_a = venlacpu.LanguageModel(
        16,
        16,
        8,
        2,
        16,
        1,
        True,
    )

    prompt = venlacpu.tensor_int32(
        [1, 2, 3]
    )

    config = venlacpu.GenerationConfig()

    config.max_new_tokens = 5
    config.temperature = 0.0
    config.eos_token_id = -1
    config.stop_on_eos = True

    # Generate from the original model before saving.
    generated_before = venlacpu.generate(
        model_a,
        prompt,
        config,
    )

    with tempfile.TemporaryDirectory() as directory:
        weights_path = os.path.join(
            directory,
            "model_v3_weights.bin",
        )

        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------
        model_a.save_weights(
            weights_path
        )

        assert os.path.exists(
            weights_path
        )

        assert os.path.getsize(
            weights_path
        ) > 0

        # --------------------------------------------------
        # FRESH MODEL
        # --------------------------------------------------
        model_b = venlacpu.LanguageModel(
            16,
            16,
            8,
            2,
            16,
            1,
            True,
        )

        assert (
            model_b.vocab_size()
            == model_a.vocab_size()
        )

        assert (
            model_b.max_seq_len()
            == model_a.max_seq_len()
        )

        assert (
            model_b.embed_dim()
            == model_a.embed_dim()
        )

        assert (
            model_b.num_heads()
            == model_a.num_heads()
        )

        assert (
            model_b.hidden_dim()
            == model_a.hidden_dim()
        )

        assert (
            model_b.num_layers()
            == model_a.num_layers()
        )

        assert (
            model_b.has_bias()
            == model_a.has_bias()
        )

        # --------------------------------------------------
        # LOAD
        # --------------------------------------------------
        model_b.load_weights(
            weights_path
        )

        # --------------------------------------------------
        # GENERATE AFTER LOAD
        # --------------------------------------------------
        generated_after = venlacpu.generate(
            model_b,
            prompt,
            config,
        )

        assert (
            generated_before.dtype
            == venlacpu.DType.Int64
        )

        assert (
            generated_after.dtype
            == venlacpu.DType.Int64
        )

        assert (
            generated_before.numel
            == generated_after.numel
        )

        before = (
            generated_before.tolist()
        )

        after = (
            generated_after.tolist()
        )

        # Deterministic generation must produce
        # exactly the same sequence after reload.
        assert before == after

    # TemporaryDirectory guarantees that the
    # persistence artifact is removed.
    assert not os.path.exists(
        weights_path
    )
