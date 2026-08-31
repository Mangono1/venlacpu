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
