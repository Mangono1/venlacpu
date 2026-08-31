import venlacpu


def test_package_version():
    assert venlacpu.__version__ == "0.1.0"


def test_native_available():
    assert venlacpu.native_available()


def test_tensor():
    native = venlacpu._native

    tensor = native.ones(
        native.Shape([2, 3]),
        native.DType.Float32,
        native.Device.cpu(),
    )

    assert tensor.shape().dimensions() == [2, 3]
    assert tensor.numel() == 6
    assert tensor.dtype() == native.DType.Float32


def test_causal_dataset():
    native = venlacpu._native

    dataset = native.CausalLMDataset()

    dataset.add_sequence(
        [1, 2, 3, 4]
    )

    dataset.add_sequence(
        [5, 6, 7]
    )

    assert dataset.size() == 2
    assert dataset.max_sequence_length() == 4

    batch = dataset.batch(
        0,
        2
    )

    assert batch.batch_size == 2
    assert batch.sequence_length == 3
    assert batch.valid_tokens == 5


def test_language_model():
    native = venlacpu._native

    model = native.LanguageModel(
        16,
        8,
        8,
        2,
        16,
        1,
    )

    assert model.vocab_size() == 16
    assert model.max_seq_len() == 8
    assert model.embed_dim() == 8
    assert model.num_heads() == 2
    assert model.hidden_dim() == 16
    assert model.num_layers() == 1


def test_optimizer():
    native = venlacpu._native

    model = native.LanguageModel(
        16,
        8,
        8,
        2,
        16,
        1,
    )

    optimizer = native.Adam(
        0.001
    )

    native.add_model_parameters(
        optimizer,
        model,
    )

    assert optimizer.parameter_count() > 0
