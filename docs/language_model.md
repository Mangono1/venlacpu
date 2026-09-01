# Language Model

VENLACPU provides a native causal language model.

    import venlacpu
    model = venlacpu.LanguageModel(vocab_size=49, max_seq_len=128, embed_dim=64, num_heads=2, hidden_dim=128, num_layers=2)
    logits = model.forward(venlacpu.tensor([1, 2, 3]))
    print(logits.shape())
