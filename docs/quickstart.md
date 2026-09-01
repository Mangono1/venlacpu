# Quickstart

    import venlacpu

    x = venlacpu.tensor([1, 2, 3])
    print(x)
    print(x.shape())
    print(x.tolist())

## Tokenizer

    tokenizer = venlacpu.Tokenizer()
    tokenizer.train("Bima menjaga keris")
    ids = tokenizer.encode("Bima menjaga")
    print(ids)
    print(tokenizer.decode(ids))

## Language Model

    model = venlacpu.LanguageModel(vocab_size=49, max_seq_len=128, embed_dim=64, num_heads=2, hidden_dim=128, num_layers=2)
