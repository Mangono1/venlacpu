# Tutorial 02 - Tokenizer

    import venlacpu
    tok = venlacpu.Tokenizer()
    tok.train("Bima menjaga keris pusaka. Keris itu sangat sakti.")
    ids = tok.encode("Bima menjaga")
    print(ids)
    print(tok.decode(ids))

VENLACPU 2.1.0 guarantees prefix stability for causal generation.
