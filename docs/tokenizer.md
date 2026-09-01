# Tokenizer

VENLACPU provides native tokenization.

    import venlacpu
    text = "Bima menjaga keris pusaka."
    tok = venlacpu.Tokenizer()
    tok.train(text)
    ids = tok.encode("Bima menjaga")
    print(ids)
    print(tok.decode(ids))

## Prefix Stability

VENLACPU 2.0.0 introduces prefix-stable tokenization required by causal generation.

The token IDs produced for an existing prefix remain unchanged when a continuation is appended.

The 2.0.0 audit tested 7 prefix cases.

PASS: 7
FAIL: 0

Therefore tokenizer prefix stability passes.
