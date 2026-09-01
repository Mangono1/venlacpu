# Tutorial 04 - Generate Text

Use Tokenizer to encode a prompt, LanguageModel to produce logits, and generate() to produce new token IDs.

    result = venlacpu.generate(model, venlacpu.tensor(prompt_ids), config)
    print(tokenizer.decode(result.tolist()))
