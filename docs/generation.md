# Generation

Text generation uses the native generation engine.

    config = venlacpu.GenerationConfig()
    config.max_new_tokens = 20
    config.temperature = 0.0
    result = venlacpu.generate(model, venlacpu.tensor(prompt_ids), config)

Temperature 0.0 requests greedy generation.
