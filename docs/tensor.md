Tensor
Tensor is the fundamental numerical data structure in VENLACPU.
Basic construction:
import venlacpu

x = venlacpu.tensor([1, 2, 3])

print(x.shape())
print(x.dtype())
print(x.device())
print(x.numel())
print(x.tolist())
VENLACPU tensors are CPU-first and backed by the native C++17 engine. EOF
cat > docs/tokenizer.md <<'EOF'
Tokenizer
VENLACPU provides native tokenization.
import venlacpu

text = "Bima menjaga keris pusaka."

tok = venlacpu.Tokenizer()
tok.train(text)

ids = tok.encode("Bima menjaga")
decoded = tok.decode(ids)

print(ids)
print(decoded)
Prefix Stability
VENLACPU 2.0.0 introduces prefix-stable tokenization behavior required by causal generation.
For example:
encode("Bima")
must remain a prefix of:
encode("Bima menjaga")
This property is important because causal generation extends an already-tokenized prompt one token at a time. EOF
cat > docs/autograd.md <<'EOF'
Autograd
VENLACPU provides automatic differentiation through its native computation system.
The autograd subsystem supports differentiable tensor operations and gradient propagation used by neural network training.
The optimizer consumes model parameters and their gradients during training. EOF
cat > docs/neural_networks.md <<'EOF'
Neural Networks
VENLACPU includes native neural-network building blocks including:
Linear
Activation
Sequential
Embedding
LayerNorm
Feed Forward
MSE Loss
Cross Entropy Loss
Positional Encoding
Example:
import venlacpu

model = venlacpu.LanguageModel(
    vocab_size=100,
    max_seq_len=128,
    embed_dim=64,
    num_heads=2,
    hidden_dim=128,
    num_layers=2,
)

print(len(model.parameters()))
