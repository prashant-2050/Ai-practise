# Lightweight LLM Training Project

This project implements and trains a small GPT-2 style transformer model locally on macOS with MPS acceleration.

## 🧮 Mathematical Overview

Our transformer implements the complete mathematical framework for language modeling:

**Core Architecture**: `X_l = X_{l-1} + MultiHeadAttention(LN(X_{l-1})) + FFN(LN(X_l))`  
**Attention Mechanism**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`  
**Training Objective**: `L = -∑log(P(x_{t+1}|x_1...x_t))` (cross-entropy loss)  
**Model Scale**: 30M parameters, 6 layers, 384 embedding dimensions

📚 **Detailed mathematics**: [README-transformer-math.md](README-transformer-math.md)

## Quick Start

1. **Setup environment**:
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (already done if you followed PyTorch setup)
pip install -r requirements.txt
```

2. **Train a model**:
```bash
# Train on Shakespeare dataset (500 steps, ~5-10 minutes)
python train.py
```

3. **Generate text**:
```bash
# Demo generation with sample prompts
python generate.py demo

# Interactive generation
python generate.py interactive

# Evaluate model
python generate.py eval
```

## Model Sizes

The project includes three model size configurations:

- **Nano**: ~12M parameters (192 embd, 6 layers)
- **Micro**: ~30M parameters (384 embd, 6 layers) - **Default**
- **Small**: ~51M parameters (512 embd, 8 layers)

## Project Structure

```
├── model_config.py          # Model configuration classes
├── light_llm.py             # GPT-2 style transformer implementation
├── dataset.py               # Dataset preparation and loading
├── train.py                 # Training script with MPS support
├── generate.py              # Text generation and evaluation
├── requirements.txt         # Python dependencies
├── README-transformer-math.md # 📊 Mathematical deep dive with formulas
├── checkpoints/             # Model checkpoints (created during training)
└── shakespeare.txt    # Training data (downloaded automatically)
```

## Training Details

- **Architecture**: GPT-2 style transformer
- **Optimizer**: AdamW with weight decay (β₁=0.9, β₂=0.95)
- **Scheduler**: Cosine annealing with linear warmup
- **Device**: Automatic detection (MPS > CUDA > CPU)
- **Dataset**: Shakespeare corpus (~1.1M characters)
- **Context Length**: 1024 tokens (T_max)
- **Batch Size**: 4 (adjustable based on memory)

## Mathematical Foundations

Our transformer implements core mathematical concepts:

### 🧮 **Core Architecture Math**
- **Embedding**: `E ∈ ℝ^(50257 × 384)` (vocabulary × model dimension)
- **Attention**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V` with d_k = 64
- **Multi-Head**: 6 parallel attention heads, each d_k = d_model/n_heads
- **Feed Forward**: `FFN(x) = GELU(xW₁)W₂` with 4× expansion (384→1536→384)

### 📊 **Training Mathematics**
- **Loss**: Cross-entropy `L = -∑log(P(x_{t+1}|x₁...x_t))`
- **Optimization**: AdamW with momentum and adaptive learning rates
- **Regularization**: Weight decay, dropout, gradient clipping
- **Scheduling**: Cosine decay with linear warmup

### 🎯 **Generation Math**
- **Temperature Sampling**: `P(x) = exp(logits/τ)/∑exp(logits_j/τ)`
- **Top-K/Top-P**: Probability mass truncation for quality control
- **Autoregressive**: `P(sequence) = ∏P(x_t|x_{<t})`

📚 **For complete mathematical derivations and formulas, see: [README-transformer-math.md](README-transformer-math.md)**

## Example Usage

### Training a Custom Model

```python
from model_config import ModelConfig
from light_llm import LightLLM
from train import Trainer
from dataset import prepare_dataset, create_dataloader

# Create custom config
config = ModelConfig.nano()  # For faster training

# Prepare data
dataset, tokenizer = prepare_dataset("shakespeare", max_length=512)
train_loader = create_dataloader(dataset, batch_size=8)

# Create and train model
model = LightLLM(config)
trainer = Trainer(model, train_loader, max_steps=1000)
trainer.train()
```

### Text Generation

```python
from generate import TextGenerator

# Load trained model
generator = TextGenerator("checkpoints/best_model.pt")

# Generate text with mathematical sampling controls
text = generator.generate(
    prompt="To be or not to be",
    max_new_tokens=100,
    temperature=0.8,        # τ in P(x)=exp(logits/τ)/Z
    top_k=40,              # Keep top-40 highest probability tokens
    top_p=0.9,             # Nucleus sampling: cumulative prob ≤ 0.9
    repetition_penalty=1.1  # Penalize recently used tokens
)
print(text)
```

## Training Tips

1. **Memory Management**: Start with nano config if you have limited memory
2. **Training Time**: Micro model takes ~10-15 minutes for 500 steps on Apple Silicon
3. **Dataset**: Shakespeare works well for testing; try TinyStories for cleaner output
4. **Mathematical Hyperparameters**: 
   - **Temperature** (0.5-0.7): Lower = more focused (sharper softmax)
   - **Temperature** (0.9-1.2): Higher = more creative (flatter distribution)
   - **Top-k/Top-p**: Balance between quality and diversity
   - **Learning Rate**: 3e-4 with cosine scheduling works well

## Model Performance

After 500 training steps on Shakespeare:
- **Cross-entropy Loss**: ~3.5-4.0 (varies by model size)
- **Perplexity**: exp(loss) ≈ 30-50 (lower is better)
- **Generation Quality**: Coherent Shakespeare-style text with some repetition
- **Convergence**: Loss typically plateaus around step 300-400

## Advanced Usage

### Experiment with Architecture Mathematics

```python
# Create custom model config with mathematical considerations
config = ModelConfig(
    n_embd=256,              # d_model: embedding dimension
    n_layer=8,               # L: number of transformer layers  
    n_head=8,                # n_h: attention heads (d_model must be divisible)
    n_positions=512,         # T_max: maximum sequence length
    dropout=0.1              # p_dropout: regularization probability
)

# Calculate model parameters: ~L×(12×d²+13×d×V) where d=n_embd, V=vocab_size
print(f"Estimated parameters: {get_model_size(config):,}")
```

### Mathematical Monitoring and Analysis

```python
# Monitor key mathematical metrics during training
def analyze_training_dynamics(model, dataloader):
    with torch.no_grad():
        # 1. Gradient norms (check for vanishing/exploding gradients)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # 2. Attention entropy (measure of attention distribution)
        # Higher entropy = more distributed attention
        
        # 3. Loss landscape analysis
        # Track loss curvature and optimization trajectory
        
        return {"grad_norm": total_norm}
```

### Monitor Training

```bash
# Use wandb for experiment tracking with mathematical metrics
# Set WANDB_API_KEY environment variable  
python train.py  # Logs loss, gradients, learning rate, attention patterns
```

## Troubleshooting

**Memory Issues** (Mathematical causes):
- **Attention matrices**: O(T²) memory scaling with sequence length
- **Gradient storage**: O(parameters) additional memory during backprop
- **Solutions**: Reduce batch size, use gradient checkpointing, shorter sequences

**Slow Training** (Computational complexity):
- **Time complexity**: O(L×T²×d) per forward pass
- **Check MPS**: `torch.backends.mps.is_available()` for Apple Silicon acceleration
- **Bottlenecks**: Attention computation dominates for long sequences

**Poor Generation Quality** (Mathematical indicators):
- **High perplexity**: exp(loss) > 100 indicates poor language modeling
- **Loss plateaus**: May need longer training or learning rate adjustment
- **Attention collapse**: All heads attending to same positions
- **Solutions**: Try different datasets, increase model size, adjust hyperparameters

## Mathematical Quick Reference

### Key Variables in Our Code
```python
# Dimensions and Architecture
B = batch_size                    # Batch dimension
T = sequence_length              # Time/sequence dimension  
C = config.n_embd               # Channel/embedding dimension (d_model)
V = config.vocab_size           # Vocabulary size
L = config.n_layer              # Number of layers
H = config.n_head               # Number of attention heads

# Mathematical Operations
q @ k.transpose(-2, -1)         # QK^T: attention scores
F.softmax(att, dim=-1)          # softmax(QK^T/√d_k): attention weights  
att @ v                         # Attention(Q,K,V) = softmax(QK^T/√d_k)V
F.cross_entropy(logits, targets) # L = -∑log(P(x_t|x_{<t}))
```

### Formula → Code Mapping
```python
# Attention Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = F.softmax(att, dim=-1)  
y = att @ v

# Layer Norm: LN(x) = γ⊙(x-μ)/σ + β
self.ln_1 = nn.LayerNorm(config.n_embd)

# Feed Forward: FFN(x) = GELU(xW₁)W₂  
x = F.gelu(self.c_fc(x))
x = self.c_proj(x)
```

📚 **For complete mathematical derivations**: See [README-transformer-math.md](README-transformer-math.md)

## Next Steps

1. **Experiment with larger models** (if you have more memory)
2. **Try different datasets** (TinyStories, WikiText, custom data)
3. **Implement advanced features**:
   - Rotary positional embeddings
   - Flash attention
   - Model parallelism
4. **Fine-tuning on specific tasks**
5. **Export to ONNX or other formats for deployment**

## References

- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Inspiration for this implementation
- [PyTorch MPS Guide](https://pytorch.org/docs/stable/notes/mps.html)