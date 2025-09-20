# The Transformer Model: Mathematical Deep Dive

## Mathematical Foundations and Implementation Details

This document extends the basic transformer explanation with comprehensive mathematical formulations and their implementations in our code.

---

## 1. Mathematical Notation and Core Concepts

### 📐 Basic Mathematical Symbols Used

| Symbol | Meaning | In Our Code |
|--------|---------|-------------|
| **d_model** | Model dimension (embedding size) | `config.n_embd = 384` |
| **d_k, d_v** | Key/Value dimensions | `C // self.n_head = 64` |
| **n_h** | Number of attention heads | `config.n_head = 6` |
| **L** | Number of layers | `config.n_layer = 6` |
| **T** | Sequence length | `T` in our code |
| **V** | Vocabulary size | `config.vocab_size = 50257` |
| **⊕** | Element-wise addition | `+` operator |
| **⊙** | Element-wise multiplication | `*` operator |
| **||** | Concatenation | `torch.cat()` |
| **σ** | Activation function | `F.softmax()`, `F.gelu()` |

---

## 2. Input Representations: Mathematical Formulation

### 🔢 Token Embedding Matrix

**Mathematical Definition**:
```
E ∈ ℝ^(V × d_model)
where V = vocabulary size, d_model = embedding dimension
```

**In Our Implementation**:
```python
# light_llm.py
self.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)
# Creates matrix: E ∈ ℝ^(50257 × 384)
```

**Token Lookup Operation**:
```
x_i = E[token_id_i]
where x_i ∈ ℝ^d_model
```

**Code Example**:
```python
# For token_id = 464 ("The")
tok_emb = self.transformer.wte(idx)  # idx[0] = 464
# Result: tok_emb[0] = E[464] ∈ ℝ^384
```

### 📍 Positional Encoding Matrix

**Mathematical Definition**:
```
P ∈ ℝ^(T_max × d_model)
where T_max = maximum sequence length
```

**Position Embedding Lookup**:
```
p_i = P[i] for position i
```

**In Our Code**:
```python
self.transformer.wpe = nn.Embedding(config.n_positions, config.n_embd)
# Creates: P ∈ ℝ^(1024 × 384)

pos = torch.arange(0, t, dtype=torch.long, device=device)
pos_emb = self.transformer.wpe(pos)  # p_i = P[i]
```

### 🔄 Input Combination

**Mathematical Formula**:
```
X = E_tokens + P_positions
where X ∈ ℝ^(T × d_model)
```

**Implementation**:
```python
x = self.transformer.drop(tok_emb + pos_emb)
# X[i] = E[token_i] + P[i] ∈ ℝ^384
```

---

## 3. Self-Attention: Complete Mathematical Framework

### 🧮 Query, Key, Value Projections

**Mathematical Formulation**:
```
Q = XW^Q,  where W^Q ∈ ℝ^(d_model × d_k)
K = XW^K,  where W^K ∈ ℝ^(d_model × d_k)  
V = XW^V,  where W^V ∈ ℝ^(d_model × d_v)

Typically: d_k = d_v = d_model / n_h
```

**In Our Implementation**:
```python
# Single linear layer produces all three matrices
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
# W ∈ ℝ^(384 × 1152), splits into 3 × ℝ^(384 × 384)

qkv = self.c_attn(x)  # ℝ^(B×T×384) → ℝ^(B×T×1152)
q, k, v = qkv.split(self.n_embd, dim=2)  # Each: ℝ^(B×T×384)
```

### 🔀 Multi-Head Reshaping

**Mathematical Transformation**:
```
Q_h = reshape(Q, [B, T, n_h, d_k]) 
K_h = reshape(K, [B, T, n_h, d_k])
V_h = reshape(V, [B, T, n_h, d_v])

Then transpose to: [B, n_h, T, d_k]
```

**Code Implementation**:
```python
# Reshape: [B, T, 384] → [B, T, 6, 64] → [B, 6, T, 64]
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

### ⚡ Attention Score Computation

**Mathematical Formula**:
```
A = softmax(QK^T / √d_k) ∈ ℝ^(T × T)

Where:
- QK^T creates compatibility scores
- √d_k provides scale normalization
- softmax converts to probabilities
```

**Detailed Breakdown**:
```python
# Step 1: Compute raw scores
att = q @ k.transpose(-2, -1)  # [B, n_h, T, T]
# att[i,j] = Σ(Q[i] ⊙ K[j]) = compatibility between positions i,j

# Step 2: Scale normalization  
att = att * (1.0 / math.sqrt(k.size(-1)))  # Divide by √d_k = √64 = 8
# Prevents softmax saturation for large d_k
```

**Why √d_k Scaling?**
- **Problem**: For large d_k, dot products grow large → softmax saturates → gradients vanish
- **Solution**: Scale by √d_k keeps dot products in reasonable range
- **Mathematical Intuition**: If Q,K ~ N(0,1), then QK^T ~ N(0, d_k), so QK^T/√d_k ~ N(0,1)

### 🎭 Causal Masking

**Mathematical Definition**:
```
M ∈ ℝ^(T × T) where M[i,j] = {
    0 if i < j (future positions)
    1 if i ≥ j (current and past positions)
}

A_masked = A ⊙ M + (1-M) × (-∞)
```

**Implementation**:
```python
# Create lower triangular mask
self.register_buffer("bias", torch.tril(torch.ones(n_positions, n_positions)))

# Apply mask (set future positions to -inf)
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
```

**Mask Visualization**:
```
For sequence "The cat sat":
       The  cat  sat
The  [  1,   0,   0 ]  ← "The" can only see itself
cat  [  1,   1,   0 ]  ← "cat" can see "The" and itself  
sat  [  1,   1,   1 ]  ← "sat" can see all previous tokens
```

### 🎯 Attention Application

**Mathematical Formula**:
```
Output = Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

**Step-by-Step**:
```python
# Step 1: Softmax normalization
att = F.softmax(att, dim=-1)  # Each row sums to 1
# att[i] = [p_i1, p_i2, ..., p_iT] where Σp_ij = 1

# Step 2: Weighted combination of values
y = att @ v  # [B, n_h, T, d_v]
# y[i] = Σ(att[i,j] * V[j]) = weighted average of all values
```

### 🔧 Multi-Head Concatenation

**Mathematical Operation**:
```
MultiHead(Q,K,V) = Concat(head_1, head_2, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Implementation**:
```python
# Concatenate heads: [B, n_h, T, d_v] → [B, T, n_h × d_v] = [B, T, d_model]
y = y.transpose(1, 2).contiguous().view(B, T, C)

# Output projection
y = self.resid_dropout(self.c_proj(y))  # W^O ∈ ℝ^(d_model × d_model)
```

---

## 4. Feed Forward Network: Mathematical Analysis

### 🧠 MLP Architecture

**Mathematical Definition**:
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
or in our case:
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

Where:
W_1 ∈ ℝ^(d_model × d_ff), typically d_ff = 4 × d_model
W_2 ∈ ℝ^(d_ff × d_model)
```

**In Our Implementation**:
```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # W_1: 384→1536
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # W_2: 1536→384

    def forward(self, x):
        x = self.c_fc(x)        # x ∈ ℝ^(B×T×384) → ℝ^(B×T×1536)
        x = F.gelu(x)           # Apply GELU activation
        x = self.c_proj(x)      # ℝ^(B×T×1536) → ℝ^(B×T×384)
        return x
```

### 📊 GELU Activation Function

**Mathematical Definition**:
```
GELU(x) = x · Φ(x) = x · P(X ≤ x), where X ~ N(0,1)

Approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

**Why GELU over ReLU?**
- **Smooth**: Differentiable everywhere (better gradients)
- **Probabilistic**: Based on normal distribution
- **Non-monotonic**: Can suppress less important features

**Implementation Detail**:
```python
x = F.gelu(x)  # PyTorch provides optimized GELU implementation
```

---

## 5. Layer Normalization: Mathematical Foundation

### 📏 LayerNorm Formula

**Mathematical Definition**:
```
LayerNorm(x) = γ ⊙ (x - μ)/σ + β

Where for each sample in batch:
μ = (1/d_model) Σ x_i        (mean across features)
σ = √((1/d_model) Σ (x_i - μ)²)  (standard deviation)
γ, β ∈ ℝ^d_model are learned parameters
```

**Implementation**:
```python
self.ln_1 = nn.LayerNorm(config.n_embd)  # γ, β ∈ ℝ^384

# In forward pass:
# For each position in sequence, normalize across the 384 dimensions
normalized = self.ln_1(x)  # x ∈ ℝ^(B×T×384) → normalized ∈ ℝ^(B×T×384)
```

**Why LayerNorm?**
- **Stability**: Keeps activations in reasonable range
- **Speed**: Reduces internal covariate shift
- **Independence**: Each sample normalized independently

---

## 6. Residual Connections: Mathematical Analysis

### 🔄 Residual Formula

**Mathematical Definition**:
```
y = x + F(x)
where F(x) is the transformation (attention or MLP)
```

**In Our Transformer Block**:
```python
def forward(self, x):
    # First residual connection
    x = x + self.attn(self.ln_1(x))    # x_new = x_old + Attention(LN(x_old))
    
    # Second residual connection  
    x = x + self.mlp(self.ln_2(x))     # x_final = x_new + MLP(LN(x_new))
    return x
```

**Mathematical Benefits**:
1. **Gradient Flow**: ∇x = ∇y(1 + ∇F(x)) ≥ ∇y (gradients can't vanish)
2. **Identity Mapping**: If F(x) = 0, then y = x (preserve input)
3. **Easier Optimization**: Model can learn incremental improvements

---

## 7. Training Mathematics: Loss and Optimization

### 📉 Cross-Entropy Loss

**Mathematical Definition**:
```
L = -Σ y_true[i] × log(y_pred[i])

For language modeling:
L = -(1/T) Σ log(P(x_{t+1} | x_1, ..., x_t))
```

**Implementation**:
```python
# Compute logits for all positions and vocabulary
logits = self.lm_head(x)  # [B, T, V] = [4, 1023, 50257]

# Cross-entropy between predictions and targets
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),  # Flatten: [B×T, V]
    targets.view(-1),                   # Flatten: [B×T]
    ignore_index=-1
)
```

**Detailed Cross-Entropy Calculation**:
```python
# For each position t, we have:
# logits[t] ∈ ℝ^V (raw scores for each vocabulary token)
# target[t] ∈ {0, 1, ..., V-1} (true next token)

# Convert to probabilities
probs = F.softmax(logits[t], dim=-1)  # Σ probs[i] = 1

# Cross-entropy for this position
loss_t = -log(probs[target[t]])
```

### 🎯 AdamW Optimizer Mathematics

**Adam Update Rule**:
```
m_t = β₁m_{t-1} + (1-β₁)g_t           (momentum)
v_t = β₂v_{t-1} + (1-β₂)g_t²          (second moment)
m̂_t = m_t / (1-β₁^t)                  (bias correction)
v̂_t = v_t / (1-β₂^t)                  (bias correction)
θ_t = θ_{t-1} - α(m̂_t/(√v̂_t + ε) + λθ_{t-1})
```

**Where**:
- g_t = gradient at step t
- β₁ = 0.9 (momentum decay)
- β₂ = 0.95 (second moment decay)  
- α = learning rate
- λ = weight decay coefficient
- ε = 1e-8 (numerical stability)

**Implementation**:
```python
self.optimizer = torch.optim.AdamW(
    optim_groups, 
    lr=learning_rate,           # α
    betas=(0.9, 0.95),         # β₁, β₂
    weight_decay=weight_decay   # λ
)
```

### 📈 Learning Rate Scheduling

**Cosine Annealing with Warmup**:
```
lr(t) = {
    lr_max × (t/t_warmup)                           if t ≤ t_warmup
    lr_max × 0.5(1 + cos(π(t-t_warmup)/(t_max-t_warmup)))  if t > t_warmup
}
```

**Implementation**:
```python
def lr_lambda(step):
    if step < self.warmup_steps:
        return step / self.warmup_steps  # Linear warmup
    else:
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine decay
        return coeff
```

---

## 8. Text Generation: Probabilistic Sampling

### 🎲 Temperature Sampling

**Mathematical Formula**:
```
P_temp(x_i) = exp(logits_i / τ) / Σ exp(logits_j / τ)

Where τ is temperature:
τ → 0: Deterministic (argmax)
τ = 1: Standard softmax  
τ → ∞: Uniform sampling
```

**Implementation**:
```python
logits = logits / temperature  # Scale by temperature
probs = F.softmax(logits, dim=-1)  # Convert to probabilities
next_token = torch.multinomial(probs, num_samples=1)  # Sample
```

### 🔝 Top-K Sampling

**Mathematical Definition**:
```
P_topk(x_i) = {
    P(x_i) / Z  if x_i ∈ TopK(logits)
    0           otherwise
}
Where Z = Σ P(x_j) for x_j ∈ TopK(logits)
```

**Implementation**:
```python
# Keep only top-k logits, set others to -infinity
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')
```

### 🎯 Top-P (Nucleus) Sampling

**Mathematical Definition**:
```
TopP(p) = {x_i : Σ P(x_j) ≤ p for j ∈ sorted_indices[0:i]}
```

**Algorithm**:
1. Sort probabilities in descending order
2. Find smallest set where cumulative probability ≥ p
3. Sample only from this set

**Implementation**:
```python
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

# Remove tokens with cumulative probability above threshold
sorted_indices_to_remove = cumulative_probs > top_p
# Keep at least the first token
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = 0
```

---

## 9. Computational Complexity Analysis

### ⏱️ Time Complexity

**Self-Attention**:
```
O(T² × d_model + T × d_model²)
= O(T²d + Td²)

Where T = sequence length, d = model dimension
```

**Feed Forward**:
```
O(T × d_model × d_ff) = O(T × d²) 
(since d_ff = 4 × d_model)
```

**Total per Layer**:
```
O(T²d + Td²)
```

**Full Model** (L layers):
```
O(L × (T²d + Td²))
```

### 💾 Memory Complexity

**Attention Matrices**:
```
O(n_h × T²) for attention scores
= O(6 × 1024²) ≈ 6M parameters for our model
```

**Activations**:
```
O(L × T × d_model) for storing intermediate activations
= O(6 × 1024 × 384) ≈ 2.4M parameters
```

**Gradients** (during training):
```
O(model_parameters) = O(30M) for backward pass storage
```

---

## 10. Numerical Stability Considerations

### 🔢 Gradient Clipping

**Mathematical Motivation**:
```
If ||∇|| > threshold:
    ∇_clipped = ∇ × (threshold / ||∇||)
```

**Implementation**:
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
```

### 📊 Weight Initialization

**Xavier/Glorot Initialization**:
```
W ~ N(0, 2/(fan_in + fan_out))
```

**Our Implementation**:
```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
```

### ⚖️ Numerical Precision

**Mixed Precision Training** (ready for implementation):
```python
# Use torch.cuda.amp.autocast() for FP16 computation
with torch.cuda.amp.autocast():
    logits, loss = model(inputs, targets)
```

---

## Summary: Mathematical Architecture

Our transformer implements the complete mathematical framework:

**Forward Pass**:
```
X₀ = TokenEmb(tokens) + PosEmb(positions)
For l = 1 to L:
    X_l = X_{l-1} + MultiHeadAttention(LayerNorm(X_{l-1}))
    X_l = X_l + FFN(LayerNorm(X_l))
Output = Softmax(X_L × W_vocab)
```

**Training**:
```
Loss = CrossEntropy(Output, Targets)
Gradients = ∇_θ Loss
θ_new = AdamW(θ_old, Gradients)
```

**Generation**:
```
For t = 1 to max_length:
    P(x_t | x_{<t}) = Transformer(x_{<t})
    x_t ~ Sample(P, temperature, top_k, top_p)
```

This mathematical foundation enables our 30M parameter model to learn language patterns and generate coherent text through the elegant interplay of linear algebra, probability theory, and optimization mathematics! 🧮✨