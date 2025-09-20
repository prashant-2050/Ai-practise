---
title: LLM Demo - Lightweight Transformer
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 Lightweight LLM Demo

A custom-built transformer model trained from scratch! This demo showcases a lightweight Language Model implemented in PyTorch with a clean, educational codebase.

## ✨ Features

- **Custom Transformer**: Built from scratch with attention mechanisms
- **Multiple Model Sizes**: Nano (22M), Micro (30M), Small (51M) parameters
- **Interactive Generation**: Real-time text generation with customizable parameters
- **Educational**: Well-documented code with mathematical explanations

## 🚀 Try It Out

Use the interface below to generate text! The model was trained on Shakespeare's works, so it works best with classical or dramatic prompts.

### 📝 Example Prompts
- "To be or not to be"
- "Once upon a time"
- "The king said"
- "In fair Verona"

### 🎛️ Generation Controls
- **Temperature**: Controls randomness (0.1 = focused, 1.0 = creative)
- **Top-k**: Limits to top K most likely tokens
- **Top-p**: Nucleus sampling threshold
- **Max Tokens**: Length of generated text

## 🏗️ Architecture

This model implements a GPT-2 style transformer with:
- **Causal Self-Attention**: Ensures autoregressive generation
- **Multi-Layer Perceptron**: Feed-forward processing
- **Layer Normalization**: Stable training
- **Positional Embeddings**: Sequence understanding
- **Residual Connections**: Deep network training

## 📊 Model Details

| Model | Parameters | Layers | Attention Heads | Embedding Dim |
|-------|------------|---------|-----------------|---------------|
| Nano  | 22M        | 6       | 6               | 384           |
| Micro | 30M        | 8       | 6               | 384           |
| Small | 51M        | 12      | 12              | 768           |

## 🔧 Technical Implementation

- **Framework**: PyTorch 2.0+
- **Training**: AdamW optimizer with cosine scheduling
- **Dataset**: Shakespeare corpus with custom tokenization
- **Hardware**: Optimized for Apple Silicon MPS and CUDA

## 📚 Educational Resources

- [Complete Mathematical Guide](README-transformer-math.md)
- [Deployment Options](README-deployment.md)
- [GitHub Actions Setup](README-github-actions.md)

## 🛠️ Local Development

```bash
# Clone repository
git clone https://github.com/prashant-2050/Ai-practise.git
cd Ai-practise

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Generate text
python generate.py --prompt "Your prompt here"

# Run web interface
python gradio_app.py
```

## 🚀 Deployment

This project includes automated deployment via GitHub Actions:
- **Push to main** → Automatic deployment to Hugging Face Spaces
- **Manual deployment** options for Railway, Render, and more
- **Google Colab** notebook for cloud training

## 📄 License

MIT License - Feel free to use this code for learning and experimentation!

---

**Built with ❤️ for AI education and experimentation**
