# 🚀 Google Colab Training Guide

Train your LLM models on Google Colab with **free GPU access**!

## ⚡ Quick Start (3 minutes)

### Option 1: Use the Full Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prashant-2050/Ai-practise/blob/main/colab_training.ipynb)

### Option 2: Quick Setup in Any Colab Notebook
```python
# Run this in any Colab notebook
!wget -q https://raw.githubusercontent.com/prashant-2050/Ai-practise/main/colab_setup.py
exec(open('colab_setup.py').read())
```

## 🎯 Why Colab for Training?

| Feature | Google Colab | GitHub Actions |
|---------|-------------|----------------|
| **GPU Access** | ✅ Free Tesla T4/T4 | ❌ CPU only |
| **Memory** | ✅ 12+ GB RAM | ❌ ~7 GB |
| **Training Time** | ✅ Hours/unlimited | ❌ 2-6 hour limit |
| **Interactive** | ✅ Jupyter notebooks | ❌ Logs only |
| **Debugging** | ✅ Cell-by-cell | ❌ Limited |
| **Cost** | ✅ Free (Pro for $10/mo) | ❌ Paid for long runs |

## 🔧 Training Configurations

### Nano Model (Fast - 5 minutes)
```python
config = {
    'model_size': 'nano',
    'max_steps': 1000,
    'batch_size': 64,  # With GPU
}
# ~22M parameters, good for testing
```

### Micro Model (Recommended - 15 minutes)
```python
config = {
    'model_size': 'micro', 
    'max_steps': 3000,
    'batch_size': 64,
}
# ~30M parameters, balanced quality/speed
```

### Small Model (Best Quality - 45 minutes)
```python
config = {
    'model_size': 'small',
    'max_steps': 5000,
    'batch_size': 32,  # Larger model needs smaller batch
}
# ~51M parameters, highest quality
```

## 📊 Expected Results

| Model | Steps | Time | Final Loss | Quality |
|-------|-------|------|------------|---------|
| Nano | 1000 | 5 min | ~2.5 | Basic |
| Micro | 3000 | 15 min | ~2.0 | Good |
| Small | 5000 | 45 min | ~1.5 | Excellent |

## 🛠️ Advanced Features

### 💾 Auto-Save to Google Drive
```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Models will be saved to Drive automatically
```

### 📈 Live Training Plots
- Real-time loss visualization
- Progress tracking with ETA
- Automatic best model saving

### 🎭 Text Generation During Training
- Generate samples every 500 steps
- Test different prompts
- Save best outputs automatically

### 🔄 GitHub Integration
- Auto-upload results to GitHub
- Create training result branches
- Commit checkpoints and logs

## 🚀 Step-by-Step Workflow

1. **Open Colab Notebook**
   ```
   https://colab.research.google.com/github/prashant-2050/Ai-practise/blob/main/colab_training.ipynb
   ```

2. **Enable GPU**
   - Runtime → Change runtime type → GPU → T4

3. **Run Setup Cells**
   - Repository cloning
   - Dependencies installation
   - Data preparation

4. **Configure Training**
   - Choose model size
   - Set training steps
   - Configure batch size

5. **Start Training**
   - Watch live progress
   - See loss curves update
   - Generate text samples

6. **Download Results**
   - Trained models
   - Training logs
   - Generated samples

## 🎯 Colab Pro Benefits

Upgrade to Colab Pro ($10/month) for:
- **A100 GPUs** (much faster)
- **Longer runtimes** (24+ hours)
- **More memory** (51 GB RAM)
- **Priority access** to resources

## 💡 Tips & Tricks

### Memory Optimization
```python
# Clear GPU memory between runs
torch.cuda.empty_cache()

# Use gradient checkpointing for larger models
config['gradient_checkpointing'] = True
```

### Better Generation
```python
# Experiment with generation parameters
generator.generate(
    prompt="To be or not to be",
    max_tokens=200,
    temperature=0.8,  # Lower = more focused
    top_k=50,         # Top-k sampling
    top_p=0.9         # Nucleus sampling
)
```

### Monitoring Training
```python
# Use Weights & Biases for experiment tracking
import wandb
wandb.login()  # Enter your API key

config['use_wandb'] = True
config['wandb_project'] = 'llm-training'
```

## 🔧 Troubleshooting

### Out of Memory?
- Reduce batch size: `batch_size: 32 → 16`
- Use gradient accumulation: `gradient_accumulation_steps: 2`
- Try smaller model: `small → micro`

### Training Too Slow?
- Check GPU is enabled: `Runtime → Change runtime type`
- Use mixed precision: `mixed_precision: true`
- Increase batch size if memory allows

### Generation Quality Poor?
- Train for more steps: `max_steps: 3000 → 5000`
- Try larger model: `micro → small`
- Adjust learning rate: `learning_rate: 3e-4 → 1e-4`

## 📁 File Structure After Training

```
/content/Ai-practise/
├── checkpoints/
│   ├── best_model.pt          # Best performing model
│   ├── final_model.pt         # Final training state
│   └── checkpoint_step_*.pt   # Periodic saves
├── logs/
│   ├── training.log          # Detailed training logs
│   ├── config.json           # Training configuration
│   ├── training_summary.json # Final metrics
│   ├── training_summary.png  # Loss plots
│   └── generated_samples.txt # Text samples
└── colab_training.ipynb      # This notebook
```

## 🌟 Next Steps

After training in Colab:

1. **Deploy with GitHub Actions**
   - Models auto-deploy to Hugging Face
   - Create demo interfaces
   - Set up model APIs

2. **Experiment Further**
   - Try different architectures
   - Fine-tune on custom data
   - Implement new features

3. **Share Your Work**
   - Upload to Hugging Face Hub
   - Create blog posts
   - Share on social media

---

## 🎉 Happy Training!

Your LLM training journey starts here. With Colab's free GPUs, you can train state-of-the-art models in minutes!

Questions? Issues? Create an issue on [GitHub](https://github.com/prashant-2050/Ai-practise/issues).