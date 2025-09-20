# 🚀 GitHub Codespaces Development Guide

## Quick Start with GitHub Codespaces

GitHub Codespaces provides a cloud-based development environment with GPU access for training your LLM. Here's how to get started:

### 🎯 1. Launch Codespace

1. **Go to your repository**: `https://github.com/prashant-2050/Ai-practise`
2. **Click the green "Code" button**
3. **Select "Codespaces" tab**
4. **Click "Create codespace on main"**

Your codespace will launch with:
- ✅ Python 3.11 pre-installed
- ✅ All dependencies from `requirements.txt`
- ✅ GPU support (if available)
- ✅ VS Code with AI extensions
- ✅ Jupyter Lab ready

### 🎮 2. GPU-Enabled Training

**Check GPU availability:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Start training:**
```bash
# Quick training (1000 steps)
python train_cloud.py --model_size micro --max_steps 1000

# Extended training (5000 steps)
python train_cloud.py --model_size small --max_steps 5000 --learning_rate 2e-4
```

**Monitor training:**
```bash
# Watch logs in real-time
tail -f logs/training.log

# Check training progress
cat logs/training_summary.json
```

### 🎭 3. Interactive Development

**Start Jupyter Lab:**
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Launch Gradio demo:**
```bash
python gradio_app.py
```

**Generate text samples:**
```bash
python generate.py --prompt "To be or not to be" --max_tokens 100
```

### 📊 4. Training Workflows

#### A. Manual Training in Codespace
```bash
# 1. Open terminal in Codespace
# 2. Run training script
python train_cloud.py --model_size micro --max_steps 2000

# 3. Download checkpoints (Files tab → checkpoints/)
```

#### B. Automated GitHub Actions Training
```bash
# 1. Go to Actions tab in your repository
# 2. Select "Train LLM Model" workflow  
# 3. Click "Run workflow"
# 4. Choose parameters:
#    - Model size: micro/small/nano
#    - Max steps: 1000-5000
#    - Learning rate: 1e-4 to 5e-4
# 5. Click "Run workflow"
```

### 💾 5. Managing Checkpoints

**Local development (Codespace):**
```bash
# Checkpoints saved to: ./checkpoints/
ls -la checkpoints/

# Best model: checkpoints/best_model.pt
# Final model: checkpoints/final_model.pt
```

**GitHub Actions artifacts:**
- Go to Actions → Select completed run
- Download "model-checkpoints" artifact
- Extract and use locally

### 🔄 6. Development Workflow

**Recommended workflow:**
1. **Code in Codespace** → Develop and test interactively
2. **Quick training** → `python train_cloud.py --max_steps 500`
3. **Push changes** → Commit your improvements
4. **Production training** → Use GitHub Actions for longer runs
5. **Deploy** → Automatic deployment to Hugging Face Spaces

### 🎛️ 7. Environment Configuration

**Pre-configured features:**
- **GPU support**: CUDA/MPS auto-detection
- **Port forwarding**: Gradio (7860), Jupyter (8888)
- **Extensions**: Python, Jupyter, GitHub Copilot
- **Formatters**: Black, Pylint

**Custom configuration:**
```bash
# Edit devcontainer for your needs
.devcontainer/devcontainer.json

# Add custom packages
echo "your-package" >> requirements.txt
```

### 🐛 8. Troubleshooting

**Common issues:**

**GPU not available:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory:**
```bash
# Reduce batch size
python train_cloud.py --batch_size 2 --model_size nano

# Use gradient checkpointing (modify model)
```

**Slow training:**
```bash
# Use smaller model for testing
python train_cloud.py --model_size nano --max_steps 100

# Monitor system resources
htop
```

### 🎯 9. Best Practices

**For development:**
- Use **Codespace** for interactive work and quick experiments
- Keep training runs under 30 minutes to avoid timeouts
- Save work frequently (`git commit` often)

**For production training:**
- Use **GitHub Actions** for longer training runs
- Set reasonable step limits (1000-5000 steps)
- Monitor via Actions logs and artifacts

**Resource management:**
- **Stop Codespace** when not in use (saves compute hours)
- **Download important checkpoints** before stopping
- **Use version control** for all code changes

### 📈 10. Scaling Up

**For larger models:**
```bash
# Increase model size gradually
python train_cloud.py --model_size small --max_steps 3000

# Use learning rate scheduling
python train_cloud.py --learning_rate 1e-4 --max_steps 5000
```

**For production:**
- Consider **GitHub Actions** with larger runners
- Use **external GPU services** (RunPod, Lambda Labs)
- Deploy to **Hugging Face Spaces** with Pro GPUs

## 🔗 Quick Links

- **Your Repository**: https://github.com/prashant-2050/Ai-practise
- **Launch Codespace**: https://github.com/prashant-2050/Ai-practise/codespaces
- **GitHub Actions**: https://github.com/prashant-2050/Ai-practise/actions
- **Hugging Face Profile**: https://huggingface.co/YOUR_USERNAME

Happy coding in the cloud! 🚀✨