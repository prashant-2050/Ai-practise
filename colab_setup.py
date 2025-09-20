"""
🚀 Quick Colab Setup Script
Run this first in any Colab notebook to set up your training environment

Usage in Colab:
!wget -q https://raw.githubusercontent.com/prashant-2050/Ai-practise/main/colab_setup.py
exec(open('colab_setup.py').read())
"""

import os
import subprocess
import sys
import torch

def setup_colab_environment():
    """Set up Colab environment for LLM training"""
    print("🚀 Setting up Colab environment for LLM training...")
    
    # 1. Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = 'cuda'
    else:
        print("⚠️ No GPU detected - will use CPU (slower)")
        device = 'cpu'
    
    # 2. Clone repository
    repo_url = "https://github.com/prashant-2050/Ai-practise.git"
    if not os.path.exists('/content/Ai-practise'):
        print("📥 Cloning repository...")
        os.system(f"git clone {repo_url}")
        print("✅ Repository cloned!")
    else:
        print("📁 Repository exists, pulling latest changes...")
        os.chdir('/content/Ai-practise')
        os.system("git pull origin main")
    
    os.chdir('/content/Ai-practise')
    
    # 3. Install dependencies
    print("📦 Installing dependencies...")
    os.system("pip install -q torch torchvision torchaudio")
    os.system("pip install -q transformers datasets tokenizers")
    os.system("pip install -q matplotlib seaborn plotly tqdm")
    os.system("pip install -q wandb tensorboard huggingface_hub")
    
    # 4. Download data
    if not os.path.exists('shakespeare.txt'):
        print("📚 Downloading training data...")
        os.system("wget -q -O shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    
    # 5. Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("✅ Colab environment ready!")
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"🔥 Device: {device}")
    
    return device

def quick_train(model_size='micro', max_steps=2000, device='auto'):
    """Quick training function for Colab"""
    print(f"🚀 Starting quick training: {model_size} model, {max_steps} steps")
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run training with optimal Colab settings
    cmd = f"""python train_cloud.py \\
        --model_size {model_size} \\
        --max_steps {max_steps} \\
        --batch_size {64 if device == 'cuda' else 16} \\
        --device {device} \\
        --mixed_precision true \\
        --eval_interval 200 \\
        --save_interval 500"""
    
    os.system(cmd)
    
    print("✅ Training completed!")
    print("📁 Check checkpoints/ for saved models")

if __name__ == "__main__":
    device = setup_colab_environment()
    
    # Optional: start training immediately
    start_training = input("Start training now? (y/N): ").lower() == 'y'
    if start_training:
        model_size = input("Model size (nano/micro/small) [micro]: ") or 'micro'
        max_steps = int(input("Max steps [2000]: ") or 2000)
        quick_train(model_size, max_steps, device)