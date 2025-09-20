#!/bin/bash

# DevContainer setup script for LLM training environment
echo "🚀 Setting up LLM Training Environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    htop \
    nano \
    vim \
    tree \
    zip \
    unzip

# Install Python packages
echo "🐍 Installing Python packages..."
pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📋 Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, installing basic packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers gradio huggingface_hub datasets accelerate
fi

# Install Jupyter
echo "📓 Installing Jupyter..."
pip install jupyter jupyterlab notebook

# Install development tools
echo "🛠️  Installing development tools..."
pip install black pylint pytest mypy

# Set up workspace
echo "📁 Setting up workspace..."
mkdir -p /workspace/.cache
mkdir -p checkpoints
mkdir -p demo_checkpoints

# Check GPU availability
echo "🎮 Checking GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Running on CPU')
"

# Download sample data if needed
echo "📥 Preparing sample data..."
if [ ! -f "shakespeare.txt" ]; then
    echo "Downloading Shakespeare dataset..."
    wget -O shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi

echo "✅ Setup complete! Ready for LLM training and development."
echo ""
echo "🎯 Available commands:"
echo "  python train.py          - Train the model"
echo "  python generate.py       - Generate text"
echo "  python gradio_app.py     - Start Gradio interface"
echo "  jupyter lab              - Start Jupyter Lab"
echo ""
echo "🚀 Happy coding!"