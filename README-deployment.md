# Free Deployment Options for LLM Training System

This guide covers multiple free platforms where you can deploy and test your lightweight LLM training system in real cloud environments.

---

## 🚀 **Option 1: Google Colab (Recommended for Training)**

### **Why Choose Colab:**
- **Free GPU access** (T4, sometimes A100)
- **Pre-installed ML libraries**
- **Easy sharing and collaboration**
- **No setup required**

### **Setup Instructions:**

1. **Upload to Google Drive:**
```bash
# Zip your project
zip -r llm-training.zip *.py *.md requirements.txt

# Upload to Google Drive
```

2. **Create Colab Notebook:**
```python
# Cell 1: Mount Drive and Setup
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your project
%cd /content/drive/MyDrive/llm-training

# Install dependencies
!pip install torch torchvision torchaudio transformers datasets tokenizers accelerate tqdm matplotlib seaborn

# Cell 2: Test Model Creation
from model_config import ModelConfig
from light_llm import LightLLM
import torch

config = ModelConfig.nano()  # Start small on free tier
model = LightLLM(config)
print(f"Model created: {model.count_parameters():,} parameters")
print(f"GPU available: {torch.cuda.is_available()}")

# Cell 3: Quick Training Demo
!python train.py

# Cell 4: Generate Text
!python generate.py demo
```

3. **Colab-Specific Optimizations:**
```python
# Add to train.py for Colab
import os
if 'COLAB_GPU' in os.environ:
    # Use smaller batch size for free tier
    batch_size = 2
    max_steps = 200  # Shorter training for demo
```

**Colab Deployment URL:** `https://colab.research.google.com/`

---

## 🔧 **Option 2: Kaggle Notebooks (30hrs/week GPU)**

### **Why Choose Kaggle:**
- **30 hours/week free GPU**
- **Persistent datasets**
- **Competition-ready environment**

### **Setup Instructions:**

1. **Create Kaggle Dataset:**
```bash
# Create kaggle-metadata.json
{
    "title": "Lightweight LLM Training",
    "id": "yourusername/lightweight-llm",
    "licenses": [{"name": "MIT"}]
}

# Upload via Kaggle API
kaggle datasets create -p ./
```

2. **Kaggle Notebook Template:**
```python
# Input: Add your dataset as input
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Copy files to working directory
!cp -r /kaggle/input/lightweight-llm/* /kaggle/working/
%cd /kaggle/working

# Install additional packages
!pip install transformers datasets tokenizers

# Run training
!python train.py
```

**Kaggle URL:** `https://www.kaggle.com/code`

---

## ☁️ **Option 3: Hugging Face Spaces (Web Apps)**

### **Why Choose HF Spaces:**
- **Free web app hosting**
- **Gradio/Streamlit integration**
- **Direct model sharing**
- **Community visibility**

### **Setup Instructions:**

1. **Create Space Configuration:**
```yaml
# Create: .huggingface/config.yaml
title: "Lightweight LLM Demo"
emoji: "🤖"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
pinned: false
license: "mit"
```

2. **Create Gradio App:**
```python
# app.py - Main Gradio interface
import gradio as gr
import torch
from generate import TextGenerator
import os

# Download pre-trained model or use demo model
model_path = "demo_model.pt"  # Upload your trained model

def generate_text(prompt, temperature, max_tokens, top_k, top_p):
    try:
        generator = TextGenerator(model_path)
        result = generator.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your text prompt..."),
        gr.Slider(0.1, 2.0, value=0.8, label="Temperature"),
        gr.Slider(10, 200, value=100, label="Max Tokens"),
        gr.Slider(1, 100, value=40, label="Top-K"),
        gr.Slider(0.1, 1.0, value=0.9, label="Top-P")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="🤖 Lightweight LLM Text Generator",
    description="A GPT-2 style transformer trained from scratch"
)

if __name__ == "__main__":
    demo.launch()
```

3. **Requirements for Spaces:**
```txt
# requirements.txt for HF Spaces
torch
transformers
gradio
numpy
```

4. **Deploy to Spaces:**
```bash
# Clone your space
git clone https://huggingface.co/spaces/yourusername/lightweight-llm
cd lightweight-llm

# Add files
cp app.py requirements.txt ./
cp checkpoints/best_model.pt demo_model.pt

# Push to deploy
git add .
git commit -m "Deploy LLM demo"
git push
```

**Hugging Face Spaces URL:** `https://huggingface.co/spaces`

---

## 🐙 **Option 4: GitHub Codespaces (60hrs/month)**

### **Why Choose Codespaces:**
- **Full development environment**
- **60 hours/month free**
- **Direct GitHub integration**
- **VS Code in browser**

### **Setup Instructions:**

1. **Add Codespace Configuration:**
```json
// .devcontainer/devcontainer.json
{
    "name": "LLM Training Environment",
    "image": "mcr.microsoft.com/devcontainers/python:3.11",
    "features": {
        "ghcr.io/devcontainers/features/nvidia-cuda:1": {
            "installToolkit": true
        }
    },
    "postCreateCommand": "pip install -r requirements.txt",
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-toolsai.jupyter"
            ]
        }
    }
}
```

2. **Codespace Workflow:**
```bash
# In Codespace terminal
python train.py        # Train model
python generate.py demo # Test generation

# Share via web preview
python -m http.server 8000
# Use port forwarding to share demo
```

**GitHub Codespaces URL:** Access via your GitHub repository → Code → Codespaces

---

## 🐍 **Option 5: Railway (Free Tier)**

### **Why Choose Railway:**
- **Easy deployment**
- **Automatic HTTPS**
- **Environment variables**
- **Database options**

### **Setup Instructions:**

1. **Railway Configuration:**
```toml
# railway.toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "python app.py"
healthcheckPath = "/health"
```

2. **Web App for Railway:**
```python
# app.py - Flask web interface
from flask import Flask, request, jsonify, render_template_string
from generate import TextGenerator
import os

app = Flask(__name__)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Text Generator</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; }
        .result { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🤖 Lightweight LLM Generator</h1>
    <form id="generateForm">
        <textarea id="prompt" placeholder="Enter your prompt..."></textarea><br>
        <label>Temperature: <input type="range" id="temperature" min="0.1" max="2" step="0.1" value="0.8"></label>
        <span id="tempValue">0.8</span><br>
        <button type="submit">Generate Text</button>
    </form>
    <div id="result" class="result" style="display:none;"></div>

    <script>
        document.getElementById('temperature').oninput = function() {
            document.getElementById('tempValue').innerHTML = this.value;
        }
        
        document.getElementById('generateForm').onsubmit = function(e) {
            e.preventDefault();
            const prompt = document.getElementById('prompt').value;
            const temperature = document.getElementById('temperature').value;
            
            fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt, temperature: parseFloat(temperature)})
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').innerHTML = '<h3>Generated:</h3><p>' + data.text + '</p>';
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    temperature = data.get('temperature', 0.8)
    
    try:
        # Use a smaller model for web deployment
        generator = TextGenerator("demo_model.pt")
        result = generator.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=temperature
        )
        return jsonify({"text": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

**Railway URL:** `https://railway.app/`

---

## 🆓 **Option 6: Render (Free Tier)**

### **Why Choose Render:**
- **Free web services**
- **Automatic deployments**
- **Custom domains**

### **Setup Instructions:**

```yaml
# render.yaml
services:
  - type: web
    name: llm-generator
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

**Render URL:** `https://render.com/`

---

## 📊 **Comparison Table**

| Platform | GPU | CPU | Storage | Time Limit | Best For |
|----------|-----|-----|---------|------------|----------|
| **Colab** | ✅ T4/A100 | Good | Temporary | Session-based | Training |
| **Kaggle** | ✅ P100/T4 | Good | 20GB | 30hrs/week | Experiments |
| **HF Spaces** | ❌ CPU only | Limited | 3GB | Always-on | Demos |
| **Codespaces** | ❌ CPU only | Good | 32GB | 60hrs/month | Development |
| **Railway** | ❌ CPU only | Good | 1GB | Always-on | Web Apps |
| **Render** | ❌ CPU only | Limited | 1GB | Always-on | Simple Demos |

---

## 🎯 **Recommended Deployment Strategy**

### **Phase 1: Training & Development**
1. **Use Colab** for initial training and experimentation
2. **Use Kaggle** for longer training runs and dataset experiments

### **Phase 2: Demo & Sharing**
1. **Deploy to HF Spaces** for interactive text generation demos
2. **Use Railway/Render** for custom web applications

### **Phase 3: Collaboration**
1. **Share via GitHub** with Codespaces for collaborative development
2. **Publish models** on Hugging Face Hub

---

## 🚀 **Quick Start Deployment**

### **1-Click Colab Deployment:**
```python
# Save this as colab_deploy.py
!git clone https://github.com/yourusername/Ai-practise.git
%cd Ai-practise
!pip install -r requirements.txt
!python train.py  # Quick training
!python generate.py demo  # Test generation
```

### **1-Click HF Spaces Deployment:**
1. Fork the repository
2. Create new Space on Hugging Face
3. Upload `app.py` and trained model
4. Deploy instantly!

Each platform offers unique advantages - choose based on your specific needs for training, development, or demonstration! 🎉