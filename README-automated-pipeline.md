# 🚀 Automated ML Pipeline Guide

This guide covers the complete automated machine learning pipeline for training and deploying your custom LLM using Google Colab and GitHub Actions.

## 🎯 Pipeline Overview

Our automated pipeline combines the best of both worlds:
- **Google Colab**: Free GPU training with Tesla T4 accelerators
- **GitHub Actions**: Automated packaging and deployment to Hugging Face Hub
- **Full Automation**: Single-command training and deployment

## 📋 Quick Start

### Option 1: Manual Colab Training (Recommended for beginners)

1. **Open the training notebook**: 
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `colab_training.ipynb` from this repository

2. **Run one-command setup**:
   ```python
   !curl -sSL https://raw.githubusercontent.com/prashant-2050/Ai-practise/main/colab_setup.py | python
   ```

3. **Monitor training**: The notebook includes live progress tracking and automatic saving

### Option 2: Automated Pipeline (Advanced)

1. **Setup webhook server** (in Colab):
   ```python
   from colab_automated import ColabAutomatedTraining
   trainer = ColabAutomatedTraining()
   trainer.start_training()
   ```

2. **Trigger from GitHub** (locally):
   ```bash
   python pipeline.py --model_size micro --max_steps 3000
   ```

### Option 3: GitHub Actions Only (For existing models)

1. **Trigger workflow**: Go to Actions → "Automated ML Pipeline" → Run workflow
2. **Select parameters**:
   - Model size: nano/micro/small
   - Max steps: 1000-5000
   - Auto-deploy: true/false

## 🛠️ Setup Requirements

### For Google Colab Training

**No setup required!** Everything runs in the cloud with free GPU access.

**Optional**: For automated integration, you'll need:
- GitHub Personal Access Token (for automated commits)
- ngrok account (for webhook connectivity)

### For GitHub Actions Deployment

**Required secrets in your GitHub repository**:
- `HF_TOKEN`: Your Hugging Face access token
- `HF_USERNAME`: Your Hugging Face username (optional, defaults to 'prashant-2050')

**Setup steps**:
1. Go to your repository → Settings → Secrets and variables → Actions
2. Add `HF_TOKEN`: Get from [Hugging Face Settings](https://huggingface.co/settings/tokens)
3. Add `HF_USERNAME`: Your HF username (optional)

## 🔧 Pipeline Components

### 1. Training Scripts

- **`colab_training.ipynb`**: Interactive Jupyter notebook with GPU training
- **`colab_setup.py`**: One-command setup for any Colab environment
- **`colab_automated.py`**: Webhook-enabled automated training

### 2. Orchestration

- **`pipeline.py`**: Main orchestration script for end-to-end automation
- **`create_model_info.py`**: Model metadata generation
- **`validate_model.py`**: Model file validation
- **`deploy_hf.py`**: Hugging Face Hub deployment

### 3. GitHub Actions

- **`simple-pipeline.yml`**: Streamlined workflow for packaging and deployment
- **`automated-pipeline.yml`**: Full automation with training triggers (advanced)

## 📊 Model Configurations

Choose your model size based on your needs:

| Size | Parameters | Training Time | Memory | Use Case |
|------|------------|---------------|---------|----------|
| **nano** | ~0.1M | 2-5 minutes | <1GB | Quick testing |
| **micro** | ~1M | 10-15 minutes | ~2GB | Development |
| **small** | ~10M | 30-60 minutes | ~4GB | Production |

## 🎮 Usage Examples

### Training with Different Configurations

```bash
# Quick nano model for testing
python pipeline.py --model_size nano --max_steps 500

# Standard micro model
python pipeline.py --model_size micro --max_steps 3000

# Production small model
python pipeline.py --model_size small --max_steps 10000 --auto_deploy true
```

### Manual Training in Colab

```python
# In Google Colab
!git clone https://github.com/prashant-2050/Ai-practise.git
%cd Ai-practise

# Install dependencies
!pip install -r requirements.txt

# Train model
!python train.py --model_config micro --max_steps 3000

# Generate sample
!python generate_cli.py --checkpoint checkpoints/best_model.pt --prompt "To be or not to be"
```

### Deployment Only

If you already have trained models, deploy them directly:

```bash
# Trigger GitHub Actions deployment workflow
gh workflow run simple-pipeline.yml -f trigger_training=false -f auto_deploy=true
```

## 📈 Monitoring and Results

### Training Progress

- **Colab Notebook**: Real-time loss plots and sample generations
- **GitHub Actions**: Step-by-step progress in workflow logs
- **Artifacts**: Model files and metadata automatically saved

### Expected Results

**Training Loss**: Should decrease from ~10 to ~2-4 over training
**Sample Quality**: Improves from gibberish to coherent text
**Training Time**: 
- Nano: 2-5 minutes
- Micro: 10-15 minutes  
- Small: 30-60 minutes

### Output Artifacts

1. **Model Files**: `checkpoints/best_model.pt`, `final_model.pt`
2. **Metadata**: `model_info.json` with training details
3. **Samples**: `test_generation.txt` with example outputs
4. **Logs**: Training progress and metrics

## 🚀 Deployment Options

### Hugging Face Hub

Automatic deployment creates:
- Model repository with PyTorch weights
- README with usage instructions
- Sample generations
- Training metadata

**Access your model**: `https://huggingface.co/your-username/model-name`

### Gradio Demo (Optional)

```python
# Create interactive demo
python gradio_app.py --checkpoint checkpoints/best_model.pt
```

## 🔧 Advanced Configuration

### Webhook Integration

For fully automated pipeline with webhook triggers:

1. **Setup ngrok** (in Colab):
   ```python
   !pip install pyngrok
   from pyngrok import ngrok
   public_url = ngrok.connect(5000)
   print(f"Webhook URL: {public_url}")
   ```

2. **Configure webhook** (in your automation script):
   ```python
   import requests
   webhook_url = "your-ngrok-url/train"
   payload = {"model_size": "micro", "max_steps": 3000}
   requests.post(webhook_url, json=payload)
   ```

### Custom Model Configurations

Edit `model_config.py` to create custom configurations:

```python
@staticmethod
def custom():
    return ModelConfig(
        vocab_size=1000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
        dropout=0.1
    )
```

## 🛡️ Troubleshooting

### Common Issues

**1. Colab session timeout**
- Solution: Enable automatic saving to Google Drive
- Notebook includes auto-save functionality

**2. GitHub Actions deployment fails**
- Check HF_TOKEN is set correctly
- Verify Hugging Face token permissions

**3. Model file too large**
- Use smaller model configuration
- Enable gradient checkpointing for memory efficiency

**4. Training stuck or slow**
- Ensure GPU is enabled in Colab
- Try smaller batch size or model

### Debug Commands

```bash
# Check model file
python validate_model.py checkpoints/best_model.pt

# Test generation
python generate_cli.py --checkpoint checkpoints/best_model.pt --prompt "test"

# Check environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

## 🎯 Best Practices

1. **Start Small**: Begin with nano model for testing
2. **Monitor Progress**: Use Colab notebook for interactive development
3. **Save Frequently**: Enable auto-save to prevent data loss
4. **Version Control**: Commit model checkpoints and metadata
5. **Test Generation**: Always validate model output quality

## 🔄 Complete Workflow

Here's the recommended end-to-end workflow:

1. **Development**: Use Colab notebook for experimentation
2. **Training**: Run automated training with desired configuration
3. **Validation**: Check model quality with test generations
4. **Deployment**: Automatic packaging and upload to Hugging Face
5. **Sharing**: Create interactive demos and share with community

## 📚 Next Steps

- **Scale Up**: Try larger models with more training steps
- **Custom Data**: Replace Shakespeare with your own dataset
- **Fine-tuning**: Start from a pretrained checkpoint
- **Production**: Deploy as API endpoint or web application
- **Community**: Share your models on Hugging Face Hub

## 🆘 Support

- **Documentation**: Check individual README files for components
- **Issues**: Open GitHub issues for bugs or feature requests  
- **Community**: Share your results and get help in discussions

---

🎉 **Happy Training!** You now have a complete automated ML pipeline for training and deploying custom language models!