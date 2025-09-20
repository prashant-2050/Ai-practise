# Colab Training Notebook - Upload this to Google Colab

# Cell 1: Setup and Installation
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Install required packages
!pip install torch torchvision torchaudio transformers datasets tokenizers accelerate tqdm matplotlib seaborn requests

# Clone or upload your project
# Option 1: Clone from GitHub (if you've pushed to GitHub)
# !git clone https://github.com/yourusername/Ai-practise.git
# %cd Ai-practise

# Option 2: Use uploaded files from Drive
# Create a folder in your Drive called "llm-training" and upload all .py files there
%cd /content/drive/MyDrive/llm-training

# List files to verify
!ls -la

# Cell 2: Check GPU and Environment
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test model creation
from model_config import ModelConfig
from light_llm import LightLLM

# Use nano config for faster training on free tier
config = ModelConfig.nano()
model = LightLLM(config)
print(f"Model created: {model.count_parameters():,} parameters")

# Cell 3: Data Preparation
from dataset import prepare_dataset, create_dataloader

# Prepare Shakespeare dataset
print("Preparing dataset...")
dataset, tokenizer = prepare_dataset("shakespeare", max_length=256)  # Shorter for faster training
train_loader = create_dataloader(dataset, batch_size=4, shuffle=True)

print(f"Dataset size: {len(dataset)} sequences")
print("Data preparation complete!")

# Cell 4: Training Setup
from train import Trainer

# Create trainer with Colab-optimized settings
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    learning_rate=5e-4,        # Slightly higher LR for shorter training
    max_steps=300,             # Reduced for demo (usually 500-1000)
    eval_interval=50,          # More frequent evaluation
    save_dir="/content/drive/MyDrive/llm-training/checkpoints"  # Save to Drive
)

print("Trainer initialized!")
print(f"Device: {trainer.device}")
print(f"Training steps: {trainer.max_steps}")

# Cell 5: Start Training
print("🚀 Starting training...")
print("=" * 50)

# Train the model
trainer.train()

print("=" * 50)
print("✅ Training completed!")

# Cell 6: Test Generation
from generate import TextGenerator

# Load the best model
model_path = "/content/drive/MyDrive/llm-training/checkpoints/best_model.pt"

if os.path.exists(model_path):
    print("Loading trained model...")
    generator = TextGenerator(model_path)
    
    # Test prompts
    prompts = [
        "To be or not to be",
        "Once upon a time",
        "The king said",
        "In fair Verona"
    ]
    
    print("🎭 Generating text samples...")
    print("=" * 60)
    
    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        print("-" * 40)
        generated = generator.generate(
            prompt=prompt,
            max_new_tokens=80,
            temperature=0.8,
            top_k=40,
            top_p=0.9
        )
        # Show only the generated part
        new_text = generated[len(prompt):] if generated.startswith(prompt) else generated
        print(f"Generated: {new_text}")
        print("=" * 60)
else:
    print("❌ Model not found. Training may have failed.")

# Cell 7: Interactive Generation
def interactive_generation():
    """Interactive text generation in Colab"""
    print("🤖 Interactive LLM Generator")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    while True:
        prompt = input("\n📝 Enter prompt: ")
        if prompt.lower() == 'quit':
            break
            
        if prompt.strip():
            try:
                result = generator.generate(
                    prompt=prompt,
                    max_new_tokens=100,
                    temperature=0.8
                )
                print(f"\n🎭 Generated: {result}")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("Please enter a valid prompt!")

# Uncomment to run interactive mode
# interactive_generation()

# Cell 8: Save Model to Drive (Backup)
import shutil

# Copy final model to Drive for download
source_dir = "/content/drive/MyDrive/llm-training/checkpoints"
backup_dir = "/content/drive/MyDrive/llm-training/model_backup"

if os.path.exists(source_dir):
    os.makedirs(backup_dir, exist_ok=True)
    
    # Copy all checkpoints
    for file in os.listdir(source_dir):
        if file.endswith('.pt'):
            shutil.copy2(os.path.join(source_dir, file), backup_dir)
            print(f"✅ Backed up: {file}")
    
    print(f"📁 Models saved to: {backup_dir}")
    print("You can download these files from Google Drive!")

# Cell 9: Performance Analysis
def analyze_training():
    """Analyze training performance"""
    import matplotlib.pyplot as plt
    
    # If you logged training metrics, plot them here
    print("📊 Training Analysis")
    print(f"Final model parameters: {model.count_parameters():,}")
    
    # GPU utilization check
    if torch.cuda.is_available():
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Model size analysis
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"Model size: {model_size_mb:.1f} MB")

analyze_training()

print("""
🎉 Colab Training Complete!

Next Steps:
1. Download models from Google Drive
2. Deploy to Hugging Face Spaces for sharing
3. Experiment with different hyperparameters
4. Try larger models if you have Colab Pro

📚 Resources:
- README-deployment.md for deployment options
- README-transformer-math.md for technical details
""")