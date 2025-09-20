#!/usr/bin/env python3
"""Deploy model to Hugging Face Hub."""

import os
from huggingface_hub import HfApi, login, create_repo
import json
import sys

def main():
    # Get parameters
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'unknown-model'
    
    # Login
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("❌ HF_TOKEN not found")
        exit(1)
    
    login(token=token)
    api = HfApi()
    
    # Repository info
    username = os.environ.get('HF_USERNAME', 'prashant-2050')
    repo_id = f'{username}/{model_name}'
    
    print(f"📤 Uploading to: {repo_id}")
    
    try:
        # Create repository
        create_repo(repo_id, repo_type='model', exist_ok=True)
        print("✅ Repository created/exists")
        
        # Upload model file
        model_files = os.listdir('model_package/')
        for file in model_files:
            if file.endswith('.pt'):
                api.upload_file(
                    path_or_fileobj=f'model_package/{file}',
                    path_in_repo='pytorch_model.pt',
                    repo_id=repo_id
                )
                print(f"✅ Uploaded: {file}")
                break
        
        # Upload model info
        if 'model_info.json' in model_files:
            api.upload_file(
                path_or_fileobj='model_package/model_info.json',
                path_in_repo='model_info.json',
                repo_id=repo_id
            )
            print("✅ Uploaded: model_info.json")
        
        # Upload test generation
        if 'test_generation.txt' in model_files:
            api.upload_file(
                path_or_fileobj='model_package/test_generation.txt',
                path_in_repo='sample_generation.txt',
                repo_id=repo_id
            )
            print("✅ Uploaded: sample_generation.txt")
        
        # Create README
        readme_content = f"""# {model_name}

A custom transformer model trained using Google Colab.

## Model Details
- **Architecture**: Custom Light LLM
- **Training**: Google Colab (GPU)
- **Repository**: [prashant-2050/Ai-practise](https://github.com/prashant-2050/Ai-practise)

## Usage

```python
import torch

# Load model
checkpoint = torch.load('pytorch_model.pt', map_location='cpu')
# ... model loading code ...
```

## Training

This model was trained using our automated pipeline:
1. Google Colab for GPU training
2. GitHub Actions for deployment
3. Automatic upload to Hugging Face Hub

See the [repository](https://github.com/prashant-2050/Ai-practise) for full training code and pipeline setup.
"""
        
        # Upload README
        with open('README.md', 'w') as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj='README.md',
            path_in_repo='README.md',
            repo_id=repo_id
        )
        print("✅ Uploaded: README.md")
        
        print(f"🎉 Deployment complete: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()