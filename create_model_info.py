#!/usr/bin/env python3
"""Create model info JSON file."""

import json
import sys
import os
from datetime import datetime

def main():
    if len(sys.argv) != 5:
        print("Usage: python create_model_info.py <model_name> <model_size> <max_steps> <model_file>")
        exit(1)
    
    model_name = sys.argv[1]
    model_size = sys.argv[2]
    max_steps = sys.argv[3]
    model_file = sys.argv[4]
    
    model_info = {
        "name": model_name,
        "created_at": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "repository": "prashant-2050/Ai-practise",
        "training_method": "Google Colab",
        "model_size": model_size,
        "max_steps": max_steps,
        "files": [
            os.path.basename(model_file),
            "test_generation.txt",
            "model_info.json"
        ]
    }
    
    with open(f"packages/{model_name}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Created model_info.json for {model_name}")

if __name__ == "__main__":
    main()