#!/usr/bin/env python3
"""Simple model validation script for CI."""

import torch
import os
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_model.py <model_path>")
        exit(1)
    
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        exit(1)
    
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        print("✅ Model loaded successfully")
        print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"📈 Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        if "config" in checkpoint:
            print(f"⚙️ Model config: {checkpoint['config']}")
            
        if "loss" in checkpoint:
            print(f"📉 Final loss: {checkpoint['loss']:.4f}")
            
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()