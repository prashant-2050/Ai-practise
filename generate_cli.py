#!/usr/bin/env python3
"""
Command-line interface for text generation
"""

import argparse
import sys
import os
from generate import TextGenerator

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Generate text from trained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='To be or not to be',
                       help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found!")
        print("Available checkpoints:")
        if os.path.exists('checkpoints'):
            for f in os.listdir('checkpoints'):
                if f.endswith('.pt'):
                    print(f"  - checkpoints/{f}")
        sys.exit(1)
    
    try:
        # Initialize generator
        print(f"Loading model from: {args.checkpoint}")
        generator = TextGenerator(args.checkpoint, device=args.device)
        
        # Generate text
        print(f"\nPrompt: '{args.prompt}'")
        print("Generated text:")
        print("-" * 50)
        
        generated_text = generator.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print(generated_text)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Falling back to demo mode...")
        
        # Fallback to simple demo
        try:
            from generate import demo_generation
            demo_generation()
        except Exception as e2:
            print(f"Demo mode also failed: {e2}")
            print("Please check your model checkpoints and try again.")
            sys.exit(1)

if __name__ == "__main__":
    main()