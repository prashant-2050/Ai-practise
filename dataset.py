"""
Dataset preparation for lightweight LLM training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import requests
import os
from typing import List, Optional

class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize the entire text
        tokens = tokenizer.encode(text)
        
        # Create sequences of max_length
        self.sequences = []
        for i in range(0, len(tokens) - max_length, max_length):
            sequence = tokens[i:i + max_length]
            if len(sequence) == max_length:
                self.sequences.append(torch.tensor(sequence, dtype=torch.long))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def download_shakespeare():
    """Download a small text corpus for training"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    if not os.path.exists("shakespeare.txt"):
        print("Downloading Shakespeare dataset...")
        response = requests.get(url)
        with open("shakespeare.txt", "w") as f:
            f.write(response.text)
        print("Download complete!")
    else:
        print("Shakespeare dataset already exists")
    
    return "shakespeare.txt"

def download_tinystories():
    """Download TinyStories dataset (smaller version)"""
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    
    # For simplicity, we'll create a small sample text
    sample_stories = """
Once upon a time, there was a little girl named Lucy. She had a red ball that she loved to play with.
One day, Lucy went to the park with her mom. She saw other children playing and wanted to join them.
Lucy threw her red ball high in the sky. It came down and bounced on the ground.
A little boy named Tom saw Lucy's ball. He picked it up and gave it back to her.
"Thank you!" said Lucy with a big smile. They became friends and played together all day.
The end.

Once there was a small cat named Whiskers. Whiskers lived in a cozy house with a kind family.
Every morning, Whiskers would wake up and stretch his paws. He loved to sit by the window and watch birds.
One sunny day, Whiskers saw a butterfly in the garden. He wanted to catch it and play.
Whiskers ran outside and chased the butterfly around the flowers. But the butterfly was too fast.
Whiskers sat down and watched the butterfly dance in the air. It was beautiful.
The family called Whiskers for dinner. He ran back inside, happy and tired from his adventure.
The end.

There was once a magic tree in a forest. The tree had golden leaves that sparkled in the sunlight.
Animals from all over the forest came to see the magic tree. They made wishes under its branches.
A little rabbit wished for more carrots. A squirrel wished for more nuts. A bird wished for a prettier song.
The magic tree heard all their wishes. Its golden leaves glowed brighter and granted each wish.
All the animals were happy and thankful. They took care of the magic tree and kept the forest clean.
The magic tree continued to help animals for many years, spreading joy throughout the forest.
The end.
"""
    
    if not os.path.exists("tinystories.txt"):
        with open("tinystories.txt", "w") as f:
            f.write(sample_stories)
        print("Created sample TinyStories dataset")
    
    return "tinystories.txt"

def prepare_dataset(dataset_name: str = "shakespeare", max_length: int = 512):
    """Prepare dataset for training"""
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Download and load dataset
    if dataset_name == "shakespeare":
        file_path = download_shakespeare()
    elif dataset_name == "tinystories":
        file_path = download_tinystories()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Read text
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset size: {len(text)} characters")
    
    # Create dataset
    dataset = TextDataset(text, tokenizer, max_length)
    print(f"Created {len(dataset)} training sequences")
    
    return dataset, tokenizer

def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = True):
    """Create DataLoader for training"""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0  # Set to 0 for macOS compatibility
    )

if __name__ == "__main__":
    # Test dataset preparation
    print("Testing dataset preparation...")
    
    dataset, tokenizer = prepare_dataset("shakespeare", max_length=128)
    dataloader = create_dataloader(dataset, batch_size=2)
    
    # Test a batch
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")
    print(f"Sample tokens: {batch[0][:10]}")
    print(f"Sample text: {tokenizer.decode(batch[0][:50])}")