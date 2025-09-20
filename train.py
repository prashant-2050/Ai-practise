"""
Training script for lightweight LLM
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import time
import os
from typing import Optional

from model_config import ModelConfig
from light_llm import LightLLM
from dataset import prepare_dataset, create_dataloader

class Trainer:
    def __init__(
        self,
        model: LightLLM,
        train_loader: DataLoader,
        device: str = "auto",
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        eval_interval: int = 100,
        save_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        
        # Device setup
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using MPS (Apple Silicon) acceleration")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA acceleration")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer setup
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'LayerNorm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
        
        # Learning rate scheduler
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                return coeff
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Training state
        self.step = 0
        self.best_loss = float('inf')
        
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Forward pass
        # For language modeling, targets are the input shifted by one
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        logits, loss = self.model(inputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self, eval_steps: int = 10):
        """Evaluate model on a few batches"""
        self.model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= eval_steps:
                    break
                
                batch = batch.to(self.device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                logits, loss = self.model(inputs, targets)
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else 0
    
    def save_checkpoint(self, filename: Optional[str] = None):
        """Save model checkpoint"""
        if filename is None:
            filename = f"checkpoint_step_{self.step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'config': self.model.config
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.max_steps} steps...")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        # Create infinite data iterator
        data_iter = iter(self.train_loader)
        
        for step in range(self.max_steps):
            self.step = step
            
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart iterator
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            
            # Training step
            loss = self.train_step(batch)
            
            # Logging
            if step % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"Step {step:4d} | Loss: {loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")
            
            # Evaluation and checkpointing
            if step % self.eval_interval == 0 and step > 0:
                eval_loss = self.evaluate()
                print(f"Eval loss: {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.save_checkpoint("best_model.pt")
                
                # Save regular checkpoint
                self.save_checkpoint()
        
        print("Training completed!")
        self.save_checkpoint("final_model.pt")

def main():
    """Main training function"""
    
    # Configuration
    config = ModelConfig.nano()  # ~22M parameters (smaller, faster)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset, tokenizer = prepare_dataset("shakespeare", max_length=config.n_positions)
    train_loader = create_dataloader(dataset, batch_size=4, shuffle=True)
    
    # Create model
    print("Creating model...")
    model = LightLLM(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        learning_rate=3e-4,
        max_steps=200,  # Quick demo training
        eval_interval=50,
        save_dir="checkpoints"
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()