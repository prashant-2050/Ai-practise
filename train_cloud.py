"""
Cloud-optimized training script for lightweight LLM
Enhanced for GitHub Actions and cloud environments
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import time
import os
import argparse
import json
import logging
from typing import Optional
from datetime import datetime

from model_config import ModelConfig
from light_llm import LightLLM
from dataset import prepare_dataset, create_dataloader

# Set up logging
def setup_logging(log_dir: str):
    """Set up structured logging for cloud training"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Set up logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class CloudTrainer:
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
        save_interval: int = 200,
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.logger = logger or logging.getLogger('training')
        
        # Device setup with better cloud detection
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.logger.info("Using MPS (Apple Silicon) acceleration")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.logger.info(f"Using CUDA acceleration - {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                self.logger.info("Using CPU (consider GPU for faster training)")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.logger.info(f"Model moved to {self.device}")
        
        # Optimizer setup (same as original)
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
        
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Training state
        self.step = 0
        self.best_loss = float('inf')
        self.start_time = None
        
        # Metrics tracking
        self.training_metrics = {
            'steps': [],
            'losses': [],
            'learning_rates': [],
            'eval_losses': []
        }
    
    def train_step(self, batch):
        """Single training step with enhanced logging"""
        self.model.train()
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Forward pass
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        logits, loss = self.model(inputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping with monitoring
        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if self.step % 100 == 0:
                self.logger.info(f"Grad norm: {grad_norm:.4f}")
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self, eval_steps: int = 10):
        """Evaluate model with enhanced metrics"""
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
        
        avg_loss = total_loss / count if count > 0 else 0
        self.training_metrics['eval_losses'].append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, filename: Optional[str] = None, is_best: bool = False):
        """Enhanced checkpoint saving with metadata"""
        if filename is None:
            filename = f"checkpoint_step_{self.step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'config': self.model.config,
            'best_loss': self.best_loss,
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint: {filepath}")
        
        # Also save training metrics as JSON
        metrics_file = os.path.join(self.log_dir, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved: {best_path}")
    
    def train(self):
        """Enhanced training loop with cloud optimizations"""
        self.start_time = time.time()
        
        self.logger.info("=" * 50)
        self.logger.info("STARTING LLM TRAINING")
        self.logger.info("=" * 50)
        self.logger.info(f"Model: {self.model.config}")
        self.logger.info(f"Parameters: {self.model.count_parameters():,}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Max steps: {self.max_steps}")
        self.logger.info(f"Batch size: {self.train_loader.batch_size}")
        self.logger.info("=" * 50)
        
        # Create infinite data iterator
        data_iter = iter(self.train_loader)
        
        try:
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
                
                # Track metrics
                self.training_metrics['steps'].append(step)
                self.training_metrics['losses'].append(loss)
                self.training_metrics['learning_rates'].append(self.scheduler.get_last_lr()[0])
                
                # Enhanced logging
                if step % 10 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - self.start_time
                    steps_per_sec = (step + 1) / elapsed
                    eta = (self.max_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0
                    
                    log_msg = (f"Step {step:4d}/{self.max_steps} | "
                              f"Loss: {loss:.4f} | "
                              f"LR: {lr:.2e} | "
                              f"Speed: {steps_per_sec:.2f} steps/s | "
                              f"ETA: {eta/60:.1f}m")
                    
                    self.logger.info(log_msg)
                    print(log_msg)  # Also print for GitHub Actions logs
                
                # Evaluation and checkpointing
                if step % self.eval_interval == 0 and step > 0:
                    eval_loss = self.evaluate()
                    eval_msg = f"Eval loss: {eval_loss:.4f}"
                    self.logger.info(eval_msg)
                    print(eval_msg)
                    
                    # Save best model
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint("best_model.pt", is_best=True)
                
                # Regular checkpointing
                if step % self.save_interval == 0 and step > 0:
                    self.save_checkpoint()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Final save and summary
            total_time = time.time() - self.start_time
            self.logger.info("=" * 50)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info(f"Total time: {total_time/60:.1f} minutes")
            self.logger.info(f"Final loss: {self.training_metrics['losses'][-1]:.4f}")
            self.logger.info(f"Best eval loss: {self.best_loss:.4f}")
            self.logger.info("=" * 50)
            
            self.save_checkpoint("final_model.pt")
            self.create_training_summary()
    
    def create_training_summary(self):
        """Create a comprehensive training summary"""
        summary = {
            'model_config': str(self.model.config),
            'total_parameters': self.model.count_parameters(),
            'device': str(self.device),
            'training_steps': self.max_steps,
            'final_loss': self.training_metrics['losses'][-1] if self.training_metrics['losses'] else None,
            'best_eval_loss': self.best_loss,
            'training_time_minutes': (time.time() - self.start_time) / 60,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = os.path.join(self.log_dir, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training summary saved: {summary_file}")

def main():
    """Main training function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train LLM in cloud environment')
    parser.add_argument('--model_size', type=str, default='micro', 
                        choices=['nano', 'micro', 'small'],
                        help='Model size configuration')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=200,
                        help='Checkpoint save interval')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--mixed_precision', type=str, default='false',
                        help='Use mixed precision training (true/false)')
    
    args = parser.parse_args()
    
    # Convert string boolean to actual boolean
    args.mixed_precision = args.mixed_precision.lower() == 'true'
    
    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting training with args: {args}")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    # Configuration
    if args.model_size == 'nano':
        config = ModelConfig.nano()
    elif args.model_size == 'micro':
        config = ModelConfig.micro()
    elif args.model_size == 'small':
        config = ModelConfig.small()
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    logger.info(f"Using {args.model_size} model configuration")
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    try:
        dataset, tokenizer = prepare_dataset("shakespeare", max_length=config.n_positions)
        train_loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
        logger.info(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Create model
    logger.info("Creating model...")
    model = LightLLM(config)
    logger.info(f"Model created with {model.count_parameters():,} parameters")
    
    # Create trainer
    trainer = CloudTrainer(
        model=model,
        train_loader=train_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        logger=logger
    )
    
    # Start training
    trainer.train()
    logger.info("Training script completed successfully!")

if __name__ == "__main__":
    main()