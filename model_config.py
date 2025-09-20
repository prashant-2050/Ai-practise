"""
Lightweight LLM Configuration for Local Training
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # Model architecture
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    n_positions: int = 1024  # max sequence length
    n_embd: int = 384  # embedding dimension
    n_layer: int = 6   # number of transformer layers
    n_head: int = 6    # number of attention heads
    
    # Training config
    dropout: float = 0.1
    bias: bool = True
    
    # Model size variants
    @classmethod
    def nano(cls):
        """~1M parameters"""
        return cls(
            n_embd=192,
            n_layer=6,
            n_head=6,
            n_positions=512
        )
    
    @classmethod
    def micro(cls):
        """~10M parameters"""
        return cls(
            n_embd=384,
            n_layer=6,
            n_head=6,
            n_positions=1024
        )
    
    @classmethod
    def small(cls):
        """~25M parameters"""
        return cls(
            n_embd=512,
            n_layer=8,
            n_head=8,
            n_positions=1024
        )

def get_model_size(config: ModelConfig) -> int:
    """Estimate model parameters"""
    # Rough calculation
    emb_params = config.vocab_size * config.n_embd
    pos_params = config.n_positions * config.n_embd
    
    # Each layer: attention + mlp
    layer_params = (
        # Attention: qkv projection + output projection
        4 * config.n_embd * config.n_embd +
        # MLP: 2 linear layers (4x expansion)
        2 * config.n_embd * (4 * config.n_embd)
    )
    
    total_params = emb_params + pos_params + (config.n_layer * layer_params)
    return total_params

if __name__ == "__main__":
    configs = {
        "nano": ModelConfig.nano(),
        "micro": ModelConfig.micro(), 
        "small": ModelConfig.small()
    }
    
    for name, config in configs.items():
        size = get_model_size(config)
        print(f"{name}: {size:,} parameters")
        print(f"  - layers: {config.n_layer}")
        print(f"  - embd: {config.n_embd}")
        print(f"  - heads: {config.n_head}")
        print(f"  - context: {config.n_positions}")
        print()