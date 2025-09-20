"""
Text generation and inference for the trained lightweight LLM
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import os
from typing import Optional, List

from model_config import ModelConfig
from light_llm import LightLLM

class TextGenerator:
    def __init__(self, model_path: str, device: str = "auto"):
        
        # Device setup
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.model = LightLLM(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1
    ) -> str:
        """Generate text continuation from a prompt"""
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.config.n_positions - max_new_tokens:
            tokens = tokens[-(self.config.n_positions - max_new_tokens):]
        
        x = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generation loop
        for _ in range(max_new_tokens):
            # Forward pass
            if x.size(1) >= self.config.n_positions:
                # If we exceed context length, take the last (n_positions-1) tokens
                x = x[:, -(self.config.n_positions-1):]
            
            logits, _ = self.model(x)
            logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(x[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            x = torch.cat((x, next_token), dim=1)
            
            # Check for end token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode generated text
        generated_tokens = x[0].tolist()
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text
    
    def interactive_generation(self):
        """Interactive text generation"""
        print("Interactive Text Generation")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            prompt = input("\nPrompt: ")
            if prompt.lower() == 'quit':
                break
            
            if prompt.strip():
                generated = self.generate(prompt, max_new_tokens=150)
                print(f"\nGenerated:\n{generated}")
                print("-" * 50)

def demo_generation(model_path: str = "checkpoints/best_model.pt"):
    """Demo text generation with sample prompts"""
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train a model first using train.py")
        return
    
    generator = TextGenerator(model_path)
    
    # Sample prompts
    prompts = [
        "Once upon a time",
        "The king said",
        "In the forest there lived",
        "Romeo and Juliet",
        "To be or not to be"
    ]
    
    print("Generating text with sample prompts...")
    print("=" * 60)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        generated = generator.generate(prompt, max_new_tokens=100, temperature=0.8)
        # Extract only the new part (after the prompt)
        if generated.startswith(prompt):
            new_text = generated[len(prompt):]
        else:
            new_text = generated
        print(f"Generated: {new_text}")
        print("=" * 60)

def evaluate_model(model_path: str = "checkpoints/best_model.pt"):
    """Evaluate model perplexity on test data"""
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    generator = TextGenerator(model_path)
    
    # Test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question",
        "Once upon a time in a land far away",
        "The sun was shining brightly in the clear blue sky"
    ]
    
    print("Evaluating model...")
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for sentence in test_sentences:
            tokens = generator.tokenizer.encode(sentence)
            if len(tokens) > 1:
                x = torch.tensor(tokens[:-1], dtype=torch.long, device=generator.device).unsqueeze(0)
                targets = torch.tensor(tokens[1:], dtype=torch.long, device=generator.device).unsqueeze(0)
                
                logits, loss = generator.model(x, targets)
                total_loss += loss.item() * len(tokens)
                total_tokens += len(tokens)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_generation()
        elif sys.argv[1] == "interactive":
            generator = TextGenerator("checkpoints/best_model.pt")
            generator.interactive_generation()
        elif sys.argv[1] == "eval":
            evaluate_model()
        else:
            print("Usage: python generate.py [demo|interactive|eval]")
    else:
        demo_generation()