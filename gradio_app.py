# Hugging Face Spaces Deployment App
import gradio as gr
import torch
import os
from typing import Optional

# Import our model components
try:
    from model_config import ModelConfig
    from light_llm import LightLLM
    from transformers import GPT2Tokenizer
    import torch.nn.functional as F
    import math
except ImportError as e:
    print(f"Import error: {e}")

class LightweightGenerator:
    def __init__(self, model_path: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use demo model or create a small random model for demo
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Create a demo model with random weights for demonstration
            self.create_demo_model()
    
    def create_demo_model(self):
        """Create a small demo model for testing (not trained)"""
        config = ModelConfig.nano()
        self.model = LightLLM(config)
        self.model.to(self.device)
        self.model.eval()
        print("Created demo model (random weights - for interface testing only)")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        self.model = LightLLM(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded trained model from {model_path}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9
    ) -> str:
        """Generate text continuation"""
        try:
            # Encode prompt
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) > 512:  # Limit context for demo
                tokens = tokens[-512:]
            
            x = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Generation loop
            for _ in range(max_new_tokens):
                # Forward pass
                if x.size(1) >= 1024:  # Context limit
                    x = x[:, -1023:]
                
                logits, _ = self.model(x)
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append token
                x = torch.cat((x, next_token), dim=1)
                
                # Check for end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            # Decode
            generated_tokens = x[0].tolist()
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text
        
        except Exception as e:
            return f"Generation error: {str(e)}"

# Initialize generator
generator = LightweightGenerator()

def generate_text(prompt, temperature, max_tokens, top_k, top_p):
    """Gradio interface function"""
    if not prompt.strip():
        return "Please enter a prompt!"
    
    # Add some safety limits
    max_tokens = min(max_tokens, 200)
    
    result = generator.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p
    )
    
    return result

# Create Gradio interface
with gr.Blocks(title="🤖 Lightweight LLM Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Lightweight LLM Text Generator")
    gr.Markdown("""
    This is a GPT-2 style transformer model trained from scratch! 
    
    **Note**: This demo uses a model with random weights for interface testing. 
    Upload a trained model checkpoint to see real text generation capabilities.
    
    **Features**: 
    - 🧠 Custom transformer architecture (30M parameters)
    - ⚡ Efficient attention mechanism
    - 🎛️ Advanced sampling controls
    - 📱 Responsive web interface
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="✍️ Enter your prompt",
                placeholder="Once upon a time...",
                lines=3
            )
            
            with gr.Row():
                temp_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="🌡️ Temperature",
                    info="Higher = more creative, Lower = more focused"
                )
                
                tokens_slider = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=100,
                    step=10,
                    label="📏 Max Tokens",
                    info="Maximum number of tokens to generate"
                )
            
            with gr.Row():
                topk_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=40,
                    step=1,
                    label="🔝 Top-K",
                    info="Limit vocabulary to top-k tokens"
                )
                
                topp_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="📊 Top-P",
                    info="Nucleus sampling threshold"
                )
            
            generate_btn = gr.Button("🚀 Generate Text", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="📝 Generated Text",
                lines=10,
                max_lines=15,
                show_copy_button=True
            )
            
            gr.Markdown("""
            ### 🎛️ Parameter Guide:
            - **Temperature**: Controls randomness (0.1 = conservative, 2.0 = very creative)
            - **Top-K**: Only consider the K most likely next tokens
            - **Top-P**: Only consider tokens that make up the top P probability mass
            """)
    
    # Examples
    gr.Examples(
        examples=[
            ["Once upon a time in a land far away", 0.8, 100, 40, 0.9],
            ["The future of artificial intelligence", 0.7, 80, 30, 0.85],
            ["In the beginning was the word", 0.9, 120, 50, 0.95],
            ["To be or not to be", 0.6, 60, 25, 0.8],
        ],
        inputs=[prompt_input, temp_slider, tokens_slider, topk_slider, topp_slider],
        label="💡 Try these examples:"
    )
    
    # Event handlers
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_input, temp_slider, tokens_slider, topk_slider, topp_slider],
        outputs=output_text
    )
    
    prompt_input.submit(
        fn=generate_text,
        inputs=[prompt_input, temp_slider, tokens_slider, topk_slider, topp_slider],
        outputs=output_text
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )