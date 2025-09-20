#!/usr/bin/env python3
"""
Hugging Face Spaces Entry Point
This is the main entry point for the Hugging Face Spaces deployment.
It imports and launches the Gradio app from gradio_app.py
"""

import os
import sys
import warnings

# Suppress warnings for cleaner deployment logs
warnings.filterwarnings("ignore")

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the main app from gradio_app.py
    from gradio_app import create_app
    
    # Create and launch the Gradio interface
    app = create_app()
    
    # Launch with Hugging Face Spaces configuration
    if __name__ == "__main__":
        app.launch(
            server_name="0.0.0.0",  # Required for HF Spaces
            server_port=7860,       # HF Spaces default port
            show_error=True,        # Show errors in interface
            show_tips=False,        # Hide Gradio tips
            enable_queue=True,      # Enable request queuing
            max_threads=10          # Limit concurrent requests
        )

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Creating fallback demo app...")
    
    # Fallback demo if imports fail
    import gradio as gr
    
    def fallback_demo(text):
        return f"🚧 Demo Mode: You entered '{text}'\n\nThis is a fallback demo. The main model is not available yet."
    
    # Create simple fallback interface
    demo = gr.Interface(
        fn=fallback_demo,
        inputs=gr.Textbox(
            label="Enter some text",
            placeholder="Type something here...",
            lines=2
        ),
        outputs=gr.Textbox(
            label="Output",
            lines=4
        ),
        title="🤖 LLM Demo (Fallback Mode)",
        description="The main model is being set up. This is a temporary demo.",
        examples=[
            "Hello, how are you?",
            "Tell me a story",
            "What is AI?"
        ]
    )
    
    # Launch fallback demo
    if __name__ == "__main__":
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )

except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Last resort: minimal Gradio app
    import gradio as gr
    
    error_app = gr.Interface(
        fn=lambda x: f"Error: {str(e)}",
        inputs="text",
        outputs="text",
        title="🚨 Error Demo"
    )
    
    if __name__ == "__main__":
        error_app.launch(server_name="0.0.0.0", server_port=7860)