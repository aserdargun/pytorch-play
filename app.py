"""
Hugging Face Spaces entry point for PyTorch Playground.
"""

import gradio as gr
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pytorch_playground.app import create_app
    demo = create_app()
except Exception as e:
    # Fallback to simple app if main app fails
    with gr.Blocks() as demo:
        gr.Markdown(f"# PyTorch Playground")
        gr.Markdown(f"Error loading main app: {e}")
        gr.Markdown("Please check the logs for details.")

if __name__ == "__main__":
    demo.launch()
