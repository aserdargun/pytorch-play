"""
Hugging Face Spaces entry point for PyTorch Playground.
Minimal test version to debug deployment issues.
"""

import gradio as gr

# Create a minimal test app first
with gr.Blocks(title="PyTorch Playground") as demo:
    gr.Markdown("# PyTorch Playground")
    gr.Markdown("Testing deployment...")

    with gr.Row():
        text_input = gr.Textbox(label="Input", placeholder="Type something...")
        text_output = gr.Textbox(label="Output")

    btn = gr.Button("Test")
    btn.click(fn=lambda x: f"You typed: {x}", inputs=text_input, outputs=text_output)

    gr.Markdown("---")
    gr.Markdown("If you see this, the Space is working! Full app coming soon.")

if __name__ == "__main__":
    demo.launch()
