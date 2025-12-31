"""
Vercel serverless function entry point for PyTorch Playground.

Note: Gradio apps have limitations on serverless platforms due to
WebSocket requirements and cold start times.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr

from pytorch_playground.app import create_app

# Create FastAPI app
app = FastAPI()

# Create Gradio app
gradio_app = create_app()

# Mount Gradio app on FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
