"""
Hugging Face Spaces entry point for PyTorch Playground.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pytorch_playground.app import create_app

# Create and launch app
app = create_app()

if __name__ == "__main__":
    app.launch()
