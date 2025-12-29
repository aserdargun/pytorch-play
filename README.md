# PyTorch Playground

An educational web application for learning PyTorch concepts interactively, organized by skill level (Beginner/Intermediate/Advanced).

## Features

- **Level-based Learning**: Content and parameter complexity adapt to your chosen level
- **Device Support**: Automatic detection of CPU, CUDA (NVIDIA), MPS (Apple Silicon), and ROCm (AMD)
- **Interactive Demos**: Hands-on experimentation with PyTorch concepts
- **Install Wizard**: Generates installation commands mirroring the official PyTorch selector
- **Code Generation**: See the exact code behind every operation
- **Comprehensive Topic Catalog**: Links to official PyTorch tutorials, recipes, and videos

## Quick Start

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Installation

1. **Clone and navigate to the repository:**
   ```bash
   git clone <repository-url>
   cd pytorch-playground
   ```

2. **Install PyTorch first (IMPORTANT):**

   PyTorch installation depends on your system and desired compute platform.

   **Always verify the command on the official selector:** https://pytorch.org/get-started/locally/

   Example commands (verify before using):

   ```bash
   # CPU only (all platforms)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # CUDA 12.4 (Linux/Windows with NVIDIA GPU)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

   # macOS (CPU and MPS on Apple Silicon)
   pip install torch torchvision torchaudio
   ```

3. **Install the application with uv:**
   ```bash
   # Create virtual environment
   uv venv

   # Activate it
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows

   # Install dependencies (without torch, which should already be installed)
   uv pip install -e .
   ```

   **Alternative without uv:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e .
   ```

4. **Run the application:**
   ```bash
   python -m pytorch_playground.app
   ```

   Or with options:
   ```bash
   python -m pytorch_playground.app --host 0.0.0.0 --port 8080 --share
   ```

5. **Open your browser** to http://127.0.0.1:7860

## How Level Gating Works

The application adapts its content and complexity based on your selected level:

### Beginner (3-6 parameters per demo)
- **Focus**: Fundamentals and confidence building
- **Content**: Tensors, autograd basics, simple training loops
- **Style**: Minimal math, practical steps, clear outputs
- **Topics Visible**: Core concepts only

### Intermediate (6-12 parameters per demo)
- **Focus**: Modeling patterns and debugging
- **Content**: CNNs, regularization, schedulers, transfer learning
- **Style**: More technical details, common patterns
- **Topics Visible**: Beginner + intermediate topics

### Advanced (full parameter panel)
- **Focus**: Performance, systems, and deployment
- **Content**: torch.compile, profiling, distributed training concepts
- **Style**: Performance tradeoffs, memory considerations
- **Topics Visible**: All topics

## Application Structure

```
pytorch-playground/
├── pyproject.toml           # Package configuration
├── README.md                # This file
└── src/pytorch_playground/
    ├── __init__.py
    ├── app.py               # Main Gradio application
    ├── state.py             # Global state management
    ├── devices.py           # Device detection utilities
    ├── install_matrix.py    # Editable install options matrix
    ├── install_wizard.py    # Install command generator UI
    ├── topics/
    │   ├── catalog.py       # Topic catalog with level gating
    │   ├── basics.py        # Beginner explanations
    │   ├── intermediate.py  # Intermediate explanations
    │   ├── advanced.py      # Advanced explanations
    │   └── linkouts.py      # External resource links
    ├── demos/
    │   ├── tensor_lab.py    # Tensor exploration demo
    │   ├── autograd_lab.py  # Autograd demo
    │   ├── data_lab.py      # Data loading demo
    │   ├── model_builder.py # Neural network builder
    │   ├── train_eval.py    # Training loop demo
    │   ├── save_load.py     # Model persistence demo
    │   └── performance.py   # Performance optimization demo
    └── utils/
        ├── seeding.py       # Reproducibility utilities
        ├── plotting.py      # Visualization helpers
        ├── guards.py        # Safety and runtime limits
        └── logging.py       # Demo logging utilities
```

## Navigation Tabs

### Home (Dashboard)
- Environment information (Python, PyTorch versions)
- Device availability status
- Quick verification test
- Learning path suggestions

### Install Wizard
- Generates PyTorch installation commands
- Supports: Stable/Nightly, Linux/macOS/Windows, CPU/CUDA/ROCm/MPS
- Verification code snippets
- Links to official selector for confirmation

### Learn Hub
- **Tutorials**: Comprehensive guides with official links
- **Learn the Basics**: Step-by-step fundamental concepts
- **Recipes**: Bite-sized practical examples
- **YouTube Series**: Intro to PyTorch video links
- **Webinars**: Curated resources and blog posts

### Playground
Interactive demos for hands-on learning:

| Demo | Description | Level |
|------|-------------|-------|
| Tensor Lab | Create and inspect tensors | All |
| Autograd Lab | Explore automatic differentiation | All |
| Data Lab | Datasets and DataLoaders | All |
| Model Builder | Build neural network architectures | All |
| Train & Evaluate | Complete training loop | All |
| Save & Load | Model persistence | All |
| Performance | Optimization techniques | Intermediate+ |

### Settings
- About information
- Troubleshooting guide
- Version information

## Updating the Install Matrix

The install options in `src/pytorch_playground/install_matrix.py` mirror the official PyTorch selector. When PyTorch releases new versions or changes options:

1. Visit https://pytorch.org/get-started/locally/
2. Update the `COMPUTE_PLATFORMS` dictionary with new CUDA/ROCm versions
3. Update `INDEX_URLS` if the wheel URLs change
4. Update `BUILDS` if new build types are added

**Important**: Always verify installation commands on the official selector before use.

## Troubleshooting

### PyTorch Not Installed
1. Go to the Install Wizard tab
2. Select your OS and compute platform
3. Run the generated command
4. Restart the application

### CUDA Not Available
- Ensure NVIDIA drivers are installed (`nvidia-smi` to verify)
- Install the matching CUDA toolkit
- Use the correct PyTorch wheel for your CUDA version

### MPS Not Available (macOS)
- Requires macOS 12.3 or later
- Requires Apple Silicon (M1/M2/M3)
- PyTorch 1.12+ required

### DataLoader Worker Errors (Windows)
Set `num_workers=0` in the Data Lab demo, or wrap code in:
```python
if __name__ == '__main__':
    # your code here
```

### Memory Issues
- Reduce batch size
- Use smaller dataset subsets
- Enable mixed precision (CUDA)
- Reduce model complexity

## Development

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black src/
uv run ruff check src/
```

## Safety Features

The application includes several safety measures:

- **No arbitrary code execution**: All code runs through predefined demos
- **Runtime caps**: Training epochs and timing loops are bounded
- **Dataset size limits**: Based on user level
- **Platform guards**: Handles Windows DataLoader quirks
- **Graceful fallbacks**: Device unavailability handled without crashes

## Official PyTorch Resources

- [Start Locally (Install)](https://pytorch.org/get-started/locally/)
- [Tutorials](https://pytorch.org/tutorials/)
- [Documentation](https://pytorch.org/docs/stable/index.html)
- [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Recipes](https://docs.pytorch.org/tutorials/recipes_index.html)
- [YouTube Series](https://docs.pytorch.org/tutorials/beginner/introyt/introyt_index.html)
- [MPS Backend](https://docs.pytorch.org/docs/stable/notes/mps.html)

## License

MIT License

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing style
2. New demos include level-appropriate parameters
3. External links point to official PyTorch resources
4. Safety guards are maintained
