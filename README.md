---
title: PyTorch Playground
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
tags:
  - pytorch
  - education
  - machine-learning
  - deep-learning
---

# PyTorch Playground

An interactive web application for learning PyTorch concepts, organized by skill level.

## Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd pytorch-playground
uv venv && source .venv/bin/activate

# 2. Install PyTorch (verify command at https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio  # macOS
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu    # CPU only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4

# 3. Install app and run
uv pip install -e .
pytorch-playground  # Opens at http://127.0.0.1:7860
```

**Options:** `pytorch-playground --host 0.0.0.0 --port 8080 --share`

## Features

| Feature | Description |
|---------|-------------|
| **Level-based Learning** | UI complexity adapts to Beginner/Intermediate/Advanced |
| **Device Support** | Auto-detects CPU, CUDA, MPS (Apple Silicon), ROCm |
| **Interactive Demos** | Hands-on experimentation with live code generation |
| **Install Wizard** | Generates PyTorch install commands for any platform |
| **Topic Catalog** | Curated links to official tutorials, recipes, and videos |

## Skill Levels

| Level | Parameters | Focus | Content |
|-------|------------|-------|---------|
| **Beginner** | 3-6 | Fundamentals | Tensors, autograd, simple training |
| **Intermediate** | 6-12 | Patterns | CNNs, regularization, transfer learning |
| **Advanced** | Full | Performance | torch.compile, profiling, distributed |

## Demos

| Demo | Description | Min Level |
|------|-------------|-----------|
| Tensor Lab | Create and inspect tensors | Beginner |
| Autograd Lab | Automatic differentiation | Beginner |
| Data Lab | Datasets and DataLoaders | Beginner |
| Model Builder | Neural network architectures | Beginner |
| Train & Evaluate | Complete training loop | Beginner |
| Save & Load | Model persistence | Beginner |
| Performance | Optimization techniques | Intermediate |

## Project Structure

```
src/pytorch_playground/
├── app.py              # Main Gradio application
├── state.py            # Global state management
├── devices.py          # Device detection
├── install_wizard.py   # Install command generator
├── install_matrix.py   # PyTorch version/platform matrix
├── demos/              # Interactive demo modules
├── topics/             # Topic catalog and explanations
└── utils/              # Helpers (seeding, plotting, guards)
```

## Development

```bash
uv run pytest              # Run tests
uv run black src/          # Format code
uv run ruff check src/     # Lint
```

### Updating the Install Matrix

When PyTorch releases new versions, update `src/pytorch_playground/install_matrix.py`:
- `COMPUTE_PLATFORMS` - CUDA/ROCm versions
- `INDEX_URLS` - Wheel URLs
- `BUILDS` - Build types (stable/nightly)

Always verify against https://pytorch.org/get-started/locally/

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PyTorch not installed | Use Install Wizard tab to generate the correct command |
| CUDA not available | Verify drivers with `nvidia-smi`, match CUDA version to PyTorch wheel |
| MPS not available | Requires macOS 12.3+, Apple Silicon, PyTorch 1.12+ |
| DataLoader errors (Windows) | Set `num_workers=0` |
| Memory issues | Reduce batch size, use smaller datasets, enable mixed precision |

## Safety

- No arbitrary code execution (predefined demos only)
- Runtime caps on training epochs
- Dataset size limits per level
- Graceful device fallbacks

## Resources

- [PyTorch Install Selector](https://pytorch.org/get-started/locally/)
- [Tutorials](https://pytorch.org/tutorials/)
- [Documentation](https://pytorch.org/docs/stable/index.html)
- [MPS Backend Guide](https://pytorch.org/docs/stable/notes/mps.html)

## Contributing

1. Follow existing code style
2. Include level-appropriate parameters in new demos
3. Link only to official PyTorch resources
4. Maintain safety guards

## License

MIT
