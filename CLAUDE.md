# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch Playground is an educational Gradio web application for learning PyTorch concepts, organized by skill level (Beginner/Intermediate/Advanced). It features interactive demos, an install wizard, and links to official PyTorch resources.

## Commands

```bash
# Setup (requires Python 3.10+ due to union type syntax)
uv venv --python 3.12
uv sync
uv pip install torch

# Run the app
source .venv/bin/activate
pytorch-playground
# Or: python -m pytorch_playground.app --host 0.0.0.0 --port 8080

# Development
uv run pytest                    # Run tests
uv run black src/                # Format code
uv run ruff check src/           # Lint
```

## Dependency Constraints

The app requires specific package versions due to Gradio 4.x compatibility:
- `gradio>=4.0.0,<5.0.0` (Gradio 6.x has breaking API changes)
- `huggingface-hub<0.24` (newer versions break gradio imports)
- `pydantic<2.10`, `starlette<0.40`, `fastapi<0.115` (for gradio-client compatibility)

PyTorch must be installed separately via the official selector at https://pytorch.org/get-started/locally/

## Architecture

### State Management (`state.py`)
- `Level` enum controls UI complexity: BEGINNER (3-6 params), INTERMEDIATE (6-12 params), ADVANCED (full panel)
- `AppState` is a thread-safe singleton accessed via `get_state()`
- Level affects: max epochs, explanation depth, code visibility, parameter counts in demos

### Demo Pattern (`demos/`)
Each demo class follows a consistent pattern:
1. `get_params_for_level(level)` - Returns parameter config dict based on user level
2. `create_ui()` - Builds Gradio interface with level-gated parameters
3. Demo logic that respects level constraints (e.g., `level.max_epochs`)

Demos: `tensor_lab.py`, `autograd_lab.py`, `data_lab.py`, `model_builder.py`, `train_eval.py`, `save_load.py`, `performance.py`

### Topics System (`topics/`)
- `catalog.py` - Master topic registry with `Topic` dataclass containing level gating, categories, and optional demo references
- `basics.py`, `intermediate.py`, `advanced.py` - Level-specific explanations
- `linkouts.py` - External resource links to official PyTorch docs

### Device Detection (`devices.py`)
- `detect_devices()` - Returns available devices (CPU, CUDA, MPS)
- `get_device(selection)` - Resolves "auto" to best available device

### Install Wizard (`install_matrix.py`, `install_wizard.py`)
- Mirrors the official PyTorch "Start Locally" selector
- `COMPUTE_PLATFORMS`, `INDEX_URLS`, `BUILDS` dicts define available options
- Update these when PyTorch releases new CUDA/ROCm versions

## Key Patterns

When adding Gradio dropdowns, always include `choices` even for hidden placeholders:
```python
# Correct - prevents "value not in choices" warnings
gr.Dropdown(choices=["default"], value="default", visible=False)

# Wrong - causes warnings
gr.Dropdown(value="default", visible=False)
```

Level gating in demos should check `get_state().level` and adjust UI/behavior accordingly.
