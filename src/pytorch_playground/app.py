"""
PyTorch Playground - Main Gradio Application

An educational web app for learning PyTorch concepts at different levels.
"""

import gradio as gr
import sys
import platform
from typing import Optional

from pytorch_playground.state import (
    Level,
    DeviceSelection,
    get_state,
    reset_state,
    get_level_choices,
    get_device_choices,
)
from pytorch_playground.devices import (
    detect_devices,
    get_device,
    get_torch_info,
    get_environment_info,
)
from pytorch_playground.install_wizard import create_install_wizard_ui
from pytorch_playground.topics.catalog import (
    get_catalog,
    TopicCategory,
    filter_by_level,
)
from pytorch_playground.topics.linkouts import (
    get_all_tutorials,
    get_all_youtube,
    get_all_webinars,
    get_resources_by_level,
)

# Import demos
from pytorch_playground.demos.tensor_lab import TensorLabDemo
from pytorch_playground.demos.autograd_lab import AutogradLabDemo
from pytorch_playground.demos.data_lab import DataLabDemo
from pytorch_playground.demos.model_builder import ModelBuilderDemo
from pytorch_playground.demos.train_eval import TrainEvalDemo
from pytorch_playground.demos.save_load import SaveLoadDemo
from pytorch_playground.demos.performance import PerformanceDemo


# CSS for consistent styling
CUSTOM_CSS = """
.global-controls {
    background-color: #f7f7f7;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 10px;
}
.level-badge {
    font-weight: bold;
    padding: 4px 8px;
    border-radius: 4px;
}
.level-beginner { background-color: #e3f2fd; color: #1565c0; }
.level-intermediate { background-color: #fff3e0; color: #ef6c00; }
.level-advanced { background-color: #fce4ec; color: #c2185b; }
.device-badge {
    font-size: 0.9em;
    padding: 2px 6px;
    border-radius: 3px;
    margin-right: 4px;
}
.device-available { background-color: #c8e6c9; color: #2e7d32; }
.device-unavailable { background-color: #ffcdd2; color: #c62828; }
"""


def create_home_tab():
    """Create the Home/Dashboard tab."""
    with gr.Blocks() as home:
        gr.Markdown("# PyTorch Playground")
        gr.Markdown("*Learn PyTorch interactively at your own pace*")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Environment")

                # Get environment info
                env_info = get_environment_info()
                torch_info = get_torch_info()

                if torch_info.get("installed"):
                    torch_status = f"""
**PyTorch Status:** Installed

| Component | Version |
|-----------|---------|
| Python | {env_info.get('python_version', 'Unknown').split()[0]} |
| PyTorch | {torch_info.get('version', 'Unknown')} |
| torchvision | {torch_info.get('torchvision_version') or 'Not installed'} |
| torchaudio | {torch_info.get('torchaudio_version') or 'Not installed'} |
| CUDA | {torch_info.get('cuda_version') or 'N/A'} |
"""
                else:
                    torch_status = """
**PyTorch Status:** Not installed

Please use the **Install Wizard** tab to get installation instructions.
"""

                gr.Markdown(torch_status)

                gr.Markdown("## Available Devices")
                devices = detect_devices()
                device_info = ""
                for d in devices:
                    status = "[Available]" if d.available else "[Not Available]"
                    device_info += f"- **{d.name}**: {status}"
                    if d.device_name:
                        device_info += f" - {d.device_name}"
                    device_info += "\n"
                gr.Markdown(device_info)

            with gr.Column(scale=1):
                gr.Markdown("## Quick Actions")

                verify_btn = gr.Button("Run Verification", variant="primary")
                verify_output = gr.Textbox(label="Verification Result", lines=10, interactive=False)

                def run_verification():
                    """Run PyTorch verification."""
                    try:
                        import torch

                        lines = []
                        lines.append(f"PyTorch version: {torch.__version__}")
                        lines.append("")

                        # Random tensor test
                        x = torch.rand(5, 3)
                        lines.append("Random tensor (5x3):")
                        lines.append(str(x))
                        lines.append("")

                        # Device tests
                        lines.append("Device availability:")
                        lines.append(f"  CUDA: {torch.cuda.is_available()}")
                        if torch.cuda.is_available():
                            lines.append(f"  CUDA device: {torch.cuda.get_device_name(0)}")

                        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                        mps_built = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
                        lines.append(f"  MPS available: {mps_available}")
                        lines.append(f"  MPS built: {mps_built}")

                        lines.append("")
                        lines.append("Verification successful!")

                        return "\n".join(lines)

                    except ImportError:
                        return "PyTorch is not installed. Please install it first using the Install Wizard."
                    except Exception as e:
                        return f"Error during verification: {str(e)}"

                verify_btn.click(fn=run_verification, outputs=verify_output)

                gr.Markdown("## Learning Paths")
                gr.Markdown("""
Choose your learning path based on your experience level:

**Beginner Path:**
1. Tensors basics
2. Autograd fundamentals
3. Simple neural network
4. Training loop
5. Save/Load models

**Intermediate Path:**
1. CNNs for vision
2. Data loading optimization
3. Regularization techniques
4. Transfer learning
5. Debugging & metrics

**Advanced Path:**
1. Performance optimization
2. Mixed precision training
3. torch.compile
4. Distributed training concepts
5. Production deployment
""")

        gr.Markdown("---")
        gr.Markdown("""
### Quick Links

- [Official PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Start Locally (Install)](https://pytorch.org/get-started/locally/)
""")

    return home


def create_learn_hub_tab(level: Level):
    """Create the Learn Hub tab with topic catalog."""
    catalog = get_catalog()

    with gr.Blocks() as learn_hub:
        gr.Markdown("# Learn Hub")
        gr.Markdown("Explore PyTorch topics organized by category")

        with gr.Tabs():
            # Tutorials tab
            with gr.Tab("Tutorials"):
                tutorials = catalog.get_topics_by_category(TopicCategory.TUTORIALS)
                tutorials = filter_by_level(tutorials, level)

                if tutorials:
                    for topic in tutorials:
                        with gr.Accordion(topic.title, open=False):
                            gr.Markdown(f"**Level:** {topic.level_min.value}")
                            gr.Markdown(topic.description)
                            if topic.official_link:
                                gr.Markdown(f"[Official Tutorial]({topic.official_link})")
                            if topic.runnable:
                                gr.Markdown(f"*Runnable demo available in Playground: {topic.demo_id}*")
                else:
                    gr.Markdown("No tutorials available at this level.")

                # External tutorials
                gr.Markdown("### Official Tutorial Links")
                for resource in get_all_tutorials()[:10]:
                    gr.Markdown(f"- [{resource.title}]({resource.url}) - {resource.description}")

            # Learn the Basics tab
            with gr.Tab("Learn the Basics"):
                basics = catalog.get_topics_by_category(TopicCategory.BASICS)
                basics = filter_by_level(basics, level)

                gr.Markdown("""
The basics path covers fundamental PyTorch concepts step by step.
[Official Learn the Basics Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
""")

                for topic in basics:
                    with gr.Accordion(topic.title, open=False):
                        gr.Markdown(f"**Level:** {topic.level_min.value}")
                        gr.Markdown(topic.get_explanation(level))
                        if topic.outcomes:
                            gr.Markdown("**Learning Outcomes:**")
                            for outcome in topic.outcomes:
                                gr.Markdown(f"- {outcome}")
                        if topic.official_link:
                            gr.Markdown(f"[Official Tutorial]({topic.official_link})")
                        if topic.runnable:
                            gr.Markdown(f"*Try the interactive demo: {topic.demo_id}*")

            # Recipes tab
            with gr.Tab("Recipes"):
                recipes = catalog.get_topics_by_category(TopicCategory.RECIPES)
                recipes = filter_by_level(recipes, level)

                gr.Markdown("""
PyTorch Recipes are bite-sized, actionable examples for specific tasks.
[Official Recipes Index](https://docs.pytorch.org/tutorials/recipes_index.html)
""")

                for topic in recipes:
                    with gr.Accordion(topic.title, open=False):
                        gr.Markdown(f"**Level:** {topic.level_min.value}")
                        gr.Markdown(topic.description)
                        if topic.tags:
                            gr.Markdown(f"**Tags:** {', '.join(topic.tags)}")
                        if topic.official_link:
                            gr.Markdown(f"[Official Recipe]({topic.official_link})")

            # YouTube Series tab
            with gr.Tab("YouTube Series"):
                youtube = catalog.get_topics_by_category(TopicCategory.YOUTUBE)
                youtube = filter_by_level(youtube, level)

                gr.Markdown("""
## Intro to PyTorch - YouTube Series

Video tutorials covering PyTorch fundamentals.

[Series Index](https://docs.pytorch.org/tutorials/beginner/introyt/introyt_index.html)
""")

                for resource in get_all_youtube():
                    gr.Markdown(f"- [{resource.title}]({resource.url})")
                    gr.Markdown(f"  *{resource.description}*")

            # Webinars tab
            with gr.Tab("Webinars & Blog"):
                gr.Markdown("""
## PyTorch Webinars & Resources

Curated list of webinars, blog posts, and official resources.
""")

                for resource in get_all_webinars():
                    gr.Markdown(f"### [{resource.title}]({resource.url})")
                    gr.Markdown(f"{resource.description}")
                    gr.Markdown(f"*Level: {resource.level}*")
                    gr.Markdown("")

    return learn_hub


def create_playground_tab(level: Level):
    """Create the Playground tab with interactive demos."""
    with gr.Blocks() as playground:
        gr.Markdown("# Playground")
        gr.Markdown("Interactive demos to practice PyTorch concepts")

        with gr.Tabs():
            with gr.Tab("Tensor Lab"):
                demo = TensorLabDemo()
                demo.create_ui(level)

            with gr.Tab("Autograd Lab"):
                demo = AutogradLabDemo()
                demo.create_ui(level)

            with gr.Tab("Data Lab"):
                demo = DataLabDemo()
                demo.create_ui(level)

            with gr.Tab("Model Builder"):
                demo = ModelBuilderDemo()
                demo.create_ui(level)

            with gr.Tab("Train & Evaluate"):
                demo = TrainEvalDemo()
                demo.create_ui(level)

            with gr.Tab("Save & Load"):
                demo = SaveLoadDemo()
                demo.create_ui(level)

            if level in [Level.INTERMEDIATE, Level.ADVANCED]:
                with gr.Tab("Performance"):
                    demo = PerformanceDemo()
                    demo.create_ui(level)

    return playground


def create_settings_tab():
    """Create the Settings/About tab."""
    with gr.Blocks() as settings:
        gr.Markdown("# Settings & About")

        with gr.Tabs():
            with gr.Tab("About"):
                gr.Markdown("""
## PyTorch Playground

An educational web application for learning PyTorch concepts interactively.

### Features
- **Level-based learning**: Beginner, Intermediate, and Advanced paths
- **Device support**: CPU, CUDA, MPS (Apple Silicon), and ROCm
- **Interactive demos**: Hands-on experimentation with PyTorch concepts
- **Code generation**: See the code behind every operation

### How Levels Work

**Beginner** (3-6 parameters)
- Focus on fundamentals and confidence building
- Minimal math, practical steps
- Core concepts: tensors, autograd, basic training

**Intermediate** (6-12 parameters)
- Modeling patterns and debugging
- CNN, regularization, schedulers
- Debugging and stability techniques

**Advanced** (full panel)
- Performance and systems
- torch.compile, profiling, distributed concepts
- Production considerations

### Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [PyTorch Forums](https://discuss.pytorch.org/)
""")

            with gr.Tab("Troubleshooting"):
                gr.Markdown("""
## Common Issues

### PyTorch Not Found
If you see "PyTorch not installed":
1. Go to the **Install Wizard** tab
2. Select your OS and compute platform
3. Copy and run the install command
4. Restart this application

### CUDA Not Available
- Ensure you have NVIDIA drivers installed
- Install the CUDA toolkit matching your PyTorch build
- Verify with: `nvidia-smi`

### MPS Not Available (macOS)
- Requires macOS 12.3 or later
- Requires Apple Silicon (M1/M2/M3) or AMD GPU
- Install PyTorch 1.12+ for MPS support

### Memory Issues
- Reduce batch size
- Use smaller models
- Enable mixed precision (CUDA)
- Use gradient checkpointing

### DataLoader Worker Errors (Windows)
- Set `num_workers=0`
- Or wrap your code in `if __name__ == '__main__':`

### Import Errors
Ensure all dependencies are installed:
```bash
pip install gradio numpy pandas matplotlib scikit-learn
```
""")

            with gr.Tab("Version Info"):
                env_info = get_environment_info()

                version_text = f"""
## System Information

| Component | Value |
|-----------|-------|
| Platform | {env_info.get('platform', 'Unknown')} |
| Python | {env_info.get('python_version', 'Unknown').split()[0]} |
| Machine | {env_info.get('machine', 'Unknown')} |

## PyTorch Versions
"""
                torch_info = get_torch_info()
                if torch_info.get('installed'):
                    version_text += f"""
| Package | Version |
|---------|---------|
| torch | {torch_info.get('version', 'N/A')} |
| torchvision | {torch_info.get('torchvision_version') or 'N/A'} |
| torchaudio | {torch_info.get('torchaudio_version') or 'N/A'} |
| CUDA | {torch_info.get('cuda_version') or 'N/A'} |
| HIP/ROCm | {torch_info.get('hip_version') or 'N/A'} |
"""
                else:
                    version_text += "\n*PyTorch not installed*"

                gr.Markdown(version_text)

    return settings


def create_app():
    """Create the main Gradio application."""
    state = get_state()

    with gr.Blocks(
        title="PyTorch Playground",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    ) as app:
        # Global controls at the top
        with gr.Row(elem_classes=["global-controls"]):
            with gr.Column(scale=2):
                level_dropdown = gr.Dropdown(
                    choices=get_level_choices(),
                    value=state.level.value,
                    label="Level",
                    info="Controls topic visibility and parameter complexity",
                    interactive=True,
                )

            with gr.Column(scale=2):
                device_dropdown = gr.Dropdown(
                    choices=get_device_choices(),
                    value=state.device_selection.value,
                    label="Device",
                    info="Target compute device for demos",
                    interactive=True,
                )

            with gr.Column(scale=1):
                seed_input = gr.Number(
                    value=state.seed,
                    label="Seed",
                    precision=0,
                    interactive=True,
                )

            with gr.Column(scale=1):
                deterministic_check = gr.Checkbox(
                    value=state.deterministic,
                    label="Deterministic",
                    info="Enable deterministic mode",
                    interactive=True,
                )

            with gr.Column(scale=1):
                reset_btn = gr.Button("Reset Session", size="sm")

        # Device status badges
        devices = detect_devices()
        device_badges = []
        for d in devices:
            status = "Available" if d.available else "N/A"
            device_badges.append(f"**{d.name}**: {status}")
        gr.Markdown(" | ".join(device_badges))

        # State change handlers
        def on_level_change(level_str):
            level = Level(level_str)
            state.set_level(level)
            return f"Level changed to {level.value}"

        def on_device_change(device_str):
            device = DeviceSelection(device_str)
            state.set_device(device)
            return f"Device changed to {device.value}"

        def on_seed_change(seed):
            state.set_seed(int(seed) if seed else None)
            return f"Seed set to {seed}"

        def on_deterministic_change(det):
            state.set_deterministic(det)
            return f"Deterministic mode: {det}"

        def on_reset():
            reset_state()
            return (
                Level.BEGINNER.value,
                DeviceSelection.AUTO.value,
                42,
                False,
                "Session reset to defaults",
            )

        # Hidden status for feedback
        status_text = gr.Textbox(visible=False)

        level_dropdown.change(on_level_change, level_dropdown, status_text)
        device_dropdown.change(on_device_change, device_dropdown, status_text)
        seed_input.change(on_seed_change, seed_input, status_text)
        deterministic_check.change(on_deterministic_change, deterministic_check, status_text)
        reset_btn.click(
            on_reset,
            outputs=[level_dropdown, device_dropdown, seed_input, deterministic_check, status_text],
        )

        # Main navigation tabs
        with gr.Tabs():
            with gr.Tab("Home"):
                create_home_tab()

            with gr.Tab("Install Wizard"):
                create_install_wizard_ui()

            with gr.Tab("Learn Hub"):
                # Create with initial level (updates require app reload)
                create_learn_hub_tab(state.level)

            with gr.Tab("Playground"):
                create_playground_tab(state.level)

            with gr.Tab("Settings"):
                create_settings_tab()

    return app


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Playground")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("=" * 60)
    print("PyTorch Playground")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            print(f"MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("PyTorch: Not installed")
        print("Use the Install Wizard to get installation instructions.")

    print("=" * 60)
    print(f"Starting server on http://{args.host}:{args.port}")
    print("=" * 60)

    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
