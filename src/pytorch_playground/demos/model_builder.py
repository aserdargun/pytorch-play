"""
Model Builder Demo - Interactive neural network construction.

Covers:
- Building MLP architectures
- Layer types (Linear, Conv2d, etc.)
- Activations and normalization
- Parameter inspection
"""

import gradio as gr
from typing import Tuple, Optional, Any, Dict, List

from pytorch_playground.state import Level
from pytorch_playground.devices import get_device
from pytorch_playground.utils.seeding import set_seed
from pytorch_playground.utils.logging import DemoLogger, CodeBuilder, format_model_info
import matplotlib.pyplot as plt
import numpy as np


class ModelBuilderDemo:
    """Interactive model building demo."""

    def __init__(self):
        self.name = "Model Builder"
        self.description = "Build and inspect neural network architectures"

    def get_params_for_level(self, level: Level) -> Dict[str, Any]:
        """Get parameter configuration based on level."""
        if level == Level.BEGINNER:
            return {
                "max_layers": 4,
                "activations": ["ReLU", "Sigmoid", "Tanh"],
                "show_dropout": False,
                "show_batchnorm": False,
                "show_init": False,
            }
        elif level == Level.INTERMEDIATE:
            return {
                "max_layers": 6,
                "activations": ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU"],
                "show_dropout": True,
                "show_batchnorm": True,
                "show_init": False,
            }
        else:
            return {
                "max_layers": 10,
                "activations": ["ReLU", "LeakyReLU", "PReLU", "Sigmoid", "Tanh", "GELU", "SiLU", "Mish"],
                "show_dropout": True,
                "show_batchnorm": True,
                "show_init": True,
            }

    def run(
        self,
        input_size: int,
        hidden_sizes_str: str,
        output_size: int,
        activation: str,
        use_dropout: bool,
        dropout_rate: float,
        use_batchnorm: bool,
        init_method: str,
        device_selection: str,
        seed: int,
    ) -> Tuple[str, Dict[str, Any], Any, str]:
        """Build and analyze the model."""
        import torch
        import torch.nn as nn

        logger = DemoLogger(self.name)
        logger.start()

        try:
            set_seed(seed)
            logger.info(f"Seed set to {seed}")

            # Get device
            device, device_label, warning = get_device(device_selection)
            if warning:
                logger.warning(warning)
            logger.info(f"Device: {device_label}")

            # Parse hidden sizes
            try:
                hidden_sizes = [int(x.strip()) for x in hidden_sizes_str.split(",") if x.strip()]
                if not hidden_sizes:
                    hidden_sizes = [64, 32]
            except ValueError:
                logger.warning("Invalid hidden sizes, using default [64, 32]")
                hidden_sizes = [64, 32]

            # Build layer sizes
            layer_sizes = [input_size] + hidden_sizes + [output_size]
            logger.info(f"Layer sizes: {layer_sizes}")

            # Get activation
            activation_map = {
                "ReLU": nn.ReLU,
                "LeakyReLU": lambda: nn.LeakyReLU(0.1),
                "PReLU": nn.PReLU,
                "Sigmoid": nn.Sigmoid,
                "Tanh": nn.Tanh,
                "GELU": nn.GELU,
                "SiLU": nn.SiLU,
                "Mish": nn.Mish,
            }
            act_fn = activation_map.get(activation, nn.ReLU)

            # Build model
            layers = []

            for i in range(len(layer_sizes) - 1):
                in_features = layer_sizes[i]
                out_features = layer_sizes[i + 1]
                is_last = (i == len(layer_sizes) - 2)

                # Linear layer
                layers.append(nn.Linear(in_features, out_features))

                if not is_last:
                    # Batch normalization
                    if use_batchnorm:
                        layers.append(nn.BatchNorm1d(out_features))

                    # Activation
                    layers.append(act_fn() if callable(act_fn) else act_fn)

                    # Dropout
                    if use_dropout and dropout_rate > 0:
                        layers.append(nn.Dropout(dropout_rate))

            model = nn.Sequential(*layers)

            # Weight initialization
            if init_method != "default":
                self._initialize_weights(model, init_method)
                logger.info(f"Weights initialized with: {init_method}")

            # Move to device
            model = model.to(device)

            logger.info("")
            logger.info("--- Model Architecture ---")
            for i, layer in enumerate(model):
                logger.info(f"  [{i}] {layer}")

            logger.info("")
            logger.info(format_model_info(model, "Model"))

            # Test forward pass
            logger.info("")
            logger.info("--- Test Forward Pass ---")
            test_input = torch.randn(4, input_size, device=device)
            logger.info(f"Input shape: {tuple(test_input.shape)}")

            with torch.no_grad():
                output = model(test_input)

            logger.info(f"Output shape: {tuple(output.shape)}")

            # Metrics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            logger.metric("total_params", total_params)
            logger.metric("trainable_params", trainable_params)
            logger.metric("num_layers", len(list(model.modules())) - 1)
            logger.metric("input_size", input_size)
            logger.metric("output_size", output_size)

            # Visualization
            fig = self._create_model_plot(layer_sizes, activation, use_dropout, use_batchnorm)

            # Code
            code = self._build_code(
                input_size, hidden_sizes, output_size, activation,
                use_dropout, dropout_rate, use_batchnorm, init_method
            )

            logger.end(success=True)
            return logger.format_logs(), logger.metrics, fig, code

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            logger.end(success=False)
            return logger.format_logs(), {}, None, ""

    def _initialize_weights(self, model, method: str):
        """Apply weight initialization."""
        import torch.nn as nn

        for m in model.modules():
            if isinstance(m, nn.Linear):
                if method == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif method == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif method == "orthogonal":
                    nn.init.orthogonal_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _create_model_plot(
        self,
        layer_sizes: List[int],
        activation: str,
        use_dropout: bool,
        use_batchnorm: bool,
    ) -> Any:
        """Create model architecture visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Network diagram
        max_neurons = max(layer_sizes)
        n_layers = len(layer_sizes)

        for layer_idx, n_neurons in enumerate(layer_sizes):
            x = layer_idx
            spacing = 1.0 / (n_neurons + 1)

            for neuron_idx in range(min(n_neurons, 10)):  # Cap at 10 for visibility
                y = (neuron_idx + 1) * spacing

                # Draw neuron
                circle = plt.Circle((x, y), 0.03, color='steelblue', fill=True)
                ax1.add_patch(circle)

                # Draw connections to next layer
                if layer_idx < n_layers - 1:
                    next_n = min(layer_sizes[layer_idx + 1], 10)
                    next_spacing = 1.0 / (next_n + 1)

                    for next_idx in range(next_n):
                        next_y = (next_idx + 1) * next_spacing
                        ax1.plot([x, x + 1], [y, next_y], 'gray', alpha=0.2, linewidth=0.5)

            # Add "..." if truncated
            if n_neurons > 10:
                ax1.text(x, 0.5, f"({n_neurons})", ha='center', va='center', fontsize=8)

        # Layer labels
        for i, size in enumerate(layer_sizes):
            label = "Input" if i == 0 else ("Output" if i == len(layer_sizes) - 1 else f"Hidden {i}")
            ax1.text(i, -0.1, f"{label}\n({size})", ha='center', va='top', fontsize=9)

        ax1.set_xlim(-0.5, n_layers - 0.5)
        ax1.set_ylim(-0.3, 1.1)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title("Network Architecture")

        # Right: Parameter distribution
        params_per_layer = []
        layer_names = []

        for i in range(len(layer_sizes) - 1):
            in_f = layer_sizes[i]
            out_f = layer_sizes[i + 1]
            params = in_f * out_f + out_f  # weights + bias

            if use_batchnorm and i < len(layer_sizes) - 2:
                params += 2 * out_f  # gamma, beta

            params_per_layer.append(params)
            layer_names.append(f"Layer {i + 1}")

        bars = ax2.bar(layer_names, params_per_layer, color='coral')
        ax2.set_ylabel("Number of Parameters")
        ax2.set_title("Parameters per Layer")

        # Add value labels
        for bar, val in zip(bars, params_per_layer):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return fig

    def _build_code(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str,
        use_dropout: bool,
        dropout_rate: float,
        use_batchnorm: bool,
        init_method: str,
    ) -> str:
        """Build the code string."""
        builder = CodeBuilder()

        builder.add("import torch")
        builder.add("import torch.nn as nn")
        builder.add_blank()

        builder.add("class MLP(nn.Module):")
        builder.indent()

        # __init__
        builder.add(f"def __init__(self, input_size={input_size}, hidden_sizes={hidden_sizes}, output_size={output_size}):")
        builder.indent()
        builder.add("super().__init__()")
        builder.add_blank()

        builder.add("layer_sizes = [input_size] + hidden_sizes + [output_size]")
        builder.add("layers = []")
        builder.add_blank()

        builder.add("for i in range(len(layer_sizes) - 1):")
        builder.indent()
        builder.add("in_features = layer_sizes[i]")
        builder.add("out_features = layer_sizes[i + 1]")
        builder.add("is_last = (i == len(layer_sizes) - 2)")
        builder.add_blank()

        builder.add("layers.append(nn.Linear(in_features, out_features))")
        builder.add_blank()

        builder.add("if not is_last:")
        builder.indent()

        if use_batchnorm:
            builder.add("layers.append(nn.BatchNorm1d(out_features))")

        builder.add(f"layers.append(nn.{activation}())")

        if use_dropout:
            builder.add(f"layers.append(nn.Dropout({dropout_rate}))")

        builder.dedent()
        builder.dedent()
        builder.add_blank()

        builder.add("self.network = nn.Sequential(*layers)")

        if init_method != "default":
            builder.add("self._initialize_weights()")

        builder.dedent()
        builder.add_blank()

        # forward
        builder.add("def forward(self, x):")
        builder.indent()
        builder.add("return self.network(x)")
        builder.dedent()

        if init_method != "default":
            builder.add_blank()
            builder.add("def _initialize_weights(self):")
            builder.indent()
            builder.add("for m in self.modules():")
            builder.indent()
            builder.add("if isinstance(m, nn.Linear):")
            builder.indent()
            if "xavier" in init_method:
                builder.add(f"nn.init.{init_method}_(m.weight)")
            elif "kaiming" in init_method:
                builder.add(f"nn.init.{init_method}_(m.weight, nonlinearity='relu')")
            builder.add("if m.bias is not None:")
            builder.indent()
            builder.add("nn.init.zeros_(m.bias)")
            builder.dedent()
            builder.dedent()
            builder.dedent()
            builder.dedent()

        builder.dedent()
        builder.add_blank()

        builder.add_comment("Create and test model")
        builder.add("model = MLP()")
        builder.add("print(model)")
        builder.add_blank()

        builder.add_comment("Count parameters")
        builder.add("total_params = sum(p.numel() for p in model.parameters())")
        builder.add("print(f'Total parameters: {total_params:,}')")
        builder.add_blank()

        builder.add_comment("Test forward pass")
        builder.add(f"x = torch.randn(4, {input_size})")
        builder.add("output = model(x)")
        builder.add("print(f'Output shape: {output.shape}')")

        return builder.build()

    def create_ui(self, level: Level) -> gr.Blocks:
        """Create the Gradio UI."""
        params = self.get_params_for_level(level)

        with gr.Blocks() as demo:
            gr.Markdown(f"## {self.name}")
            gr.Markdown(self.description)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Architecture")

                    input_size = gr.Slider(
                        minimum=1,
                        maximum=1024,
                        value=10,
                        step=1,
                        label="Input Size",
                    )

                    hidden_sizes = gr.Textbox(
                        label="Hidden Layer Sizes (comma-separated)",
                        value="64, 32",
                        info="e.g., '128, 64, 32' for 3 hidden layers",
                    )

                    output_size = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=2,
                        step=1,
                        label="Output Size (classes)",
                    )

                    activation = gr.Dropdown(
                        choices=params["activations"],
                        value="ReLU",
                        label="Activation Function",
                    )

                    if params["show_dropout"]:
                        use_dropout = gr.Checkbox(label="Use Dropout", value=False)
                        dropout_rate = gr.Slider(
                            minimum=0.0,
                            maximum=0.8,
                            value=0.2,
                            step=0.05,
                            label="Dropout Rate",
                        )
                    else:
                        use_dropout = gr.Checkbox(value=False, visible=False)
                        dropout_rate = gr.Slider(value=0.0, visible=False)

                    if params["show_batchnorm"]:
                        use_batchnorm = gr.Checkbox(label="Use Batch Normalization", value=False)
                    else:
                        use_batchnorm = gr.Checkbox(value=False, visible=False)

                    if params["show_init"]:
                        init_method = gr.Dropdown(
                            choices=["default", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"],
                            value="default",
                            label="Weight Initialization",
                        )
                    else:
                        init_method = gr.Dropdown(choices=["default"], value="default", visible=False)

                    device_dropdown = gr.Dropdown(
                        choices=["auto", "cpu", "cuda", "mps"],
                        value="auto",
                        label="Device",
                    )

                    seed_input = gr.Number(label="Seed", value=42, precision=0)

                    run_btn = gr.Button("Build Model", variant="primary")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Output"):
                            logs_output = gr.Textbox(label="Logs", lines=25, interactive=False)
                            metrics_output = gr.JSON(label="Metrics")

                        with gr.Tab("Visualization"):
                            plot_output = gr.Plot(label="Architecture Visualization")

                        with gr.Tab("Code"):
                            code_output = gr.Code(language="python", lines=45)

            run_btn.click(
                fn=self.run,
                inputs=[
                    input_size,
                    hidden_sizes,
                    output_size,
                    activation,
                    use_dropout,
                    dropout_rate,
                    use_batchnorm,
                    init_method,
                    device_dropdown,
                    seed_input,
                ],
                outputs=[logs_output, metrics_output, plot_output, code_output],
            )

        return demo


def create_model_builder_ui(level: Level = Level.BEGINNER) -> gr.Blocks:
    """Create the Model Builder demo UI."""
    demo = ModelBuilderDemo()
    return demo.create_ui(level)
