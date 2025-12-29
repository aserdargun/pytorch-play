"""
Tensor Lab Demo - Interactive exploration of PyTorch tensors.

Covers:
- Creating tensors with different shapes and dtypes
- Moving tensors between devices
- Basic operations and indexing
- Understanding requires_grad
"""

import gradio as gr
from typing import Tuple, Optional, Any, Dict
import numpy as np

from pytorch_playground.state import Level, get_state
from pytorch_playground.devices import get_device
from pytorch_playground.utils.seeding import set_seed
from pytorch_playground.utils.logging import DemoLogger, format_tensor_info, CodeBuilder
from pytorch_playground.utils.plotting import create_tensor_visualization


class TensorLabDemo:
    """Interactive tensor exploration demo."""

    def __init__(self):
        self.name = "Tensor Lab"
        self.description = "Explore PyTorch tensors interactively"

    def get_params_for_level(self, level: Level) -> Dict[str, Any]:
        """Get parameter configuration based on level."""
        if level == Level.BEGINNER:
            return {
                "show_shape": True,
                "show_dtype": True,
                "show_device": True,
                "show_requires_grad": True,
                "show_init_method": True,
                "show_advanced": False,
                "max_dims": 3,
                "max_size": 100,
            }
        elif level == Level.INTERMEDIATE:
            return {
                "show_shape": True,
                "show_dtype": True,
                "show_device": True,
                "show_requires_grad": True,
                "show_init_method": True,
                "show_advanced": True,
                "show_memory_format": False,
                "max_dims": 4,
                "max_size": 1000,
            }
        else:  # Advanced
            return {
                "show_shape": True,
                "show_dtype": True,
                "show_device": True,
                "show_requires_grad": True,
                "show_init_method": True,
                "show_advanced": True,
                "show_memory_format": True,
                "show_strides": True,
                "max_dims": 5,
                "max_size": 10000,
            }

    def run(
        self,
        shape_str: str,
        dtype: str,
        device_selection: str,
        requires_grad: bool,
        init_method: str,
        seed: Optional[int],
        deterministic: bool,
    ) -> Tuple[str, Dict[str, Any], Any, str]:
        """
        Run the tensor creation demo.

        Returns:
            Tuple of (logs, metrics, figure, code)
        """
        import torch

        logger = DemoLogger(self.name)
        logger.start()

        try:
            # Set seed
            if seed is not None:
                seed_info = set_seed(seed, deterministic)
                logger.info(f"Seed set to {seed}")

            # Parse shape
            try:
                shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
                if not shape:
                    shape = (3, 4)
            except ValueError:
                logger.warning(f"Invalid shape '{shape_str}', using default (3, 4)")
                shape = (3, 4)

            logger.info(f"Creating tensor with shape {shape}")

            # Get device
            device, device_label, warning = get_device(device_selection)
            if warning:
                logger.warning(warning)
            logger.info(f"Using device: {device_label}")

            # Parse dtype
            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "int32": torch.int32,
                "int64": torch.int64,
                "int16": torch.int16,
                "int8": torch.int8,
                "bool": torch.bool,
            }
            torch_dtype = dtype_map.get(dtype, torch.float32)

            # Create tensor based on init method
            init_methods = {
                "zeros": lambda: torch.zeros(shape, dtype=torch_dtype, device=device),
                "ones": lambda: torch.ones(shape, dtype=torch_dtype, device=device),
                "random": lambda: torch.rand(shape, dtype=torch_dtype if torch_dtype.is_floating_point else torch.float32, device=device),
                "randn": lambda: torch.randn(shape, device=device).to(torch_dtype) if torch_dtype.is_floating_point else torch.randn(shape, device=device).to(torch.float32),
                "arange": lambda: torch.arange(np.prod(shape), dtype=torch_dtype, device=device).reshape(shape),
                "linspace": lambda: torch.linspace(0, 1, np.prod(shape), dtype=torch_dtype if torch_dtype.is_floating_point else torch.float32, device=device).reshape(shape),
            }

            if init_method not in init_methods:
                init_method = "random"

            tensor = init_methods[init_method]()

            # Set requires_grad if applicable
            if requires_grad and tensor.is_floating_point():
                tensor = tensor.requires_grad_(True)
            elif requires_grad:
                logger.warning("requires_grad only works with floating-point tensors")

            # Log tensor info
            logger.info(format_tensor_info(tensor, "Created tensor"))

            # Record metrics
            logger.metric("shape", str(shape))
            logger.metric("dtype", str(tensor.dtype))
            logger.metric("device", str(tensor.device))
            logger.metric("requires_grad", tensor.requires_grad)
            logger.metric("numel", tensor.numel())
            logger.metric("memory_bytes", tensor.nelement() * tensor.element_size())

            # Create visualization
            fig = create_tensor_visualization(tensor, f"Tensor {shape}")

            # Build code string
            code = self._build_code(
                shape, dtype, device_selection, requires_grad, init_method, seed
            )

            logger.end(success=True)

            return (
                logger.format_logs(),
                logger.metrics,
                fig,
                code,
            )

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            logger.end(success=False)
            return logger.format_logs(), {}, None, ""

    def _build_code(
        self,
        shape: tuple,
        dtype: str,
        device: str,
        requires_grad: bool,
        init_method: str,
        seed: Optional[int],
    ) -> str:
        """Build the code string for display."""
        builder = CodeBuilder()

        builder.add("import torch")
        builder.add_blank()

        if seed is not None:
            builder.add_comment("Set seed for reproducibility")
            builder.add(f"torch.manual_seed({seed})")
            builder.add_blank()

        builder.add_comment("Define tensor parameters")
        builder.add(f"shape = {shape}")
        builder.add(f"dtype = torch.{dtype}")

        if device == "auto":
            builder.add_comment("Auto-select device")
            builder.add("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        else:
            builder.add(f"device = torch.device('{device}')")

        builder.add_blank()
        builder.add_comment("Create tensor")

        if init_method == "zeros":
            builder.add("tensor = torch.zeros(shape, dtype=dtype, device=device)")
        elif init_method == "ones":
            builder.add("tensor = torch.ones(shape, dtype=dtype, device=device)")
        elif init_method == "random":
            builder.add("tensor = torch.rand(shape, dtype=dtype, device=device)")
        elif init_method == "randn":
            builder.add("tensor = torch.randn(shape, device=device)")
        elif init_method == "arange":
            builder.add("import numpy as np")
            builder.add("tensor = torch.arange(np.prod(shape), dtype=dtype, device=device).reshape(shape)")
        elif init_method == "linspace":
            builder.add("import numpy as np")
            builder.add("tensor = torch.linspace(0, 1, np.prod(shape), dtype=dtype, device=device).reshape(shape)")

        if requires_grad:
            builder.add_blank()
            builder.add_comment("Enable gradient tracking")
            builder.add("tensor = tensor.requires_grad_(True)")

        builder.add_blank()
        builder.add_comment("Inspect tensor")
        builder.add("print(f'Shape: {tensor.shape}')")
        builder.add("print(f'Dtype: {tensor.dtype}')")
        builder.add("print(f'Device: {tensor.device}')")
        builder.add("print(f'Requires grad: {tensor.requires_grad}')")
        builder.add("print(tensor)")

        return builder.build()

    def create_ui(self, level: Level) -> gr.Blocks:
        """Create the Gradio UI for this demo."""
        params = self.get_params_for_level(level)

        with gr.Blocks() as demo:
            gr.Markdown(f"## {self.name}")
            gr.Markdown(self.description)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Parameters")

                    shape_input = gr.Textbox(
                        label="Shape (comma-separated)",
                        value="3, 4",
                        info="e.g., '3, 4' for 3x4 tensor",
                    )

                    dtype_dropdown = gr.Dropdown(
                        choices=["float32", "float64", "float16", "int32", "int64", "bool"],
                        value="float32",
                        label="Data Type",
                    )

                    device_dropdown = gr.Dropdown(
                        choices=["auto", "cpu", "cuda", "mps"],
                        value="auto",
                        label="Device",
                    )

                    init_dropdown = gr.Dropdown(
                        choices=["random", "randn", "zeros", "ones", "arange", "linspace"],
                        value="random",
                        label="Initialization Method",
                    )

                    requires_grad_check = gr.Checkbox(
                        label="requires_grad",
                        value=False,
                        info="Enable gradient tracking",
                    )

                    with gr.Accordion("Reproducibility", open=False):
                        seed_input = gr.Number(
                            label="Random Seed",
                            value=42,
                            precision=0,
                        )
                        deterministic_check = gr.Checkbox(
                            label="Deterministic Mode",
                            value=False,
                        )

                    run_btn = gr.Button("Create Tensor", variant="primary")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Output"):
                            logs_output = gr.Textbox(
                                label="Logs",
                                lines=15,
                                interactive=False,
                            )
                            metrics_output = gr.JSON(label="Metrics")

                        with gr.Tab("Visualization"):
                            plot_output = gr.Plot(label="Tensor Visualization")

                        with gr.Tab("Code"):
                            code_output = gr.Code(
                                label="Generated Code",
                                language="python",
                                lines=25,
                            )

            run_btn.click(
                fn=self.run,
                inputs=[
                    shape_input,
                    dtype_dropdown,
                    device_dropdown,
                    requires_grad_check,
                    init_dropdown,
                    seed_input,
                    deterministic_check,
                ],
                outputs=[logs_output, metrics_output, plot_output, code_output],
            )

        return demo


def create_tensor_lab_ui(level: Level = Level.BEGINNER) -> gr.Blocks:
    """Create the Tensor Lab demo UI."""
    demo = TensorLabDemo()
    return demo.create_ui(level)
