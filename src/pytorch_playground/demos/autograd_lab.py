"""
Autograd Lab Demo - Interactive exploration of automatic differentiation.

Covers:
- Computing gradients with backward()
- Understanding requires_grad
- Gradient accumulation
- Detaching tensors
"""

import gradio as gr
from typing import Tuple, Optional, Any, Dict

from pytorch_playground.state import Level
from pytorch_playground.devices import get_device
from pytorch_playground.utils.seeding import set_seed
from pytorch_playground.utils.logging import DemoLogger, CodeBuilder
from pytorch_playground.utils.plotting import create_tensor_visualization


class AutogradLabDemo:
    """Interactive autograd exploration demo."""

    def __init__(self):
        self.name = "Autograd Lab"
        self.description = "Explore automatic differentiation in PyTorch"

    def get_params_for_level(self, level: Level) -> Dict[str, Any]:
        """Get parameter configuration based on level."""
        if level == Level.BEGINNER:
            return {
                "functions": ["x^2", "x^3", "sin(x)", "exp(x)"],
                "show_retain_graph": False,
                "show_create_graph": False,
            }
        elif level == Level.INTERMEDIATE:
            return {
                "functions": ["x^2", "x^3", "sin(x)", "exp(x)", "x*y", "sum(x^2)"],
                "show_retain_graph": True,
                "show_create_graph": False,
            }
        else:  # Advanced
            return {
                "functions": ["x^2", "x^3", "sin(x)", "exp(x)", "x*y", "sum(x^2)", "neural_net"],
                "show_retain_graph": True,
                "show_create_graph": True,
            }

    def run(
        self,
        function_choice: str,
        x_value: float,
        y_value: float,
        retain_graph: bool,
        create_graph: bool,
        device_selection: str,
        seed: Optional[int],
    ) -> Tuple[str, Dict[str, Any], Any, str]:
        """Run the autograd demo."""
        import torch

        logger = DemoLogger(self.name)
        logger.start()

        try:
            # Set seed
            if seed is not None:
                set_seed(seed)
                logger.info(f"Seed set to {seed}")

            # Get device
            device, device_label, warning = get_device(device_selection)
            if warning:
                logger.warning(warning)
            logger.info(f"Using device: {device_label}")

            # Create input tensor(s)
            x = torch.tensor([x_value], dtype=torch.float32, device=device, requires_grad=True)
            logger.info(f"Created x = {x.item():.4f} with requires_grad=True")

            y = None
            if function_choice == "x*y":
                y = torch.tensor([y_value], dtype=torch.float32, device=device, requires_grad=True)
                logger.info(f"Created y = {y.item():.4f} with requires_grad=True")

            # Compute function
            logger.info(f"Computing f(x) = {function_choice}")

            if function_choice == "x^2":
                result = x ** 2
                analytical_grad = 2 * x_value
            elif function_choice == "x^3":
                result = x ** 3
                analytical_grad = 3 * x_value ** 2
            elif function_choice == "sin(x)":
                result = torch.sin(x)
                import math
                analytical_grad = math.cos(x_value)
            elif function_choice == "exp(x)":
                result = torch.exp(x)
                import math
                analytical_grad = math.exp(x_value)
            elif function_choice == "x*y":
                result = x * y
                analytical_grad = y_value  # df/dx = y
            elif function_choice == "sum(x^2)":
                # Create a small vector
                x_vec = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
                result = (x_vec ** 2).sum()
                logger.info(f"x_vec = {x_vec.tolist()}")
                analytical_grad = None  # Multiple gradients
            elif function_choice == "neural_net":
                # Simple neural net
                torch.manual_seed(seed if seed else 42)
                layer = torch.nn.Linear(1, 1, device=device)
                result = layer(x.unsqueeze(0)).squeeze()
                analytical_grad = None
            else:
                result = x ** 2
                analytical_grad = 2 * x_value

            logger.info(f"Result: {result.item():.6f}")

            # Compute gradients
            logger.info("Computing gradients with backward()...")

            if function_choice == "sum(x^2)":
                result.backward(retain_graph=retain_graph, create_graph=create_graph)
                grad = x_vec.grad
                logger.info(f"Gradients: {grad.tolist()}")
                logger.info("Expected: [2.0, 4.0, 6.0] (2*x for each element)")
            else:
                result.backward(retain_graph=retain_graph, create_graph=create_graph)
                logger.info(f"x.grad = {x.grad.item():.6f}")

                if y is not None:
                    logger.info(f"y.grad = {y.grad.item():.6f}")

                if analytical_grad is not None:
                    logger.info(f"Analytical gradient: {analytical_grad:.6f}")
                    error = abs(x.grad.item() - analytical_grad)
                    logger.info(f"Numerical error: {error:.2e}")

            # Additional demonstrations
            logger.info("")
            logger.info("--- Additional Concepts ---")

            # Show detach
            logger.info("detach(): Creates a tensor without gradient tracking")
            x_detached = x.detach()
            logger.info(f"x.detach().requires_grad = {x_detached.requires_grad}")

            # Show no_grad
            logger.info("")
            logger.info("torch.no_grad(): Context for inference without gradients")

            # Record metrics
            logger.metric("function", function_choice)
            logger.metric("x_value", x_value)
            logger.metric("result", result.item())
            if x.grad is not None:
                logger.metric("x_grad", x.grad.item())
            if analytical_grad is not None:
                logger.metric("analytical_grad", analytical_grad)

            # Create visualization
            fig = self._create_gradient_plot(function_choice, x_value, device)

            # Build code
            code = self._build_code(
                function_choice, x_value, y_value, retain_graph, create_graph, device_selection, seed
            )

            logger.end(success=True)
            return logger.format_logs(), logger.metrics, fig, code

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            logger.end(success=False)
            return logger.format_logs(), {}, None, ""

    def _create_gradient_plot(self, function: str, x_val: float, device) -> Any:
        """Create a plot showing the function and gradient."""
        import torch
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Create x range
        x_range = np.linspace(x_val - 3, x_val + 3, 100)

        # Compute function values
        if function == "x^2":
            y_vals = x_range ** 2
            grad_vals = 2 * x_range
            func_label = "$f(x) = x^2$"
            grad_label = "$f'(x) = 2x$"
        elif function == "x^3":
            y_vals = x_range ** 3
            grad_vals = 3 * x_range ** 2
            func_label = "$f(x) = x^3$"
            grad_label = "$f'(x) = 3x^2$"
        elif function == "sin(x)":
            y_vals = np.sin(x_range)
            grad_vals = np.cos(x_range)
            func_label = "$f(x) = sin(x)$"
            grad_label = "$f'(x) = cos(x)$"
        elif function == "exp(x)":
            y_vals = np.exp(np.clip(x_range, -10, 10))
            grad_vals = y_vals
            func_label = "$f(x) = e^x$"
            grad_label = "$f'(x) = e^x$"
        else:
            y_vals = x_range ** 2
            grad_vals = 2 * x_range
            func_label = "$f(x) = x^2$"
            grad_label = "$f'(x) = 2x$"

        # Plot function
        ax1.plot(x_range, y_vals, 'b-', linewidth=2, label=func_label)
        ax1.axvline(x=x_val, color='r', linestyle='--', alpha=0.7, label=f'x = {x_val}')

        # Mark the point
        if function == "sin(x)":
            y_point = np.sin(x_val)
        elif function == "exp(x)":
            y_point = np.exp(x_val)
        elif function == "x^3":
            y_point = x_val ** 3
        else:
            y_point = x_val ** 2

        ax1.scatter([x_val], [y_point], color='red', s=100, zorder=5)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot gradient
        ax2.plot(x_range, grad_vals, 'g-', linewidth=2, label=grad_label)
        ax2.axvline(x=x_val, color='r', linestyle='--', alpha=0.7)

        # Mark gradient at point
        if function == "sin(x)":
            grad_point = np.cos(x_val)
        elif function == "exp(x)":
            grad_point = np.exp(x_val)
        elif function == "x^3":
            grad_point = 3 * x_val ** 2
        else:
            grad_point = 2 * x_val

        ax2.scatter([x_val], [grad_point], color='red', s=100, zorder=5)
        ax2.set_xlabel('x')
        ax2.set_ylabel("f'(x)")
        ax2.set_title(f"Gradient (at x={x_val}: {grad_point:.4f})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _build_code(
        self,
        function: str,
        x_val: float,
        y_val: float,
        retain_graph: bool,
        create_graph: bool,
        device: str,
        seed: Optional[int],
    ) -> str:
        """Build the code string."""
        builder = CodeBuilder()

        builder.add("import torch")
        builder.add_blank()

        if seed:
            builder.add(f"torch.manual_seed({seed})")
            builder.add_blank()

        builder.add_comment("Create input tensor with gradient tracking")
        builder.add(f"x = torch.tensor([{x_val}], requires_grad=True)")
        builder.add_blank()

        builder.add_comment(f"Compute function: {function}")

        if function == "x^2":
            builder.add("y = x ** 2")
        elif function == "x^3":
            builder.add("y = x ** 3")
        elif function == "sin(x)":
            builder.add("y = torch.sin(x)")
        elif function == "exp(x)":
            builder.add("y = torch.exp(x)")
        elif function == "x*y":
            builder.add(f"y_input = torch.tensor([{y_val}], requires_grad=True)")
            builder.add("y = x * y_input")

        builder.add_blank()
        builder.add(f"print(f'y = {{y.item():.6f}}')")
        builder.add_blank()

        builder.add_comment("Compute gradients via backpropagation")

        backward_args = []
        if retain_graph:
            backward_args.append("retain_graph=True")
        if create_graph:
            backward_args.append("create_graph=True")

        if backward_args:
            builder.add(f"y.backward({', '.join(backward_args)})")
        else:
            builder.add("y.backward()")

        builder.add_blank()
        builder.add_comment("Access the computed gradient")
        builder.add("print(f'dy/dx = {x.grad.item():.6f}')")

        builder.add_blank()
        builder.add_comment("Verify: compare with analytical gradient")

        if function == "x^2":
            builder.add(f"analytical = 2 * {x_val}  # d(x^2)/dx = 2x")
        elif function == "x^3":
            builder.add(f"analytical = 3 * {x_val}**2  # d(x^3)/dx = 3x^2")

        builder.add("print(f'Analytical: {analytical:.6f}')")

        return builder.build()

    def create_ui(self, level: Level) -> gr.Blocks:
        """Create the Gradio UI."""
        params = self.get_params_for_level(level)

        with gr.Blocks() as demo:
            gr.Markdown(f"## {self.name}")
            gr.Markdown(self.description)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Parameters")

                    function_dropdown = gr.Dropdown(
                        choices=params["functions"],
                        value="x^2",
                        label="Function f(x)",
                    )

                    x_slider = gr.Slider(
                        minimum=-5,
                        maximum=5,
                        value=2.0,
                        step=0.1,
                        label="x value",
                    )

                    y_slider = gr.Slider(
                        minimum=-5,
                        maximum=5,
                        value=3.0,
                        step=0.1,
                        label="y value (for x*y only)",
                        visible=True,
                    )

                    device_dropdown = gr.Dropdown(
                        choices=["auto", "cpu", "cuda", "mps"],
                        value="auto",
                        label="Device",
                    )

                    if params.get("show_retain_graph"):
                        retain_graph_check = gr.Checkbox(
                            label="retain_graph",
                            value=False,
                            info="Keep computation graph after backward()",
                        )
                    else:
                        retain_graph_check = gr.Checkbox(value=False, visible=False)

                    if params.get("show_create_graph"):
                        create_graph_check = gr.Checkbox(
                            label="create_graph",
                            value=False,
                            info="Enable higher-order gradients",
                        )
                    else:
                        create_graph_check = gr.Checkbox(value=False, visible=False)

                    with gr.Accordion("Reproducibility", open=False):
                        seed_input = gr.Number(label="Seed", value=42, precision=0)

                    run_btn = gr.Button("Compute Gradients", variant="primary")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Output"):
                            logs_output = gr.Textbox(label="Logs", lines=20, interactive=False)
                            metrics_output = gr.JSON(label="Metrics")

                        with gr.Tab("Visualization"):
                            plot_output = gr.Plot(label="Function and Gradient")

                        with gr.Tab("Code"):
                            code_output = gr.Code(language="python", lines=30)

            run_btn.click(
                fn=self.run,
                inputs=[
                    function_dropdown,
                    x_slider,
                    y_slider,
                    retain_graph_check,
                    create_graph_check,
                    device_dropdown,
                    seed_input,
                ],
                outputs=[logs_output, metrics_output, plot_output, code_output],
            )

        return demo


def create_autograd_lab_ui(level: Level = Level.BEGINNER) -> gr.Blocks:
    """Create the Autograd Lab demo UI."""
    demo = AutogradLabDemo()
    return demo.create_ui(level)
