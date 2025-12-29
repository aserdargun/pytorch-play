"""
Save & Load Demo - Model persistence and checkpointing.

Covers:
- Saving model state_dict
- Loading for inference
- Creating training checkpoints
- Best model tracking
"""

import gradio as gr
from typing import Tuple, Optional, Any, Dict
import tempfile
import os

from pytorch_playground.state import Level
from pytorch_playground.devices import get_device
from pytorch_playground.utils.seeding import set_seed
from pytorch_playground.utils.logging import DemoLogger, CodeBuilder
import matplotlib.pyplot as plt
import numpy as np


class SaveLoadDemo:
    """Interactive save/load demo."""

    def __init__(self):
        self.name = "Save & Load"
        self.description = "Learn to save and load PyTorch models"
        self._temp_dir = tempfile.mkdtemp()
        self._last_saved_model = None

    def get_params_for_level(self, level: Level) -> Dict[str, Any]:
        """Get parameter configuration based on level."""
        if level == Level.BEGINNER:
            return {
                "show_optimizer_state": False,
                "show_checkpoint": False,
            }
        elif level == Level.INTERMEDIATE:
            return {
                "show_optimizer_state": True,
                "show_checkpoint": True,
            }
        else:
            return {
                "show_optimizer_state": True,
                "show_checkpoint": True,
                "show_torchscript": True,
            }

    def create_model(self, hidden_size: int, seed: int, device):
        """Create a simple model for demonstration."""
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)

        model = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )

        return model.to(device)

    def run_save(
        self,
        hidden_size: int,
        save_optimizer: bool,
        save_as_checkpoint: bool,
        save_torchscript: bool,
        device_selection: str,
        seed: int,
    ) -> Tuple[str, Dict[str, Any], str]:
        """Save a model."""
        import torch

        logger = DemoLogger("Save Model")
        logger.start()

        try:
            set_seed(seed)
            device, device_label, warning = get_device(device_selection)
            if warning:
                logger.warning(warning)

            # Create model
            model = self.create_model(hidden_size, seed, device)
            logger.info(f"Created model on {device_label}")

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {total_params:,}")

            # Create optimizer (for checkpoint demo)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Simulate some training
            logger.info("Simulating training (2 steps)...")
            x = torch.randn(4, 10, device=device)
            for _ in range(2):
                optimizer.zero_grad()
                out = model(x)
                loss = out.sum()
                loss.backward()
                optimizer.step()

            # Save paths
            model_path = os.path.join(self._temp_dir, "model.pth")
            checkpoint_path = os.path.join(self._temp_dir, "checkpoint.pth")
            torchscript_path = os.path.join(self._temp_dir, "model_scripted.pt")

            # Save state_dict (most common)
            logger.info("")
            logger.info("--- Saving model state_dict ---")
            torch.save(model.state_dict(), model_path)
            model_size = os.path.getsize(model_path)
            logger.info(f"Saved to: {model_path}")
            logger.info(f"File size: {model_size / 1024:.2f} KB")

            # Store reference
            self._last_saved_model = {
                "hidden_size": hidden_size,
                "state_dict_path": model_path,
                "device": device,
            }

            if save_as_checkpoint:
                logger.info("")
                logger.info("--- Saving full checkpoint ---")
                checkpoint = {
                    "epoch": 10,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict() if save_optimizer else None,
                    "loss": 0.123,
                    "best_accuracy": 0.95,
                }
                torch.save(checkpoint, checkpoint_path)
                checkpoint_size = os.path.getsize(checkpoint_path)
                logger.info(f"Saved to: {checkpoint_path}")
                logger.info(f"File size: {checkpoint_size / 1024:.2f} KB")
                logger.info("Checkpoint contains: epoch, model_state_dict, optimizer_state_dict, loss, best_accuracy")
                self._last_saved_model["checkpoint_path"] = checkpoint_path

            if save_torchscript:
                logger.info("")
                logger.info("--- Saving TorchScript model ---")
                try:
                    model.eval()
                    scripted_model = torch.jit.script(model)
                    scripted_model.save(torchscript_path)
                    ts_size = os.path.getsize(torchscript_path)
                    logger.info(f"Saved to: {torchscript_path}")
                    logger.info(f"File size: {ts_size / 1024:.2f} KB")
                    logger.info("TorchScript model can be loaded without Python!")
                    self._last_saved_model["torchscript_path"] = torchscript_path
                except Exception as e:
                    logger.warning(f"TorchScript save failed: {e}")

            # Metrics
            logger.metric("hidden_size", hidden_size)
            logger.metric("total_params", total_params)
            logger.metric("model_size_kb", model_size / 1024)

            # Build code
            code = self._build_save_code(save_optimizer, save_as_checkpoint, save_torchscript)

            logger.end(success=True)
            return logger.format_logs(), logger.metrics, code

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            logger.end(success=False)
            return logger.format_logs(), {}, ""

    def run_load(
        self,
        device_selection: str,
        load_checkpoint: bool,
    ) -> Tuple[str, Dict[str, Any], Any, str]:
        """Load a saved model."""
        import torch

        logger = DemoLogger("Load Model")
        logger.start()

        try:
            device, device_label, warning = get_device(device_selection)
            if warning:
                logger.warning(warning)

            if self._last_saved_model is None:
                logger.error("No model has been saved yet! Run 'Save Model' first.")
                logger.end(success=False)
                return logger.format_logs(), {}, None, ""

            # Create empty model with same architecture
            hidden_size = self._last_saved_model["hidden_size"]
            model = self.create_model(hidden_size, 0, device)

            if load_checkpoint and "checkpoint_path" in self._last_saved_model:
                logger.info("--- Loading from checkpoint ---")
                checkpoint = torch.load(
                    self._last_saved_model["checkpoint_path"],
                    map_location=device,
                    weights_only=False
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
                logger.info(f"Last loss: {checkpoint['loss']:.4f}")
                logger.info(f"Best accuracy: {checkpoint['best_accuracy']:.4f}")

                if checkpoint.get("optimizer_state_dict"):
                    logger.info("Optimizer state also available in checkpoint")
            else:
                logger.info("--- Loading state_dict ---")
                state_dict = torch.load(
                    self._last_saved_model["state_dict_path"],
                    map_location=device,
                    weights_only=True
                )
                model.load_state_dict(state_dict)
                logger.info("Model weights loaded successfully")

            # Test inference
            logger.info("")
            logger.info("--- Testing loaded model ---")
            model.eval()

            x_test = torch.randn(5, 10, device=device)
            with torch.no_grad():
                output = model(x_test)

            logger.info(f"Input shape: {tuple(x_test.shape)}")
            logger.info(f"Output shape: {tuple(output.shape)}")
            logger.info(f"Output sample: {output[0].tolist()}")

            # Metrics
            logger.metric("hidden_size", hidden_size)
            logger.metric("device", str(device))
            logger.metric("output_shape", list(output.shape))

            # Create comparison plot
            fig = self._create_comparison_plot(x_test, output)

            # Build code
            code = self._build_load_code(load_checkpoint)

            logger.end(success=True)
            return logger.format_logs(), logger.metrics, fig, code

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            logger.end(success=False)
            return logger.format_logs(), {}, None, ""

    def _create_comparison_plot(self, x, output) -> Any:
        """Create a visualization of model input/output."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Input visualization
        x_np = x.cpu().numpy()
        im1 = ax1.imshow(x_np, cmap='RdBu', aspect='auto')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Samples')
        ax1.set_title('Input Data')
        fig.colorbar(im1, ax=ax1)

        # Output visualization
        out_np = output.cpu().numpy()
        for i in range(out_np.shape[0]):
            ax2.bar(np.arange(out_np.shape[1]) + i * 0.15, out_np[i],
                   width=0.15, label=f'Sample {i}', alpha=0.8)
        ax2.set_xlabel('Output Classes')
        ax2.set_ylabel('Logits')
        ax2.set_title('Model Output (Logits)')
        ax2.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        return fig

    def _build_save_code(self, save_optimizer: bool, save_checkpoint: bool, save_torchscript: bool) -> str:
        """Build save code string."""
        builder = CodeBuilder()

        builder.add("import torch")
        builder.add("import torch.nn as nn")
        builder.add_blank()

        builder.add_comment("Define your model")
        builder.add("model = MyModel()")
        builder.add("optimizer = torch.optim.Adam(model.parameters())")
        builder.add_blank()

        builder.add_comment("... train your model ...")
        builder.add_blank()

        builder.add_comment("Save model state_dict (recommended)")
        builder.add("torch.save(model.state_dict(), 'model.pth')")
        builder.add_blank()

        if save_checkpoint:
            builder.add_comment("Save full checkpoint for training resumption")
            builder.add("checkpoint = {")
            builder.indent()
            builder.add("'epoch': current_epoch,")
            builder.add("'model_state_dict': model.state_dict(),")
            if save_optimizer:
                builder.add("'optimizer_state_dict': optimizer.state_dict(),")
            builder.add("'loss': current_loss,")
            builder.add("'best_accuracy': best_acc,")
            builder.dedent()
            builder.add("}")
            builder.add("torch.save(checkpoint, 'checkpoint.pth')")
            builder.add_blank()

        if save_torchscript:
            builder.add_comment("Save as TorchScript for deployment")
            builder.add("model.eval()")
            builder.add("scripted_model = torch.jit.script(model)")
            builder.add("scripted_model.save('model_scripted.pt')")

        return builder.build()

    def _build_load_code(self, load_checkpoint: bool) -> str:
        """Build load code string."""
        builder = CodeBuilder()

        builder.add("import torch")
        builder.add_blank()

        builder.add_comment("Create model with same architecture")
        builder.add("model = MyModel()")
        builder.add_blank()

        if load_checkpoint:
            builder.add_comment("Load from checkpoint")
            builder.add("checkpoint = torch.load('checkpoint.pth', weights_only=False)")
            builder.add("model.load_state_dict(checkpoint['model_state_dict'])")
            builder.add_blank()
            builder.add_comment("Optionally restore optimizer for continued training")
            builder.add("optimizer = torch.optim.Adam(model.parameters())")
            builder.add("optimizer.load_state_dict(checkpoint['optimizer_state_dict'])")
            builder.add("start_epoch = checkpoint['epoch'] + 1")
        else:
            builder.add_comment("Load state_dict")
            builder.add("state_dict = torch.load('model.pth', weights_only=True)")
            builder.add("model.load_state_dict(state_dict)")

        builder.add_blank()
        builder.add_comment("Set to evaluation mode for inference")
        builder.add("model.eval()")
        builder.add_blank()
        builder.add_comment("Run inference")
        builder.add("with torch.no_grad():")
        builder.indent()
        builder.add("output = model(input_data)")

        return builder.build()

    def create_ui(self, level: Level) -> gr.Blocks:
        """Create the Gradio UI."""
        params = self.get_params_for_level(level)

        with gr.Blocks() as demo:
            gr.Markdown(f"## {self.name}")
            gr.Markdown(self.description)

            with gr.Tabs():
                with gr.Tab("Save Model"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Model Configuration")
                            hidden_size = gr.Slider(16, 256, value=64, step=16, label="Hidden Size")

                            gr.Markdown("### Save Options")

                            if params.get("show_optimizer_state"):
                                save_optimizer = gr.Checkbox(label="Include optimizer state", value=False)
                            else:
                                save_optimizer = gr.Checkbox(value=False, visible=False)

                            if params.get("show_checkpoint"):
                                save_checkpoint = gr.Checkbox(label="Save as full checkpoint", value=False)
                            else:
                                save_checkpoint = gr.Checkbox(value=False, visible=False)

                            if params.get("show_torchscript"):
                                save_torchscript = gr.Checkbox(label="Save TorchScript", value=False)
                            else:
                                save_torchscript = gr.Checkbox(value=False, visible=False)

                            device = gr.Dropdown(["auto", "cpu", "cuda", "mps"], value="auto", label="Device")
                            seed = gr.Number(value=42, precision=0, label="Seed")

                            save_btn = gr.Button("Save Model", variant="primary")

                        with gr.Column(scale=2):
                            save_logs = gr.Textbox(label="Logs", lines=20, interactive=False)
                            save_metrics = gr.JSON(label="Metrics")
                            save_code = gr.Code(language="python", label="Code", lines=25)

                    save_btn.click(
                        fn=self.run_save,
                        inputs=[hidden_size, save_optimizer, save_checkpoint, save_torchscript, device, seed],
                        outputs=[save_logs, save_metrics, save_code],
                    )

                with gr.Tab("Load Model"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Load Options")
                            gr.Markdown("*First save a model using the 'Save Model' tab*")

                            if params.get("show_checkpoint"):
                                load_checkpoint = gr.Checkbox(label="Load from checkpoint", value=False)
                            else:
                                load_checkpoint = gr.Checkbox(value=False, visible=False)

                            load_device = gr.Dropdown(["auto", "cpu", "cuda", "mps"], value="auto", label="Device")

                            load_btn = gr.Button("Load Model", variant="primary")

                        with gr.Column(scale=2):
                            with gr.Tabs():
                                with gr.Tab("Output"):
                                    load_logs = gr.Textbox(label="Logs", lines=20, interactive=False)
                                    load_metrics = gr.JSON(label="Metrics")

                                with gr.Tab("Visualization"):
                                    load_plot = gr.Plot(label="Model Output")

                                with gr.Tab("Code"):
                                    load_code = gr.Code(language="python", lines=25)

                    load_btn.click(
                        fn=self.run_load,
                        inputs=[load_device, load_checkpoint],
                        outputs=[load_logs, load_metrics, load_plot, load_code],
                    )

        return demo


def create_save_load_ui(level: Level = Level.BEGINNER) -> gr.Blocks:
    """Create the Save & Load demo UI."""
    demo = SaveLoadDemo()
    return demo.create_ui(level)
