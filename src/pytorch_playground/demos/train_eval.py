"""
Train & Evaluate Demo - Interactive training loop with metrics.

Covers:
- Training loop implementation
- Loss functions and optimizers
- Learning rate schedulers
- Evaluation metrics
"""

import gradio as gr
from typing import Tuple, Optional, Any, Dict, List
import numpy as np
import time

from pytorch_playground.state import Level
from pytorch_playground.devices import get_device
from pytorch_playground.utils.seeding import set_seed
from pytorch_playground.utils.logging import DemoLogger, CodeBuilder, format_training_step
from pytorch_playground.utils.guards import cap_epochs, TimeGuard
from pytorch_playground.utils.plotting import create_loss_plot, create_confusion_matrix_plot
import matplotlib.pyplot as plt


class TrainEvalDemo:
    """Interactive training and evaluation demo."""

    def __init__(self):
        self.name = "Train & Evaluate"
        self.description = "Train models and visualize training progress"

    def get_params_for_level(self, level: Level) -> Dict[str, Any]:
        """Get parameter configuration based on level."""
        if level == Level.BEGINNER:
            return {
                "max_epochs": 10,
                "optimizers": ["SGD", "Adam"],
                "losses": ["CrossEntropyLoss", "MSELoss"],
                "show_scheduler": False,
                "show_gradient_clip": False,
                "show_weight_decay": False,
            }
        elif level == Level.INTERMEDIATE:
            return {
                "max_epochs": 30,
                "optimizers": ["SGD", "Adam", "AdamW"],
                "losses": ["CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss"],
                "show_scheduler": True,
                "schedulers": ["None", "StepLR", "CosineAnnealing"],
                "show_gradient_clip": True,
                "show_weight_decay": True,
            }
        else:
            return {
                "max_epochs": 100,
                "optimizers": ["SGD", "Adam", "AdamW", "RMSprop"],
                "losses": ["CrossEntropyLoss", "MSELoss", "BCEWithLogitsLoss", "NLLLoss"],
                "show_scheduler": True,
                "schedulers": ["None", "StepLR", "CosineAnnealing", "OneCycleLR", "ReduceLROnPlateau"],
                "show_gradient_clip": True,
                "show_weight_decay": True,
                "show_mixed_precision": True,
            }

    def create_synthetic_dataset(self, n_samples: int, n_features: int, n_classes: int, seed: int):
        """Create a synthetic classification dataset."""
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate cluster centers
        centers = np.random.randn(n_classes, n_features) * 2

        # Generate samples
        samples_per_class = n_samples // n_classes
        X = []
        y = []

        for class_idx in range(n_classes):
            class_samples = centers[class_idx] + np.random.randn(samples_per_class, n_features) * 0.5
            X.append(class_samples)
            y.extend([class_idx] * samples_per_class)

        X = np.vstack(X).astype(np.float32)
        y = np.array(y, dtype=np.int64)

        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return torch.from_numpy(X), torch.from_numpy(y)

    def run(
        self,
        n_samples: int,
        n_features: int,
        n_classes: int,
        hidden_sizes_str: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optimizer_name: str,
        loss_name: str,
        scheduler_name: str,
        weight_decay: float,
        gradient_clip: float,
        use_mixed_precision: bool,
        device_selection: str,
        seed: int,
        level_str: str,
    ) -> Tuple[str, Dict[str, Any], Any, str]:
        """Run the training demo."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        logger = DemoLogger(self.name)
        logger.start()

        # Track metrics for plotting
        train_losses = []
        val_losses = []
        accuracies = []

        try:
            # Parse level
            level = Level(level_str) if level_str else Level.BEGINNER

            # Cap epochs
            epochs, epoch_warning = cap_epochs(epochs, level)
            if epoch_warning:
                logger.warning(epoch_warning)

            set_seed(seed)
            logger.info(f"Seed set to {seed}")

            # Get device
            device, device_label, warning = get_device(device_selection)
            if warning:
                logger.warning(warning)
            logger.info(f"Device: {device_label}")

            # Check mixed precision availability
            use_amp = False
            if use_mixed_precision and device.type == "cuda":
                use_amp = True
                logger.info("Mixed precision enabled")
            elif use_mixed_precision:
                logger.warning("Mixed precision requires CUDA, disabled")

            # Create dataset
            logger.info(f"Creating synthetic dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
            X, y = self.create_synthetic_dataset(n_samples, n_features, n_classes, seed)

            # Split into train/val
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Parse hidden sizes
            try:
                hidden_sizes = [int(x.strip()) for x in hidden_sizes_str.split(",") if x.strip()]
                if not hidden_sizes:
                    hidden_sizes = [64, 32]
            except ValueError:
                hidden_sizes = [64, 32]

            # Build model
            layers = []
            layer_sizes = [n_features] + hidden_sizes + [n_classes]

            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                if i < len(layer_sizes) - 2:
                    layers.append(nn.ReLU())

            model = nn.Sequential(*layers).to(device)
            logger.info(f"Model: {layer_sizes}")

            # Loss function
            loss_map = {
                "CrossEntropyLoss": nn.CrossEntropyLoss(),
                "MSELoss": nn.MSELoss(),
                "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
                "NLLLoss": nn.NLLLoss(),
            }
            criterion = loss_map.get(loss_name, nn.CrossEntropyLoss())

            # Optimizer
            opt_kwargs = {"lr": learning_rate}
            if weight_decay > 0:
                opt_kwargs["weight_decay"] = weight_decay

            opt_map = {
                "SGD": lambda: torch.optim.SGD(model.parameters(), **opt_kwargs),
                "Adam": lambda: torch.optim.Adam(model.parameters(), **opt_kwargs),
                "AdamW": lambda: torch.optim.AdamW(model.parameters(), **opt_kwargs),
                "RMSprop": lambda: torch.optim.RMSprop(model.parameters(), **opt_kwargs),
            }
            optimizer = opt_map.get(optimizer_name, opt_map["Adam"])()

            # Scheduler
            scheduler = None
            if scheduler_name and scheduler_name != "None":
                if scheduler_name == "StepLR":
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
                elif scheduler_name == "CosineAnnealing":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                elif scheduler_name == "OneCycleLR":
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimizer, max_lr=learning_rate * 10, epochs=epochs, steps_per_epoch=len(train_loader)
                    )
                elif scheduler_name == "ReduceLROnPlateau":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
                logger.info(f"Scheduler: {scheduler_name}")

            # Mixed precision scaler
            scaler = torch.amp.GradScaler('cuda') if use_amp else None

            # Training loop
            logger.info("")
            logger.info("--- Training ---")

            with TimeGuard(timeout_seconds=120) as guard:
                for epoch in range(1, epochs + 1):
                    if guard.should_stop:
                        logger.warning("Training stopped due to timeout")
                        break

                    model.train()
                    epoch_loss = 0.0
                    n_batches = 0

                    for batch_x, batch_y in train_loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)

                        optimizer.zero_grad()

                        if use_amp:
                            with torch.amp.autocast('cuda'):
                                outputs = model(batch_x)
                                loss = criterion(outputs, batch_y)
                            scaler.scale(loss).backward()

                            if gradient_clip > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)
                            loss.backward()

                            if gradient_clip > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                            optimizer.step()

                        epoch_loss += loss.item()
                        n_batches += 1

                        # Step scheduler if OneCycleLR
                        if scheduler_name == "OneCycleLR" and scheduler:
                            scheduler.step()

                    epoch_loss /= n_batches
                    train_losses.append(epoch_loss)

                    # Validation
                    model.eval()
                    val_loss = 0.0
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            batch_x = batch_x.to(device)
                            batch_y = batch_y.to(device)

                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()

                            _, predicted = outputs.max(1)
                            correct += (predicted == batch_y).sum().item()
                            total += batch_y.size(0)

                    val_loss /= len(val_loader)
                    val_losses.append(val_loss)

                    accuracy = correct / total if total > 0 else 0
                    accuracies.append(accuracy)

                    # Log progress
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(format_training_step(
                        epoch, epochs, epoch_loss,
                        {"val_loss": val_loss, "accuracy": accuracy},
                        current_lr
                    ))

                    # Step scheduler
                    if scheduler and scheduler_name not in ["OneCycleLR"]:
                        if scheduler_name == "ReduceLROnPlateau":
                            scheduler.step(val_loss)
                        else:
                            scheduler.step()

            # Final evaluation
            logger.info("")
            logger.info("--- Final Evaluation ---")

            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.numpy())

            final_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            logger.info(f"Final Validation Accuracy: {final_accuracy:.4f}")

            # Metrics
            logger.metric("final_train_loss", train_losses[-1] if train_losses else 0)
            logger.metric("final_val_loss", val_losses[-1] if val_losses else 0)
            logger.metric("final_accuracy", final_accuracy)
            logger.metric("epochs_completed", len(train_losses))
            logger.metric("total_params", sum(p.numel() for p in model.parameters()))

            # Create plots
            fig = self._create_training_plots(train_losses, val_losses, accuracies, all_labels, all_preds, n_classes)

            # Build code
            code = self._build_code(
                n_features, hidden_sizes, n_classes, epochs, batch_size,
                learning_rate, optimizer_name, loss_name, scheduler_name,
                weight_decay, gradient_clip, use_amp
            )

            logger.end(success=True)
            return logger.format_logs(), logger.metrics, fig, code

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            logger.end(success=False)

            # Create empty plot on error
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return logger.format_logs(), {}, fig, ""

    def _create_training_plots(
        self,
        train_losses: List[float],
        val_losses: List[float],
        accuracies: List[float],
        labels: List[int],
        preds: List[int],
        n_classes: int,
    ) -> Any:
        """Create training visualization plots."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss curves
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r--', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy curve
        axes[1].plot(epochs, accuracies, 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds, labels=range(n_classes))
        im = axes[2].imshow(cm, cmap='Blues')
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('True')
        axes[2].set_title('Confusion Matrix')
        fig.colorbar(im, ax=axes[2])

        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                axes[2].text(j, i, str(cm[i, j]), ha='center', va='center',
                           color='white' if cm[i, j] > cm.max() / 2 else 'black')

        plt.tight_layout()
        return fig

    def _build_code(
        self,
        n_features: int,
        hidden_sizes: List[int],
        n_classes: int,
        epochs: int,
        batch_size: int,
        lr: float,
        optimizer: str,
        loss: str,
        scheduler: str,
        weight_decay: float,
        grad_clip: float,
        use_amp: bool,
    ) -> str:
        """Build the code string."""
        builder = CodeBuilder()

        builder.add("import torch")
        builder.add("import torch.nn as nn")
        builder.add("from torch.utils.data import DataLoader, TensorDataset")
        builder.add_blank()

        builder.add_comment("Model definition")
        builder.add("class MLP(nn.Module):")
        builder.indent()
        builder.add("def __init__(self):")
        builder.indent()
        builder.add("super().__init__()")
        builder.add(f"self.net = nn.Sequential(")
        builder.indent()

        layer_sizes = [n_features] + hidden_sizes + [n_classes]
        for i in range(len(layer_sizes) - 1):
            builder.add(f"nn.Linear({layer_sizes[i]}, {layer_sizes[i + 1]}),")
            if i < len(layer_sizes) - 2:
                builder.add("nn.ReLU(),")

        builder.dedent()
        builder.add(")")
        builder.dedent()
        builder.add_blank()
        builder.add("def forward(self, x):")
        builder.indent()
        builder.add("return self.net(x)")
        builder.dedent()
        builder.dedent()
        builder.add_blank()

        builder.add_comment("Setup")
        builder.add("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
        builder.add("model = MLP().to(device)")
        builder.add(f"criterion = nn.{loss}()")

        opt_line = f"optimizer = torch.optim.{optimizer}(model.parameters(), lr={lr}"
        if weight_decay > 0:
            opt_line += f", weight_decay={weight_decay}"
        opt_line += ")"
        builder.add(opt_line)
        builder.add_blank()

        if scheduler != "None":
            builder.add_comment("Learning rate scheduler")
            if scheduler == "StepLR":
                builder.add("scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)")
            elif scheduler == "CosineAnnealing":
                builder.add(f"scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max={epochs})")
            builder.add_blank()

        if use_amp:
            builder.add("scaler = torch.amp.GradScaler('cuda')")
            builder.add_blank()

        builder.add_comment("Training loop")
        builder.add(f"for epoch in range(1, {epochs} + 1):")
        builder.indent()
        builder.add("model.train()")
        builder.add("for batch_x, batch_y in train_loader:")
        builder.indent()
        builder.add("batch_x, batch_y = batch_x.to(device), batch_y.to(device)")
        builder.add("optimizer.zero_grad()")
        builder.add_blank()

        if use_amp:
            builder.add("with torch.amp.autocast('cuda'):")
            builder.indent()
            builder.add("outputs = model(batch_x)")
            builder.add("loss = criterion(outputs, batch_y)")
            builder.dedent()
            builder.add("scaler.scale(loss).backward()")
            if grad_clip > 0:
                builder.add("scaler.unscale_(optimizer)")
                builder.add(f"torch.nn.utils.clip_grad_norm_(model.parameters(), {grad_clip})")
            builder.add("scaler.step(optimizer)")
            builder.add("scaler.update()")
        else:
            builder.add("outputs = model(batch_x)")
            builder.add("loss = criterion(outputs, batch_y)")
            builder.add("loss.backward()")
            if grad_clip > 0:
                builder.add(f"torch.nn.utils.clip_grad_norm_(model.parameters(), {grad_clip})")
            builder.add("optimizer.step()")

        builder.dedent()

        if scheduler != "None":
            builder.add("scheduler.step()")

        builder.add("print(f'Epoch {epoch}: Loss = {loss.item():.4f}')")
        builder.dedent()

        return builder.build()

    def create_ui(self, level: Level) -> gr.Blocks:
        """Create the Gradio UI."""
        params = self.get_params_for_level(level)

        with gr.Blocks() as demo:
            gr.Markdown(f"## {self.name}")
            gr.Markdown(self.description)

            # Hidden level state
            level_state = gr.Textbox(value=level.value, visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset")
                    n_samples = gr.Slider(100, 5000, value=1000, step=100, label="Samples")
                    n_features = gr.Slider(2, 100, value=10, step=1, label="Features")
                    n_classes = gr.Slider(2, 10, value=3, step=1, label="Classes")

                    gr.Markdown("### Model")
                    hidden_sizes = gr.Textbox(label="Hidden Sizes", value="64, 32")

                    gr.Markdown("### Training")
                    epochs = gr.Slider(1, params["max_epochs"], value=min(10, params["max_epochs"]), step=1, label="Epochs")
                    batch_size = gr.Slider(8, 256, value=32, step=8, label="Batch Size")
                    learning_rate = gr.Number(value=0.001, label="Learning Rate")

                    optimizer = gr.Dropdown(params["optimizers"], value="Adam", label="Optimizer")
                    loss_fn = gr.Dropdown(params["losses"], value="CrossEntropyLoss", label="Loss Function")

                    if params.get("show_scheduler"):
                        scheduler = gr.Dropdown(params.get("schedulers", ["None"]), value="None", label="LR Scheduler")
                    else:
                        scheduler = gr.Dropdown(choices=["None"], value="None", visible=False)

                    if params.get("show_weight_decay"):
                        weight_decay = gr.Number(value=0.0, label="Weight Decay")
                    else:
                        weight_decay = gr.Number(value=0.0, visible=False)

                    if params.get("show_gradient_clip"):
                        gradient_clip = gr.Number(value=0.0, label="Gradient Clip (0=disabled)")
                    else:
                        gradient_clip = gr.Number(value=0.0, visible=False)

                    if params.get("show_mixed_precision"):
                        use_amp = gr.Checkbox(label="Mixed Precision (CUDA)", value=False)
                    else:
                        use_amp = gr.Checkbox(value=False, visible=False)

                    device = gr.Dropdown(["auto", "cpu", "cuda", "mps"], value="auto", label="Device")
                    seed = gr.Number(value=42, precision=0, label="Seed")

                    run_btn = gr.Button("Train Model", variant="primary")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Output"):
                            logs = gr.Textbox(label="Training Log", lines=25, interactive=False)
                            metrics = gr.JSON(label="Final Metrics")

                        with gr.Tab("Visualization"):
                            plot = gr.Plot(label="Training Curves")

                        with gr.Tab("Code"):
                            code = gr.Code(language="python", lines=50)

            run_btn.click(
                fn=self.run,
                inputs=[
                    n_samples, n_features, n_classes, hidden_sizes,
                    epochs, batch_size, learning_rate, optimizer, loss_fn,
                    scheduler, weight_decay, gradient_clip, use_amp,
                    device, seed, level_state
                ],
                outputs=[logs, metrics, plot, code],
            )

        return demo


def create_train_eval_ui(level: Level = Level.BEGINNER) -> gr.Blocks:
    """Create the Train & Evaluate demo UI."""
    demo = TrainEvalDemo()
    return demo.create_ui(level)
