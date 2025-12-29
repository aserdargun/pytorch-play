"""
Plotting utilities for PyTorch Playground demos.

Provides consistent, styled plots for training metrics, tensors, etc.
"""

from typing import List, Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend for Gradio
matplotlib.use("Agg")

# Consistent style
COLORS = {
    "primary": "#ee4c2c",  # PyTorch orange
    "secondary": "#29b6f6",  # Light blue
    "tertiary": "#66bb6a",  # Green
    "quaternary": "#ab47bc",  # Purple
    "loss": "#ef5350",  # Red
    "accuracy": "#66bb6a",  # Green
    "lr": "#ff9800",  # Orange
}


def setup_plot_style():
    """Apply consistent plot styling."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "#333333",
            "text.color": "#333333",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "grid.color": "#eeeeee",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def create_loss_plot(
    losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Loss",
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Create a training loss plot.

    Args:
        losses: List of training loss values
        val_losses: Optional list of validation loss values
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(losses) + 1)

    ax.plot(epochs, losses, color=COLORS["loss"], linewidth=2, label="Training Loss")

    if val_losses:
        ax.plot(
            epochs,
            val_losses,
            color=COLORS["secondary"],
            linewidth=2,
            linestyle="--",
            label="Validation Loss",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    # Set y-axis to start at 0 if all losses are positive
    if min(losses) >= 0:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def create_metrics_plot(
    metrics: Dict[str, List[float]],
    title: str = "Training Metrics",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Create a multi-metric plot.

    Args:
        metrics: Dictionary of metric name -> values
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    if len(metrics) == 1:
        axes = [axes]

    colors = list(COLORS.values())

    for idx, (name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        epochs = range(1, len(values) + 1)

        ax.plot(epochs, values, color=colors[idx % len(colors)], linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name.capitalize())
        ax.set_title(name.capitalize())

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def create_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Create a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array
        class_names: Optional list of class names
        title: Plot title
        figsize: Figure size
        cmap: Colormap name

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    return fig


def create_tensor_visualization(
    tensor: "torch.Tensor",
    title: str = "Tensor Visualization",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Create a visualization of a tensor.

    Args:
        tensor: PyTorch tensor to visualize
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    import torch

    setup_plot_style()

    # Convert to numpy for plotting
    data = tensor.detach().cpu().numpy()

    if data.ndim == 1:
        # 1D: Bar chart
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(data)), data, color=COLORS["primary"])
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title(f"{title} (1D Tensor)")

    elif data.ndim == 2:
        # 2D: Heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data, cmap="RdYlBu_r", aspect="auto")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(f"{title} (2D Tensor: {data.shape})")

    elif data.ndim == 3:
        # 3D: Show as image if channel-like, else show slices
        if data.shape[0] in [1, 3, 4]:  # Likely CHW format
            if data.shape[0] == 1:
                data = data[0]  # Squeeze channel dim
                fig, ax = plt.subplots(figsize=figsize)
                im = ax.imshow(data, cmap="gray")
                ax.set_title(f"{title} (Grayscale Image)")
            else:
                # RGB(A) image
                if data.shape[0] == 3:
                    data = np.transpose(data, (1, 2, 0))
                else:
                    data = np.transpose(data, (1, 2, 0))[:, :, :3]
                # Normalize to 0-1
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                fig, ax = plt.subplots(figsize=figsize)
                ax.imshow(data)
                ax.set_title(f"{title} (RGB Image)")
        else:
            # Show first slice
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(data[0], cmap="viridis", aspect="auto")
            ax.figure.colorbar(im, ax=ax)
            ax.set_title(f"{title} (3D Tensor slice 0 of {data.shape[0]})")

    elif data.ndim == 4:
        # 4D: Show grid of first batch items
        n_show = min(4, data.shape[0])
        fig, axes = plt.subplots(1, n_show, figsize=(figsize[0], figsize[1] // 2))
        if n_show == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if data.shape[1] in [1, 3]:
                img = data[i]
                if img.shape[0] == 1:
                    img = img[0]
                    ax.imshow(img, cmap="gray")
                else:
                    img = np.transpose(img, (1, 2, 0))
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    ax.imshow(img)
            else:
                ax.imshow(data[i, 0], cmap="viridis")
            ax.set_title(f"Sample {i}")
            ax.axis("off")

        fig.suptitle(f"{title} (4D Tensor: {data.shape})")

    else:
        # Higher dimensions: just show stats
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            f"Tensor shape: {data.shape}\n"
            f"Min: {data.min():.4f}\n"
            f"Max: {data.max():.4f}\n"
            f"Mean: {data.mean():.4f}\n"
            f"Std: {data.std():.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="center",
        )
        ax.set_title(f"{title} (High-dimensional Tensor)")
        ax.axis("off")

    plt.tight_layout()
    return fig


def create_lr_schedule_plot(
    lrs: List[float],
    title: str = "Learning Rate Schedule",
    figsize: Tuple[int, int] = (8, 4),
) -> plt.Figure:
    """
    Plot learning rate schedule.

    Args:
        lrs: List of learning rate values
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(range(len(lrs)), lrs, color=COLORS["lr"], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title)
    ax.set_yscale("log")

    plt.tight_layout()
    return fig


def create_timing_comparison_plot(
    results: Dict[str, float],
    title: str = "Timing Comparison",
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Create a bar chart comparing timing results.

    Args:
        results: Dictionary of method name -> time in ms
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    names = list(results.keys())
    times = list(results.values())
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"]][: len(names)]

    bars = ax.bar(names, times, color=colors)

    ax.set_ylabel("Time (ms)")
    ax.set_title(title)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.annotate(
            f"{time:.2f}ms",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    return fig
