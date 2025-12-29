"""
Logging utilities for PyTorch Playground demos.

Provides structured logging for demo outputs and results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import io


@dataclass
class LogEntry:
    """A single log entry."""

    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class DemoResult:
    """Result from a demo run."""

    success: bool
    logs: str
    metrics: Dict[str, Any]
    code: str
    figure: Optional[Any] = None  # matplotlib figure
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


class DemoLogger:
    """
    Logger for demo execution.

    Collects logs and provides formatted output for Gradio display.
    """

    def __init__(self, name: str = "Demo"):
        self.name = name
        self.entries: List[LogEntry] = []
        self.metrics: Dict[str, Any] = {}
        self._start_time: Optional[datetime] = None

    def start(self) -> None:
        """Mark start of demo execution."""
        self._start_time = datetime.now()
        self.info(f"Starting {self.name}...")

    def end(self, success: bool = True) -> None:
        """Mark end of demo execution."""
        elapsed = self.elapsed_seconds
        status = "completed" if success else "failed"
        self.info(f"{self.name} {status} in {elapsed:.2f}s")

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def _log(self, level: str, message: str, data: Optional[Dict] = None) -> None:
        """Add a log entry."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            data=data,
        )
        self.entries.append(entry)

    def info(self, message: str, data: Optional[Dict] = None) -> None:
        """Log info message."""
        self._log("INFO", message, data)

    def warning(self, message: str, data: Optional[Dict] = None) -> None:
        """Log warning message."""
        self._log("WARNING", message, data)

    def error(self, message: str, data: Optional[Dict] = None) -> None:
        """Log error message."""
        self._log("ERROR", message, data)

    def debug(self, message: str, data: Optional[Dict] = None) -> None:
        """Log debug message."""
        self._log("DEBUG", message, data)

    def metric(self, name: str, value: Any) -> None:
        """Record a metric."""
        self.metrics[name] = value
        self.info(f"{name}: {value}")

    def progress(self, current: int, total: int, prefix: str = "Progress") -> None:
        """Log progress."""
        pct = (current / total) * 100 if total > 0 else 0
        self.info(f"{prefix}: {current}/{total} ({pct:.1f}%)")

    def format_logs(self, include_debug: bool = False) -> str:
        """Format logs for display."""
        lines = []

        for entry in self.entries:
            if entry.level == "DEBUG" and not include_debug:
                continue

            time_str = entry.timestamp.strftime("%H:%M:%S")
            level_prefix = {
                "INFO": "[INFO]",
                "WARNING": "[WARN]",
                "ERROR": "[ERROR]",
                "DEBUG": "[DEBUG]",
            }.get(entry.level, f"[{entry.level}]")

            line = f"{time_str} {level_prefix} {entry.message}"
            lines.append(line)

            if entry.data:
                for key, value in entry.data.items():
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def get_result(
        self,
        success: bool = True,
        code: str = "",
        figure: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> DemoResult:
        """Get a DemoResult from current state."""
        return DemoResult(
            success=success,
            logs=self.format_logs(),
            metrics=self.metrics.copy(),
            code=code,
            figure=figure,
            error=error,
            elapsed_seconds=self.elapsed_seconds,
        )

    def clear(self) -> None:
        """Clear all logs and metrics."""
        self.entries.clear()
        self.metrics.clear()
        self._start_time = None


def format_tensor_info(tensor: "torch.Tensor", name: str = "tensor") -> str:
    """
    Format tensor information for display.

    Args:
        tensor: PyTorch tensor
        name: Name to display for the tensor

    Returns:
        Formatted string with tensor info
    """
    lines = [
        f"{name}:",
        f"  shape: {tuple(tensor.shape)}",
        f"  dtype: {tensor.dtype}",
        f"  device: {tensor.device}",
        f"  requires_grad: {tensor.requires_grad}",
    ]

    # Add stats for numeric tensors
    if tensor.is_floating_point() or tensor.dtype in [
        "torch.int32",
        "torch.int64",
        "torch.int16",
        "torch.int8",
    ]:
        try:
            lines.append(f"  min: {tensor.min().item():.4f}")
            lines.append(f"  max: {tensor.max().item():.4f}")
            lines.append(f"  mean: {tensor.float().mean().item():.4f}")
            lines.append(f"  std: {tensor.float().std().item():.4f}")
        except Exception:
            pass

    # Show memory
    try:
        nbytes = tensor.nelement() * tensor.element_size()
        if nbytes < 1024:
            size_str = f"{nbytes} bytes"
        elif nbytes < 1024**2:
            size_str = f"{nbytes / 1024:.2f} KB"
        else:
            size_str = f"{nbytes / 1024**2:.2f} MB"
        lines.append(f"  memory: {size_str}")
    except Exception:
        pass

    return "\n".join(lines)


def format_model_info(model: "torch.nn.Module", name: str = "model") -> str:
    """
    Format model information for display.

    Args:
        model: PyTorch model
        name: Name to display

    Returns:
        Formatted string with model info
    """
    import torch

    lines = [f"{name}:"]

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines.append(f"  total parameters: {total_params:,}")
    lines.append(f"  trainable parameters: {trainable_params:,}")

    # Get device
    try:
        param = next(model.parameters())
        lines.append(f"  device: {param.device}")
        lines.append(f"  dtype: {param.dtype}")
    except StopIteration:
        lines.append("  (no parameters)")

    return "\n".join(lines)


def format_training_step(
    epoch: int,
    total_epochs: int,
    loss: float,
    metrics: Optional[Dict[str, float]] = None,
    lr: Optional[float] = None,
) -> str:
    """
    Format a training step for display.

    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        loss: Current loss
        metrics: Optional additional metrics
        lr: Optional current learning rate

    Returns:
        Formatted string
    """
    parts = [f"Epoch {epoch}/{total_epochs}", f"Loss: {loss:.4f}"]

    if metrics:
        for name, value in metrics.items():
            parts.append(f"{name}: {value:.4f}")

    if lr is not None:
        parts.append(f"LR: {lr:.2e}")

    return " | ".join(parts)


class CodeBuilder:
    """Helper to build code strings for display."""

    def __init__(self):
        self.lines: List[str] = []
        self.indent_level = 0

    def add(self, line: str) -> "CodeBuilder":
        """Add a line of code."""
        indent = "    " * self.indent_level
        self.lines.append(f"{indent}{line}")
        return self

    def add_blank(self) -> "CodeBuilder":
        """Add a blank line."""
        self.lines.append("")
        return self

    def indent(self) -> "CodeBuilder":
        """Increase indent level."""
        self.indent_level += 1
        return self

    def dedent(self) -> "CodeBuilder":
        """Decrease indent level."""
        self.indent_level = max(0, self.indent_level - 1)
        return self

    def add_comment(self, comment: str) -> "CodeBuilder":
        """Add a comment line."""
        return self.add(f"# {comment}")

    def build(self) -> str:
        """Build the final code string."""
        return "\n".join(self.lines)
