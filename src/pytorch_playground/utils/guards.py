"""
Safety guards and runtime limits for PyTorch Playground demos.

Ensures demos run within reasonable time and resource bounds.
"""

import platform
import time
import threading
from typing import Callable, Any, Optional, Tuple
from contextlib import contextmanager
from functools import wraps

from pytorch_playground.state import Level


class TimeGuard:
    """
    Context manager for enforcing time limits on operations.

    Usage:
        with TimeGuard(timeout_seconds=30) as guard:
            # do work
            if guard.should_stop:
                break
    """

    def __init__(self, timeout_seconds: float = 60.0, check_interval: float = 0.1):
        self.timeout = timeout_seconds
        self.check_interval = check_interval
        self.start_time: Optional[float] = None
        self._should_stop = False
        self._timer: Optional[threading.Timer] = None

    def __enter__(self):
        self.start_time = time.time()
        # Set timer to flag stop
        self._timer = threading.Timer(self.timeout, self._set_stop)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer:
            self._timer.cancel()
        return False

    def _set_stop(self):
        self._should_stop = True

    @property
    def should_stop(self) -> bool:
        """Check if time limit has been reached."""
        return self._should_stop

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def remaining(self) -> float:
        """Get remaining time in seconds."""
        return max(0.0, self.timeout - self.elapsed)

    def check(self) -> bool:
        """Check if we should continue. Returns True if OK to continue."""
        return not self._should_stop


def check_dataloader_workers(requested_workers: int) -> Tuple[int, Optional[str]]:
    """
    Check and adjust DataLoader num_workers based on platform.

    Args:
        requested_workers: Number of workers requested

    Returns:
        Tuple of (adjusted_workers, warning_message)
    """
    system = platform.system()
    warning = None

    if system == "Windows":
        if requested_workers > 0:
            warning = (
                "Windows multiprocessing with DataLoader can cause issues. "
                "Setting num_workers=0. For production, consider using if __name__ == '__main__' guard."
            )
            return 0, warning
    elif system == "Darwin":  # macOS
        # macOS works but may need spawn method
        if requested_workers > 4:
            warning = f"Reduced workers from {requested_workers} to 4 on macOS for stability."
            return 4, warning

    # Linux or adjusted value
    return requested_workers, warning


def cap_epochs(requested_epochs: int, level: Level) -> Tuple[int, Optional[str]]:
    """
    Cap training epochs based on level.

    Args:
        requested_epochs: Number of epochs requested
        level: Current user level

    Returns:
        Tuple of (capped_epochs, warning_message)
    """
    max_epochs = level.max_epochs
    warning = None

    if requested_epochs > max_epochs:
        warning = f"Epochs capped from {requested_epochs} to {max_epochs} for {level.value} level."
        return max_epochs, warning

    return requested_epochs, warning


def cap_dataset_size(
    requested_size: int, level: Level, dataset_type: str = "default"
) -> Tuple[int, Optional[str]]:
    """
    Cap dataset size based on level and type.

    Args:
        requested_size: Dataset size requested
        level: Current user level
        dataset_type: Type of dataset (affects limits)

    Returns:
        Tuple of (capped_size, warning_message)
    """
    # Level-based limits
    limits = {
        Level.BEGINNER: {
            "default": 1000,
            "image": 5000,
            "synthetic": 2000,
        },
        Level.INTERMEDIATE: {
            "default": 10000,
            "image": 20000,
            "synthetic": 10000,
        },
        Level.ADVANCED: {
            "default": 60000,
            "image": 60000,
            "synthetic": 50000,
        },
    }

    max_size = limits[level].get(dataset_type, limits[level]["default"])
    warning = None

    if requested_size > max_size:
        warning = f"Dataset size capped from {requested_size} to {max_size} for {level.value} level."
        return max_size, warning

    return requested_size, warning


def safe_run(
    func: Callable,
    timeout: float = 60.0,
    default: Any = None,
) -> Tuple[Any, Optional[str]]:
    """
    Run a function with timeout and error handling.

    Args:
        func: Function to run
        timeout: Timeout in seconds
        default: Default value to return on failure

    Returns:
        Tuple of (result, error_message)
    """
    result = default
    error = None

    # Simple timeout approach using threading
    # Note: This won't actually kill the thread if it hangs
    result_container = [default, None]

    def target():
        try:
            result_container[0] = func()
        except Exception as e:
            result_container[1] = str(e)

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return default, f"Operation timed out after {timeout}s"

    return result_container[0], result_container[1]


def require_torch(func: Callable) -> Callable:
    """Decorator that ensures PyTorch is available."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import torch

            return func(*args, **kwargs)
        except ImportError:
            return None, "PyTorch is not installed. Please install it first."

    return wrapper


def require_cuda(func: Callable) -> Callable:
    """Decorator that ensures CUDA is available."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import torch

            if not torch.cuda.is_available():
                return None, "CUDA is not available on this system."
            return func(*args, **kwargs)
        except ImportError:
            return None, "PyTorch is not installed."

    return wrapper


def get_safe_batch_size(
    requested: int, dataset_size: int, min_batches: int = 2
) -> Tuple[int, Optional[str]]:
    """
    Ensure batch size doesn't exceed dataset size limits.

    Args:
        requested: Requested batch size
        dataset_size: Size of the dataset
        min_batches: Minimum number of batches required

    Returns:
        Tuple of (adjusted_batch_size, warning_message)
    """
    max_batch = dataset_size // min_batches
    warning = None

    if max_batch < 1:
        max_batch = 1

    if requested > max_batch:
        warning = f"Batch size reduced from {requested} to {max_batch} to ensure {min_batches}+ batches."
        return max_batch, warning

    return requested, warning


@contextmanager
def memory_guard(max_gb: float = 8.0):
    """
    Context manager that warns about high memory usage.

    Note: This is advisory only, not a hard limit.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial = torch.cuda.memory_allocated()

        yield

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            peak_gb = peak / (1024**3)

            if peak_gb > max_gb:
                print(
                    f"Warning: Peak GPU memory usage ({peak_gb:.2f} GB) exceeded {max_gb} GB limit."
                )
    except ImportError:
        yield


def is_jupyter_environment() -> bool:
    """Check if running in a Jupyter environment."""
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


def get_platform_info() -> dict:
    """Get platform information for debugging."""
    import sys

    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "python_version": sys.version,
        "machine": platform.machine(),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
        if hasattr(torch.backends, "mps"):
            info["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        info["torch_available"] = False

    return info
