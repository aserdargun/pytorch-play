"""
Random seed management for reproducibility.

Provides utilities to set seeds consistently across all random sources.
"""

import random
from typing import Optional, Dict, Any


def set_seed(seed: Optional[int], deterministic: bool = False) -> Dict[str, Any]:
    """
    Set random seed for reproducibility across all relevant libraries.

    Args:
        seed: Random seed value. If None, seeding is skipped.
        deterministic: If True, enable deterministic algorithms (may impact performance).

    Returns:
        Dictionary with seeding information.
    """
    info = {
        "seed": seed,
        "deterministic": deterministic,
        "libraries_seeded": [],
        "notes": [],
    }

    if seed is None:
        info["notes"].append("No seed set - results will be non-deterministic")
        return info

    # Python random
    random.seed(seed)
    info["libraries_seeded"].append("random")

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)
        info["libraries_seeded"].append("numpy")
    except ImportError:
        info["notes"].append("NumPy not available")

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        info["libraries_seeded"].append("torch")

        # CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            info["libraries_seeded"].append("torch.cuda")

        # MPS - note: MPS seeding support is limited
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have explicit seeding yet, but torch.manual_seed affects it
            info["notes"].append("MPS uses torch.manual_seed (limited determinism)")

        # Deterministic mode
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            info["notes"].append("Deterministic algorithms enabled (may reduce performance)")

            # cuDNN determinism
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                info["notes"].append("cuDNN deterministic mode enabled")

    except ImportError:
        info["notes"].append("PyTorch not available")

    return info


def get_seed_info() -> Dict[str, Any]:
    """
    Get information about current random state.

    Returns:
        Dictionary with current random state information.
    """
    info = {
        "python_random": True,
        "numpy": False,
        "torch": False,
        "deterministic": False,
    }

    try:
        import numpy as np

        info["numpy"] = True
    except ImportError:
        pass

    try:
        import torch

        info["torch"] = True
        info["torch_initial_seed"] = torch.initial_seed()

        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            info["deterministic"] = torch.are_deterministic_algorithms_enabled()

        if hasattr(torch.backends, "cudnn"):
            info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
            info["cudnn_benchmark"] = torch.backends.cudnn.benchmark

    except ImportError:
        pass

    return info


def reset_seeds() -> None:
    """Reset to a default seed state (seed=42)."""
    set_seed(42, deterministic=False)


class SeedContext:
    """Context manager for temporarily setting a seed."""

    def __init__(self, seed: int, deterministic: bool = False):
        self.seed = seed
        self.deterministic = deterministic
        self._prev_state = {}

    def __enter__(self):
        # Save current state
        try:
            import torch

            self._prev_state["torch_seed"] = torch.initial_seed()
            if hasattr(torch, "are_deterministic_algorithms_enabled"):
                self._prev_state["deterministic"] = (
                    torch.are_deterministic_algorithms_enabled()
                )
        except ImportError:
            pass

        # Set new seed
        set_seed(self.seed, self.deterministic)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous state
        try:
            import torch

            if "torch_seed" in self._prev_state:
                torch.manual_seed(self._prev_state["torch_seed"])
            if "deterministic" in self._prev_state:
                torch.use_deterministic_algorithms(
                    self._prev_state["deterministic"], warn_only=True
                )
        except ImportError:
            pass

        return False
