"""
PyTorch Playground: An educational web app for learning PyTorch concepts.

This package provides:
- Level-based learning (Beginner/Intermediate/Advanced)
- Device detection and management (CPU/CUDA/MPS/ROCm)
- Interactive demos with parameterized controls
- Install wizard mirroring PyTorch's official selector
"""

__version__ = "0.1.0"
__author__ = "PyTorch Playground Team"

from pytorch_playground.state import AppState, Level
from pytorch_playground.devices import get_device, detect_devices, DeviceInfo

__all__ = [
    "AppState",
    "Level",
    "get_device",
    "detect_devices",
    "DeviceInfo",
    "__version__",
]
