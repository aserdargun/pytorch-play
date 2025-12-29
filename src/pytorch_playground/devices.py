"""
Device detection and management for PyTorch Playground.

Handles:
- CPU/CUDA/MPS/ROCm detection
- Device selection with fallback
- Device information display
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import platform


@dataclass
class DeviceInfo:
    """Information about a compute device."""

    name: str
    available: bool
    device_type: str  # cpu, cuda, mps, rocm
    device_count: int = 1
    device_name: str = ""
    memory_total: Optional[int] = None  # bytes
    compute_capability: Optional[str] = None
    notes: str = ""

    @property
    def badge(self) -> str:
        """Get a badge string for display."""
        status = "Available" if self.available else "Not available"
        if self.device_name:
            return f"{self.name} ({self.device_name}) - {status}"
        return f"{self.name} - {status}"


def _check_torch_available() -> bool:
    """Check if PyTorch is installed."""
    try:
        import torch

        return True
    except ImportError:
        return False


def detect_cpu() -> DeviceInfo:
    """Detect CPU information."""
    cpu_name = platform.processor() or platform.machine()
    return DeviceInfo(
        name="CPU",
        available=True,
        device_type="cpu",
        device_name=cpu_name,
        notes="Always available as fallback",
    )


def detect_cuda() -> DeviceInfo:
    """Detect NVIDIA CUDA availability and info."""
    if not _check_torch_available():
        return DeviceInfo(
            name="CUDA",
            available=False,
            device_type="cuda",
            notes="PyTorch not installed",
        )

    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else ""
            memory_total = None
            compute_cap = None

            try:
                props = torch.cuda.get_device_properties(0)
                memory_total = props.total_memory
                compute_cap = f"{props.major}.{props.minor}"
            except Exception:
                pass

            # Check if this is actually ROCm (HIP)
            is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

            return DeviceInfo(
                name="CUDA" if not is_rocm else "ROCm (HIP)",
                available=True,
                device_type="rocm" if is_rocm else "cuda",
                device_count=device_count,
                device_name=device_name,
                memory_total=memory_total,
                compute_capability=compute_cap,
                notes="ROCm build detected" if is_rocm else "",
            )
        else:
            return DeviceInfo(
                name="CUDA",
                available=False,
                device_type="cuda",
                notes="No CUDA devices found or CUDA not built",
            )
    except Exception as e:
        return DeviceInfo(
            name="CUDA",
            available=False,
            device_type="cuda",
            notes=f"Detection error: {str(e)}",
        )


def detect_mps() -> DeviceInfo:
    """Detect Apple Silicon MPS availability."""
    if not _check_torch_available():
        return DeviceInfo(
            name="MPS",
            available=False,
            device_type="mps",
            notes="PyTorch not installed",
        )

    try:
        import torch

        is_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        is_available = is_built and torch.backends.mps.is_available()

        if is_available:
            return DeviceInfo(
                name="MPS",
                available=True,
                device_type="mps",
                device_name="Apple Silicon GPU",
                notes="Metal Performance Shaders backend",
            )
        elif is_built:
            return DeviceInfo(
                name="MPS",
                available=False,
                device_type="mps",
                notes="MPS built but not available (check macOS version)",
            )
        else:
            return DeviceInfo(
                name="MPS",
                available=False,
                device_type="mps",
                notes="MPS not built in this PyTorch installation",
            )
    except Exception as e:
        return DeviceInfo(
            name="MPS",
            available=False,
            device_type="mps",
            notes=f"Detection error: {str(e)}",
        )


def detect_devices() -> List[DeviceInfo]:
    """Detect all available compute devices."""
    devices = [detect_cpu()]

    cuda_info = detect_cuda()
    devices.append(cuda_info)

    mps_info = detect_mps()
    devices.append(mps_info)

    return devices


def get_available_devices() -> List[str]:
    """Get list of available device type strings."""
    available = ["cpu"]  # CPU always available

    if _check_torch_available():
        import torch

        if torch.cuda.is_available():
            available.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available.append("mps")

    return available


def get_device(
    selection: str = "auto", fallback_warning: bool = True
) -> Tuple["torch.device", str, Optional[str]]:
    """
    Get a torch.device based on selection with fallback.

    Args:
        selection: "auto", "cpu", "cuda", or "mps"
        fallback_warning: Whether to generate a warning message on fallback

    Returns:
        Tuple of (device, label, warning_message)
    """
    import torch

    warning = None
    available = get_available_devices()

    if selection == "auto":
        # Priority: CUDA > MPS > CPU
        if "cuda" in available:
            return torch.device("cuda"), "CUDA (Auto)", None
        elif "mps" in available:
            return torch.device("mps"), "MPS (Auto)", None
        else:
            return torch.device("cpu"), "CPU (Auto)", None

    elif selection == "cuda":
        if "cuda" in available:
            return torch.device("cuda"), "CUDA", None
        else:
            if fallback_warning:
                warning = "CUDA not available, falling back to CPU"
            return torch.device("cpu"), "CPU (Fallback)", warning

    elif selection == "mps":
        if "mps" in available:
            return torch.device("mps"), "MPS", None
        else:
            if fallback_warning:
                warning = "MPS not available, falling back to CPU"
            return torch.device("cpu"), "CPU (Fallback)", warning

    else:  # cpu or anything else
        return torch.device("cpu"), "CPU", None


def get_device_summary() -> str:
    """Get a summary string of all detected devices."""
    devices = detect_devices()
    lines = []
    for device in devices:
        status = "[OK]" if device.available else "[X]"
        line = f"  {status} {device.badge}"
        if device.notes:
            line += f"\n      {device.notes}"
        lines.append(line)
    return "\n".join(lines)


def format_memory(bytes_val: Optional[int]) -> str:
    """Format memory size in human-readable format."""
    if bytes_val is None:
        return "Unknown"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def get_torch_info() -> dict:
    """Get PyTorch installation information."""
    if not _check_torch_available():
        return {
            "installed": False,
            "version": None,
            "cuda_version": None,
            "hip_version": None,
        }

    import torch

    info = {
        "installed": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "hip_version": torch.version.hip if hasattr(torch.version, "hip") else None,
    }

    # Try to get torchvision/torchaudio versions
    try:
        import torchvision

        info["torchvision_version"] = torchvision.__version__
    except ImportError:
        info["torchvision_version"] = None

    try:
        import torchaudio

        info["torchaudio_version"] = torchaudio.__version__
    except ImportError:
        info["torchaudio_version"] = None

    return info


def get_environment_info() -> dict:
    """Get complete environment information."""
    import sys

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    info.update(get_torch_info())

    # Add device info
    info["devices"] = [
        {
            "name": d.name,
            "available": d.available,
            "type": d.device_type,
            "device_name": d.device_name,
        }
        for d in detect_devices()
    ]

    return info
