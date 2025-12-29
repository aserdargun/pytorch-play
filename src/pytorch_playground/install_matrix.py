"""
PyTorch Install Matrix

This file mirrors the options available on the official PyTorch "Start Locally" selector.
UPDATE THIS FILE when PyTorch releases new versions or changes available options.

Official selector: https://pytorch.org/get-started/locally/

IMPORTANT: Always verify final install commands on the official selector page.
This matrix is provided for convenience and may not reflect the latest changes.

Last verified: Check official site for current versions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# =============================================================================
# EDITABLE CONFIGURATION - Update these when PyTorch releases new versions
# =============================================================================

# Available PyTorch builds
BUILDS = ["Stable", "Nightly (Preview)"]

# Available operating systems
OPERATING_SYSTEMS = ["Linux", "macOS", "Windows"]

# Available package managers (pip is primary, others are link-out)
PACKAGE_MANAGERS = {
    "pip": {"primary": True, "supported": True},
    "conda": {"primary": False, "supported": True, "link_out": True},
    "libtorch": {"primary": False, "supported": True, "link_out": True},
    "source": {"primary": False, "supported": True, "link_out": True},
}

# Available compute platforms by OS
# UPDATE these lists when PyTorch adds/removes CUDA versions
COMPUTE_PLATFORMS = {
    "Linux": [
        {"id": "cpu", "name": "CPU", "available": True},
        {"id": "cuda118", "name": "CUDA 11.8", "available": True},
        {"id": "cuda121", "name": "CUDA 12.1", "available": True},
        {"id": "cuda124", "name": "CUDA 12.4", "available": True},
        {"id": "rocm62", "name": "ROCm 6.2", "available": True},
    ],
    "macOS": [
        {"id": "cpu", "name": "CPU", "available": True},
        # MPS is automatically used on Apple Silicon when available
        {"id": "mps", "name": "MPS (Apple Silicon)", "available": True, "note": "Default on Apple Silicon"},
    ],
    "Windows": [
        {"id": "cpu", "name": "CPU", "available": True},
        {"id": "cuda118", "name": "CUDA 11.8", "available": True},
        {"id": "cuda121", "name": "CUDA 12.1", "available": True},
        {"id": "cuda124", "name": "CUDA 12.4", "available": True},
    ],
}

# PyTorch wheel index URLs
# UPDATE these when PyTorch changes their wheel hosting
INDEX_URLS = {
    "stable": {
        "cpu": "https://download.pytorch.org/whl/cpu",
        "cuda118": "https://download.pytorch.org/whl/cu118",
        "cuda121": "https://download.pytorch.org/whl/cu121",
        "cuda124": "https://download.pytorch.org/whl/cu124",
        "rocm62": "https://download.pytorch.org/whl/rocm6.2",
        "mps": None,  # Default PyPI for macOS
    },
    "nightly": {
        "cpu": "https://download.pytorch.org/whl/nightly/cpu",
        "cuda118": "https://download.pytorch.org/whl/nightly/cu118",
        "cuda121": "https://download.pytorch.org/whl/nightly/cu121",
        "cuda124": "https://download.pytorch.org/whl/nightly/cu124",
        "rocm62": "https://download.pytorch.org/whl/nightly/rocm6.2",
        "mps": None,  # Default PyPI for macOS
    },
}

# Package names
PACKAGES = {
    "core": ["torch", "torchvision", "torchaudio"],
    "torch_only": ["torch"],
}

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class InstallCommand:
    """Represents a generated install command."""

    command: str
    packages: List[str]
    index_url: Optional[str]
    notes: List[str] = field(default_factory=list)
    verified: bool = True  # False if we're uncertain about this combo


@dataclass
class InstallConfig:
    """User's selected installation configuration."""

    build: str = "Stable"
    os: str = "Linux"
    package_manager: str = "pip"
    compute: str = "cpu"
    include_audio: bool = True
    include_vision: bool = True


# =============================================================================
# Command Generation
# =============================================================================


def get_compute_options(os_name: str) -> List[Dict]:
    """Get available compute options for an OS."""
    return COMPUTE_PLATFORMS.get(os_name, COMPUTE_PLATFORMS["Linux"])


def get_index_url(build: str, compute: str) -> Optional[str]:
    """Get the wheel index URL for a build/compute combination."""
    build_key = "nightly" if "nightly" in build.lower() else "stable"
    urls = INDEX_URLS.get(build_key, INDEX_URLS["stable"])
    return urls.get(compute)


def generate_pip_command(config: InstallConfig) -> InstallCommand:
    """
    Generate a pip install command for the given configuration.

    IMPORTANT: This is guidance only. Always verify on the official selector.
    """
    packages = ["torch"]

    if config.include_vision:
        packages.append("torchvision")
    if config.include_audio:
        packages.append("torchaudio")

    index_url = get_index_url(config.build, config.compute)
    notes = []
    verified = True

    # Build the command
    if index_url:
        command = f"pip install {' '.join(packages)} --index-url {index_url}"
    else:
        # Default PyPI (macOS MPS case)
        command = f"pip install {' '.join(packages)}"

    # Add notes based on configuration
    if config.os == "macOS":
        notes.append("On Apple Silicon (M1/M2/M3), MPS backend is used automatically when available.")
        notes.append("Ensure you're using Python 3.9+ for Apple Silicon native support.")

    if "nightly" in config.build.lower():
        notes.append("Nightly builds may be unstable. Use for testing new features only.")

    if config.compute.startswith("cuda"):
        notes.append(f"Ensure you have {config.compute.upper().replace('CUDA', 'CUDA ')} drivers installed.")

    if config.compute.startswith("rocm"):
        notes.append("ROCm requires compatible AMD GPU and ROCm toolkit installation.")
        # ROCm combinations can be tricky
        verified = True  # Still mark as verified but add a note

    # Check for potentially unsupported combinations
    compute_options = get_compute_options(config.os)
    compute_ids = [opt["id"] for opt in compute_options]

    if config.compute not in compute_ids:
        verified = False
        notes.append(f"This compute option may not be available for {config.os}.")
        notes.append("Please verify on the official PyTorch selector.")

    return InstallCommand(
        command=command,
        packages=packages,
        index_url=index_url,
        notes=notes,
        verified=verified,
    )


def generate_verification_code() -> str:
    """Generate PyTorch verification code snippet."""
    return '''import torch

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Create a random tensor
x = torch.rand(5, 3)
print(f"Random tensor:\\n{x}")

# Check device availability
print(f"\\nDevice availability:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

# Check MPS (Apple Silicon)
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
mps_built = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
print(f"  MPS available: {mps_available}")
print(f"  MPS built: {mps_built}")

# Test tensor on best available device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif mps_available:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"\\nUsing device: {device}")
y = torch.rand(3, 3, device=device)
print(f"Tensor on {device}:\\n{y}")
'''


def get_conda_guidance() -> str:
    """Get guidance for conda installation."""
    return """# Conda Installation

For conda-based installation, please visit the official PyTorch selector:
https://pytorch.org/get-started/locally/

Select "Conda" as your package manager to get the appropriate command.

Example pattern (verify on official site):
```
conda install pytorch torchvision torchaudio <channel-options>
```
"""


def get_source_guidance() -> str:
    """Get guidance for building from source."""
    return """# Building from Source

For building PyTorch from source, please refer to:
https://github.com/pytorch/pytorch#from-source

This is recommended only for:
- Contributing to PyTorch
- Custom builds with specific features
- Platforms without pre-built wheels

For most users, pip or conda installation is recommended.
"""


def get_libtorch_guidance() -> str:
    """Get guidance for LibTorch (C++)."""
    return """# LibTorch (C++ Distribution)

For C++ development with LibTorch, please visit:
https://pytorch.org/get-started/locally/

Select "LibTorch" as your package type to download the appropriate archive.

Documentation: https://pytorch.org/cppdocs/
"""


# =============================================================================
# Validation
# =============================================================================


def validate_combination(config: InstallConfig) -> tuple[bool, List[str]]:
    """
    Validate an install configuration.

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []

    # Check OS-compute compatibility
    compute_options = get_compute_options(config.os)
    compute_ids = [opt["id"] for opt in compute_options]

    if config.compute not in compute_ids:
        warnings.append(
            f"'{config.compute}' may not be available for {config.os}. "
            "Please verify on the official selector."
        )

    # Check macOS-specific issues
    if config.os == "macOS" and config.compute.startswith("cuda"):
        warnings.append("CUDA is not supported on macOS. Use CPU or MPS (Apple Silicon).")
        return False, warnings

    # Check ROCm availability
    if config.compute.startswith("rocm") and config.os == "Windows":
        warnings.append("ROCm is not officially supported on Windows.")
        return False, warnings

    return len(warnings) == 0, warnings


# =============================================================================
# Matrix Export
# =============================================================================


def get_full_matrix() -> Dict:
    """Export the complete install matrix as a dictionary."""
    return {
        "builds": BUILDS,
        "operating_systems": OPERATING_SYSTEMS,
        "package_managers": PACKAGE_MANAGERS,
        "compute_platforms": COMPUTE_PLATFORMS,
        "index_urls": INDEX_URLS,
        "packages": PACKAGES,
        "official_url": "https://pytorch.org/get-started/locally/",
    }
