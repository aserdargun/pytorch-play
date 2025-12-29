"""
Global state management for PyTorch Playground.

Manages:
- Level selection (Beginner/Intermediate/Advanced)
- Device selection (auto/cpu/cuda/mps)
- Random seed and deterministic mode
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, List
import threading


class Level(Enum):
    """User expertise level that controls UI complexity and content depth."""

    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"

    @property
    def max_params(self) -> int:
        """Maximum number of parameters to show in demo UIs."""
        return {
            Level.BEGINNER: 6,
            Level.INTERMEDIATE: 12,
            Level.ADVANCED: 50,  # "full panel"
        }[self]

    @property
    def max_epochs(self) -> int:
        """Maximum training epochs allowed."""
        return {
            Level.BEGINNER: 5,
            Level.INTERMEDIATE: 20,
            Level.ADVANCED: 100,
        }[self]

    @property
    def explanation_depth(self) -> str:
        """How much detail to show in explanations."""
        return {
            Level.BEGINNER: "minimal",
            Level.INTERMEDIATE: "moderate",
            Level.ADVANCED: "deep",
        }[self]

    @property
    def show_code(self) -> bool:
        """Whether to show code snippets by default."""
        return self != Level.BEGINNER

    @property
    def show_math(self) -> bool:
        """Whether to show mathematical formulas."""
        return self == Level.ADVANCED


class DeviceSelection(Enum):
    """Device selection options."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class AppState:
    """
    Global application state.

    Thread-safe singleton that manages user preferences and session state.
    """

    level: Level = Level.BEGINNER
    device_selection: DeviceSelection = DeviceSelection.AUTO
    seed: Optional[int] = 42
    deterministic: bool = False

    # Session tracking
    session_id: str = ""

    # Callbacks for state changes
    _level_callbacks: List[Callable] = field(default_factory=list)
    _device_callbacks: List[Callable] = field(default_factory=list)

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_level(self, level: Level | str) -> None:
        """Set the current level and notify callbacks."""
        with self._lock:
            if isinstance(level, str):
                level = Level(level)
            self.level = level
            for callback in self._level_callbacks:
                try:
                    callback(level)
                except Exception:
                    pass

    def set_device(self, device: DeviceSelection | str) -> None:
        """Set the device selection and notify callbacks."""
        with self._lock:
            if isinstance(device, str):
                device = DeviceSelection(device)
            self.device_selection = device
            for callback in self._device_callbacks:
                try:
                    callback(device)
                except Exception:
                    pass

    def set_seed(self, seed: Optional[int]) -> None:
        """Set the random seed."""
        with self._lock:
            self.seed = seed

    def set_deterministic(self, deterministic: bool) -> None:
        """Set deterministic mode."""
        with self._lock:
            self.deterministic = deterministic

    def on_level_change(self, callback: Callable) -> None:
        """Register a callback for level changes."""
        self._level_callbacks.append(callback)

    def on_device_change(self, callback: Callable) -> None:
        """Register a callback for device changes."""
        self._device_callbacks.append(callback)

    def reset(self) -> None:
        """Reset to default state."""
        with self._lock:
            self.level = Level.BEGINNER
            self.device_selection = DeviceSelection.AUTO
            self.seed = 42
            self.deterministic = False

    def to_dict(self) -> dict:
        """Export state as dictionary."""
        return {
            "level": self.level.value,
            "device_selection": self.device_selection.value,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppState":
        """Create state from dictionary."""
        return cls(
            level=Level(data.get("level", "Beginner")),
            device_selection=DeviceSelection(data.get("device_selection", "auto")),
            seed=data.get("seed", 42),
            deterministic=data.get("deterministic", False),
        )


# Global singleton instance
_global_state: Optional[AppState] = None
_state_lock = threading.Lock()


def get_state() -> AppState:
    """Get the global application state singleton."""
    global _global_state
    with _state_lock:
        if _global_state is None:
            _global_state = AppState()
        return _global_state


def reset_state() -> AppState:
    """Reset and return the global application state."""
    global _global_state
    with _state_lock:
        _global_state = AppState()
        return _global_state


def get_level_choices() -> List[str]:
    """Get list of level choices for UI dropdowns."""
    return [level.value for level in Level]


def get_device_choices() -> List[str]:
    """Get list of device choices for UI dropdowns."""
    return [device.value for device in DeviceSelection]
