"""
core/interfaces.py
------------------
Abstract base classes for every swappable component.
Swap a camera, printer, enhancer, or detector by subclassing
the relevant ABC and implementing the required methods.
No other code changes needed.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional
from PIL import Image


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class SessionStatus(Enum):
    IDLE = auto()
    MOTION_DETECTED = auto()
    CAPTURING = auto()
    PROCESSING = auto()
    PRINTING = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class CaptureResult:
    image: Image.Image
    captured_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class PrintJob:
    image_path: Path
    copies: int = 1
    job_id: Optional[str] = None
    submitted_at: datetime = field(default_factory=datetime.now)


@dataclass
class PrintResult:
    job: PrintJob
    success: bool
    message: str = ""
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class MotionEvent:
    confidence: float           # 0.0 – 1.0
    region: Optional[tuple] = None    # (x, y, w, h) bounding box
    detected_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class BaseCamera(abc.ABC):
    """Abstract camera. Implement connect / capture / disconnect."""

    def __enter__(self) -> "BaseCamera":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def capture(self) -> CaptureResult: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

class BasePrinter(abc.ABC):
    """Abstract printer. Implement connect / print_image / disconnect."""

    def __enter__(self) -> "BasePrinter":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def print_image(self, job: PrintJob) -> PrintResult: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Motion detector
# ---------------------------------------------------------------------------

class BaseMotionDetector(abc.ABC):
    """Abstract motion detector."""

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def wait_for_motion(self, timeout: Optional[float] = None) -> Optional[MotionEvent]:
        """Block until motion detected or timeout. Returns None on timeout."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Image enhancer (AI / processing model)
# ---------------------------------------------------------------------------

class BaseEnhancer(abc.ABC):
    """
    Abstract image enhancer.
    Swap between Pillow, OpenAI Vision, a local ML model, or a mock.
    """

    @abc.abstractmethod
    def enhance(self, image: Image.Image) -> Image.Image: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Overlay renderer
# ---------------------------------------------------------------------------

class BaseOverlayRenderer(abc.ABC):
    """Abstract overlay / branding compositor."""

    @abc.abstractmethod
    def apply(self, image: Image.Image, context: dict) -> Image.Image:
        """Composite overlay. context: {couple_names, date, event_name, ...}"""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Guest display
# ---------------------------------------------------------------------------

class BaseDisplay(abc.ABC):
    """Abstract guest-facing display."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def show_idle(self) -> None: ...

    @abc.abstractmethod
    def show_countdown(self, seconds: int) -> None: ...

    @abc.abstractmethod
    def show_preview(self, image: Image.Image) -> None: ...

    @abc.abstractmethod
    def show_printing(self) -> None: ...

    @abc.abstractmethod
    def show_done(self) -> None: ...

    @abc.abstractmethod
    def show_error(self, message: str) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...
