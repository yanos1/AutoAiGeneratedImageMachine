"""
ui/display.py
-------------
Guest-facing display implementations.

  LogDisplay    — prints status to console (headless / CI)
  OpenCVDisplay — full-screen OpenCV window with countdown/preview
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from PIL import Image

from core.interfaces import BaseDisplay

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log display — no GUI, console only
# ---------------------------------------------------------------------------

class LogDisplay(BaseDisplay):
    """Outputs all display events to the logger. No window required."""

    def show_idle(self) -> None:
        logger.info("DISPLAY: 💍 Waiting for guests…")

    def show_countdown(self, seconds: int) -> None:
        logger.info("DISPLAY: ⏱  %d…", seconds)

    def show_preview(self, image: Image.Image) -> None:
        logger.info("DISPLAY: 📸 Preview (%dx%d)", image.width, image.height)

    def show_printing(self) -> None:
        logger.info("DISPLAY: 🖨️  Printing your photo…")

    def show_done(self) -> None:
        logger.info("DISPLAY: ✅ Your photo is ready! Enjoy!")

    def show_error(self, message: str) -> None:
        logger.warning("DISPLAY: ❌ Error — %s", message)

    def close(self) -> None:
        logger.info("DISPLAY: closed")


# ---------------------------------------------------------------------------
# OpenCV display — full-screen window
# ---------------------------------------------------------------------------

class OpenCVDisplay(BaseDisplay):
    """
    Full-screen guest display using OpenCV.
    Shows idle animations, live countdown, photo preview, and status messages.

    Designed for a dedicated monitor or tablet facing guests.
    """

    # Colour palette (BGR for OpenCV)
    BG_COLOR = (20, 15, 25)
    GOLD = (50, 175, 212)
    WHITE = (255, 255, 255)
    RED = (60, 60, 220)

    def __init__(self, window_name: str = "Wedding Photo Booth", fullscreen: bool = False) -> None:
        self.window_name = window_name
        self.fullscreen = fullscreen
        self._cv2 = None
        self._np = None

    def _cv2_lazy(self):
        if self._cv2 is None:
            try:
                import cv2
                import numpy as np
                self._cv2 = cv2
                self._np = np
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                if self.fullscreen:
                    cv2.setWindowProperty(
                        self.window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
            except ImportError:
                raise RuntimeError("OpenCV not installed. Use LogDisplay or install opencv-python.")
        return self._cv2, self._np

    def _pil_to_cv2(self, image: Image.Image):
        cv2, np = self._cv2_lazy()
        arr = np.array(image.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _make_frame(self, text_lines: list[str], sub_lines: list[str] = None,
                     color=None, size=(1280, 720)) -> "np.ndarray":
        cv2, np = self._cv2_lazy()
        color = color or self.BG_COLOR
        frame = np.full((*size[::-1], 3), color, dtype=np.uint8)
        w, h = size

        # Decorative border
        cv2.rectangle(frame, (30, 30), (w - 30, h - 30), self.GOLD, 2)
        cv2.rectangle(frame, (36, 36), (w - 36, h - 36), (*self.GOLD[:2], 80), 1)

        y = h // 2 - (len(text_lines) * 60) // 2
        for line in text_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, 2)[0]
            x = (w - text_size[0]) // 2
            cv2.putText(frame, line, (x + 2, y + 2),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (0, 0, 0), 3)
            cv2.putText(frame, line, (x, y),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, self.WHITE, 2)
            y += 70

        if sub_lines:
            y += 20
            for line in sub_lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
                x = (w - text_size[0]) // 2
                cv2.putText(frame, line, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.GOLD, 1)
                y += 40

        return frame

    def _show(self, frame) -> None:
        cv2, _ = self._cv2_lazy()
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def show_idle(self) -> None:
        frame = self._make_frame(
            ["Welcome!"],
            ["Step into the frame", "Your photo will be taken automatically"],
        )
        self._show(frame)

    def show_countdown(self, seconds: int) -> None:
        frame = self._make_frame([str(seconds)], ["Get ready!"])
        self._show(frame)

    def show_preview(self, image: Image.Image) -> None:
        cv2, np = self._cv2_lazy()
        frame = self._pil_to_cv2(image)
        h, w = frame.shape[:2]
        cv2.putText(frame, "Your photo!", (w // 2 - 100, 60),
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, self.GOLD, 2)
        self._show(frame)

    def show_printing(self) -> None:
        frame = self._make_frame(
            ["Printing..."],
            ["Your photo will be ready in a moment!"],
        )
        self._show(frame)

    def show_done(self) -> None:
        frame = self._make_frame(
            ["Enjoy!"],
            ["Your photo is ready", "Thank you for celebrating with us!"],
        )
        self._show(frame)

    def show_error(self, message: str) -> None:
        frame = self._make_frame(
            ["One moment please"],
            ["We'll be right back!"],
            color=(30, 20, 20),
        )
        self._show(frame)
        logger.warning("OpenCVDisplay: showing error (hidden from guest): %s", message)

    def close(self) -> None:
        if self._cv2:
            self._cv2.destroyAllWindows()
