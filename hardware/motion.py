"""
hardware/motion.py
------------------
Motion detector implementations.

  MockMotionDetector   — fires after a configurable delay (testing/demo)
  OpenCVMotionDetector — real motion detection via background subtraction
  KeyboardMotionDetector — press ENTER to simulate motion (demo mode)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from core.interfaces import BaseMotionDetector, MotionEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock — fires after a delay
# ---------------------------------------------------------------------------

class MockMotionDetector(BaseMotionDetector):
    """Fires a motion event after `delay_seconds`. Useful for testing."""

    def __init__(self, delay_seconds: float = 2.0, confidence: float = 0.95) -> None:
        self.delay_seconds = delay_seconds
        self.confidence = confidence
        self._event_queue: queue.Queue[Optional[MotionEvent]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._emit_loop, daemon=True)
        self._thread.start()
        logger.info("MockMotionDetector started (delay=%.1fs)", self.delay_seconds)

    def stop(self) -> None:
        self._running = False
        self._event_queue.put(None)  # unblock any waiting call

    def _emit_loop(self) -> None:
        while self._running:
            time.sleep(self.delay_seconds)
            if self._running:
                self._event_queue.put(MotionEvent(confidence=self.confidence))

    def wait_for_motion(self, timeout: Optional[float] = None) -> Optional[MotionEvent]:
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# Keyboard — press ENTER to trigger (great for live demos)
# ---------------------------------------------------------------------------

class KeyboardMotionDetector(BaseMotionDetector):
    """Press ENTER in the terminal to simulate a motion trigger."""

    def __init__(self) -> None:
        self._event_queue: queue.Queue[Optional[MotionEvent]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        logger.info("KeyboardMotionDetector started — press ENTER to trigger")

    def stop(self) -> None:
        self._running = False
        self._event_queue.put(None)

    def _listen(self) -> None:
        while self._running:
            input()  # blocks until ENTER
            if self._running:
                logger.info("Keyboard trigger received")
                self._event_queue.put(MotionEvent(confidence=1.0))

    def wait_for_motion(self, timeout: Optional[float] = None) -> Optional[MotionEvent]:
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# OpenCV — real background-subtraction motion detection
# ---------------------------------------------------------------------------

class OpenCVMotionDetector(BaseMotionDetector):
    """
    Real motion detection using OpenCV background subtraction (MOG2).

    Watches a camera feed and fires when significant motion is detected
    in the frame. Designed to be sensitive enough to catch guests walking
    into the photo zone, but not noise-triggering from subtle changes.

    Parameters
    ----------
    device_index    : int   — webcam index (0 = default)
    sensitivity     : float — min contour area as fraction of frame (0.01–0.10)
    min_confidence  : float — minimum confidence to fire (0.0–1.0)
    cooldown        : float — seconds to ignore after a trigger
    """

    def __init__(
        self,
        device_index: int = 0,
        sensitivity: float = 0.03,
        min_confidence: float = 0.6,
        cooldown: float = 2.0,
    ) -> None:
        self.device_index = device_index
        self.sensitivity = sensitivity
        self.min_confidence = min_confidence
        self.cooldown = cooldown

        self._event_queue: queue.Queue[Optional[MotionEvent]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()
        logger.info("OpenCVMotionDetector started (device=%d, sensitivity=%.2f)",
                    self.device_index, self.sensitivity)

    def stop(self) -> None:
        self._running = False
        self._event_queue.put(None)

    def _detect_loop(self) -> None:
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not installed. Run: pip install opencv-python")
            return

        cap = cv2.VideoCapture(self.device_index)
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

        last_trigger = 0.0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            frame_area = h * w

            # Apply background subtraction
            fg_mask = subtractor.apply(frame)

            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            now = time.time()
            if now - last_trigger < self.cooldown:
                continue

            for contour in contours:
                area = cv2.contourArea(contour)
                fraction = area / frame_area

                if fraction >= self.sensitivity:
                    confidence = min(1.0, fraction / (self.sensitivity * 3))
                    if confidence >= self.min_confidence:
                        x, y, cw, ch = cv2.boundingRect(contour)
                        event = MotionEvent(
                            confidence=round(confidence, 3),
                            region=(x, y, cw, ch),
                        )
                        logger.info(
                            "Motion detected: confidence=%.2f area=%.3f region=%s",
                            confidence, fraction, event.region
                        )
                        self._event_queue.put(event)
                        last_trigger = now
                        break  # one event per frame scan

        cap.release()

    def wait_for_motion(self, timeout: Optional[float] = None) -> Optional[MotionEvent]:
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None
