"""
hardware/cameras.py
-------------------
Camera implementations. All extend BaseCamera.

Available:
  MockCamera       — returns a generated test image (no hardware needed)
  WebcamCamera     — OpenCV-based webcam (index-based)
  GPhoto2Camera    — any gphoto2-compatible DSLR / mirrorless
  CanonCamera      — Canon CCAPI HTTP-based control (Canon EOS R series etc.)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from core.interfaces import BaseCamera, CaptureResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock camera — generates a synthetic test image
# ---------------------------------------------------------------------------

class MockCamera(BaseCamera):
    """Generates a synthetic image. No hardware required."""

    def __init__(self, resolution: tuple[int, int] = (1920, 1080)) -> None:
        self.resolution = resolution
        self._frame_count = 0

    def connect(self) -> None:
        logger.info("MockCamera connected (resolution=%dx%d)", *self.resolution)

    def capture(self) -> CaptureResult:
        self._frame_count += 1
        w, h = self.resolution

        # Gradient background
        img = Image.new("RGB", (w, h))
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            r = int(180 + 40 * (y / h))
            g = int(140 + 60 * (y / h))
            b = int(160 + 50 * (y / h))
            pixels[y, :] = [r, g, b]
        img = Image.fromarray(pixels)

        draw = ImageDraw.Draw(img)
        draw.text((w // 2 - 200, h // 2 - 30),
                  f"MOCK CAPTURE #{self._frame_count}",
                  fill=(255, 255, 255))
        draw.text((w // 2 - 160, h // 2 + 20),
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  fill=(220, 220, 220))

        logger.debug("MockCamera captured frame #%d", self._frame_count)
        return CaptureResult(image=img, metadata={"frame": self._frame_count, "source": "mock"})

    def disconnect(self) -> None:
        logger.info("MockCamera disconnected")


# ---------------------------------------------------------------------------
# Webcam camera — OpenCV
# ---------------------------------------------------------------------------

class WebcamCamera(BaseCamera):
    """
    Captures from a local webcam via OpenCV.
    device_index: 0 = default webcam, 1 = second camera, etc.
    """

    def __init__(self, device_index: int = 0, warmup_frames: int = 5) -> None:
        self.device_index = device_index
        self.warmup_frames = warmup_frames
        self._cap = None

    def connect(self) -> None:
        try:
            self._cap = cv2.VideoCapture(self.device_index)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open webcam at index {self.device_index}")
            # Warm up — first frames are often dark/blurry
            for _ in range(self.warmup_frames):
                self._cap.read()
            logger.info("WebcamCamera connected (device=%d)", self.device_index)
        except ImportError:
            raise RuntimeError("OpenCV (cv2) not installed. Run: pip install opencv-python")

    def capture(self) -> CaptureResult:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from webcam")
        # OpenCV uses BGR; convert to RGB for PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return CaptureResult(image=image, metadata={"device_index": self.device_index})

    def disconnect(self) -> None:
        if self._cap:
            self._cap.release()
        logger.info("WebcamCamera disconnected")


# ---------------------------------------------------------------------------
# GPhoto2 camera — any DSLR / mirrorless with gphoto2 support
# ---------------------------------------------------------------------------

class GPhoto2Camera(BaseCamera):
    """
    Controls a DSLR or mirrorless camera via python-gphoto2.
    Supports Canon, Nikon, Sony, Fuji, and many others.

    Requires: pip install gphoto2
    System dep: libgphoto2 (apt install libgphoto2-dev on Ubuntu)
    """

    def __init__(self, capture_target: str = "memory card") -> None:
        self.capture_target = capture_target
        self._camera = None

    def connect(self) -> None:
        try:
            import gphoto2 as gp
            self._camera = gp.Camera()
            self._camera.init()
            # Set capture target (memory card vs internal RAM)
            config = self._camera.get_config()
            target = config.get_child_by_name("capturetarget")
            target.set_value(self.capture_target)
            self._camera.set_config(config)
            logger.info("GPhoto2Camera connected")
        except ImportError:
            raise RuntimeError("gphoto2 not installed. Run: pip install gphoto2")

    def capture(self) -> CaptureResult:
        import gphoto2 as gp
        import io

        file_path = self._camera.capture(gp.GP_CAPTURE_IMAGE)
        camera_file = self._camera.file_get(
            file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL
        )
        file_data = camera_file.get_data_and_size()
        image = Image.open(io.BytesIO(file_data))
        image.load()
        logger.info("GPhoto2Camera captured: %s/%s", file_path.folder, file_path.name)
        return CaptureResult(image=image, metadata={"filename": file_path.name})

    def disconnect(self) -> None:
        if self._camera:
            self._camera.exit()
        logger.info("GPhoto2Camera disconnected")


# ---------------------------------------------------------------------------
# Canon CCAPI camera — Canon EOS R series HTTP API
# ---------------------------------------------------------------------------

class CanonCamera(BaseCamera):
    """
    Controls Canon cameras via the Canon Camera Connect API (CCAPI).
    The camera must have CCAPI enabled (EOS R5, R6, R3, etc.).

    Requires: pip install requests
    The camera and this machine must be on the same network.
    """

    def __init__(self, host: str = "192.168.1.2", port: int = 8080) -> None:
        self.base_url = f"http://{host}:{port}/ccapi"
        self._session = None

    def connect(self) -> None:
        try:
            import requests
            self._session = requests.Session()
            resp = self._session.get(f"{self.base_url}/ver100/deviceinformation", timeout=5)
            resp.raise_for_status()
            info = resp.json()
            logger.info("CanonCamera connected: %s", info.get("productname", "unknown"))
        except ImportError:
            raise RuntimeError("requests not installed. Run: pip install requests")
        except Exception as exc:
            raise RuntimeError(f"Cannot connect to Canon CCAPI at {self.base_url}: {exc}")

    def capture(self) -> CaptureResult:
        import requests
        import io

        # Trigger capture
        resp = self._session.post(
            f"{self.base_url}/ver100/shooting/control/shutterbutton",
            json={"af": True},
            timeout=10,
        )
        resp.raise_for_status()

        # Poll for the latest image
        time.sleep(1.5)
        storage_resp = self._session.get(
            f"{self.base_url}/ver100/contents/storage/nonremovable/100eos__",
            timeout=10,
        )
        storage_resp.raise_for_status()
        contents = storage_resp.json().get("contentlist", [])

        if not contents:
            raise RuntimeError("No images found on Canon camera after capture")

        latest = contents[-1]["url"]
        img_resp = self._session.get(latest, timeout=30)
        img_resp.raise_for_status()

        image = Image.open(io.BytesIO(img_resp.content))
        image.load()
        logger.info("CanonCamera captured image from %s", latest)
        return CaptureResult(image=image, metadata={"source_url": latest})

    def disconnect(self) -> None:
        if self._session:
            self._session.close()
        logger.info("CanonCamera disconnected")
