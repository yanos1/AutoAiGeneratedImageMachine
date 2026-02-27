"""
tests/test_pipeline.py
----------------------
Unit tests for the pipeline using fully mocked components.
Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from core.interfaces import (
    CaptureResult, PrintJob, PrintResult, MotionEvent, SessionStatus,
)
from core.pipeline import Pipeline
from core.session import Session

from hardware.cameras import MockCamera
from hardware.printers import MockPrinter, FilePrinter
from hardware.motion import MockMotionDetector, KeyboardMotionDetector
from ai.enhancers import MockEnhancer, PillowAnimeEnhancer
from ai.overlays import NoOverlay, WeddingOverlayRenderer, MinimalOverlayRenderer
from ui.display import LogDisplay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pipeline(tmp_path: Path, **overrides) -> Pipeline:
    """Build a Pipeline with all mocks; override any component via kwargs."""
    defaults = dict(
        camera=MockCamera(resolution=(640, 480)),
        printer=MockPrinter(simulated_delay=0),
        motion_detector=MockMotionDetector(delay_seconds=0.01),
        enhancer=MockEnhancer(),
        overlay_renderer=NoOverlay(),
        display=LogDisplay(),
        output_dir=tmp_path,
        config={
            "countdown_seconds": 0,
            "cooldown_seconds": 0,
            "error_cooldown_seconds": 0,
        },
    )
    defaults.update(overrides)
    return Pipeline(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMockCamera(unittest.TestCase):
    def test_capture_returns_image(self):
        cam = MockCamera(resolution=(800, 600))
        cam.connect()
        result = cam.capture()
        self.assertIsInstance(result.image, Image.Image)
        self.assertEqual(result.image.size, (800, 600))
        cam.disconnect()

    def test_frame_count_increments(self):
        cam = MockCamera()
        cam.connect()
        r1 = cam.capture()
        r2 = cam.capture()
        self.assertEqual(r1.metadata["frame"], 1)
        self.assertEqual(r2.metadata["frame"], 2)
        cam.disconnect()

    def test_context_manager(self):
        with MockCamera() as cam:
            result = cam.capture()
        self.assertIsNotNone(result.image)


class TestMockPrinter(unittest.TestCase):
    def test_print_returns_success(self):
        printer = MockPrinter(simulated_delay=0)
        printer.connect()
        job = PrintJob(image_path=Path("fake.jpg"), job_id="test-001")
        result = printer.print_image(job)
        self.assertTrue(result.success)
        self.assertEqual(len(printer.jobs), 1)
        printer.disconnect()


class TestFilePrinter(unittest.TestCase):
    def test_saves_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create a test image
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            src = tmp_path / "test.jpg"
            img.save(src)

            printer = FilePrinter(output_dir=tmp_path / "out")
            printer.connect()
            job = PrintJob(image_path=src, job_id="abc")
            result = printer.print_image(job)
            self.assertTrue(result.success)
            # Verify file was copied
            out_files = list((tmp_path / "out").iterdir())
            self.assertEqual(len(out_files), 1)


class TestMockMotionDetector(unittest.TestCase):
    def test_fires_event(self):
        detector = MockMotionDetector(delay_seconds=0.05)
        detector.start()
        event = detector.wait_for_motion(timeout=1.0)
        detector.stop()
        self.assertIsNotNone(event)
        self.assertIsInstance(event, MotionEvent)
        self.assertEqual(event.confidence, 0.95)

    def test_timeout_returns_none(self):
        detector = MockMotionDetector(delay_seconds=10)
        detector.start()
        event = detector.wait_for_motion(timeout=0.05)
        detector.stop()
        self.assertIsNone(event)


class TestPillowAnimeEnhancer(unittest.TestCase):
    def test_returns_rgb_image(self):
        enhancer = PillowAnimeEnhancer()
        img = Image.new("RGB", (200, 200), color=(128, 100, 90))
        result = enhancer.enhance(img)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (200, 200))

    def test_no_crash_on_various_modes(self):
        enhancer = PillowAnimeEnhancer()
        for mode in ("RGB", "RGBA", "L"):
            img = Image.new(mode, (100, 100))
            result = enhancer.enhance(img)
            self.assertEqual(result.mode, "RGB")

    def test_produces_different_image(self):
        """Anime transform should actually change pixel values."""
        enhancer = PillowAnimeEnhancer()
        img = Image.new("RGB", (100, 100), color=(180, 150, 130))
        # Fill with gradient so there are edges to detect
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            arr[i, :] = [i * 2, 100, 200 - i]
        img = Image.fromarray(arr)
        result = enhancer.enhance(img)
        self.assertFalse(np.array_equal(np.array(img), np.array(result)),
                         "Anime enhancer should change the image")


class TestOverlays(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (1920, 1080), color=(100, 80, 90))
        self.context = {
            "couple_names": "Alice & Bob",
            "date": "January 1, 2026",
            "event_name": "Wedding Reception",
        }

    def test_no_overlay_passthrough(self):
        renderer = NoOverlay()
        result = renderer.apply(self.img, self.context)
        self.assertIs(result, self.img)

    def test_wedding_overlay_returns_rgb(self):
        renderer = WeddingOverlayRenderer()
        result = renderer.apply(self.img, self.context)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, self.img.size)

    def test_minimal_overlay_returns_rgb(self):
        renderer = MinimalOverlayRenderer()
        result = renderer.apply(self.img, self.context)
        self.assertEqual(result.mode, "RGB")

    def test_overlay_empty_context(self):
        renderer = WeddingOverlayRenderer()
        result = renderer.apply(self.img, {})
        self.assertEqual(result.mode, "RGB")


class TestSession(unittest.TestCase):
    def test_transitions(self):
        session = Session()
        self.assertEqual(session.status, SessionStatus.IDLE)
        session.transition(SessionStatus.CAPTURING)
        self.assertEqual(session.status, SessionStatus.CAPTURING)
        self.assertEqual(len(session.transitions), 1)

    def test_summary(self):
        session = Session()
        session.transition(SessionStatus.COMPLETE)
        summary = session.summary()
        self.assertIn("session_id", summary)
        self.assertIn("status", summary)
        self.assertEqual(summary["status"], "COMPLETE")


class TestPipelineFullCycle(unittest.TestCase):
    """Integration test: full session with all mocks."""

    def test_single_session_completes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = make_pipeline(Path(tmp))
            session = pipeline.run_single_session()
            self.assertEqual(session.status, SessionStatus.COMPLETE)
            self.assertIsNotNone(session.capture_result)
            self.assertIsNotNone(session.enhanced_image_path)
            self.assertIsNotNone(session.overlaid_image_path)
            self.assertIsNotNone(session.print_result)
            self.assertTrue(session.print_result.success)

    def test_output_files_created(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pipeline = make_pipeline(tmp_path)
            session = pipeline.run_single_session()
            self.assertTrue(session.enhanced_image_path.exists())
            self.assertTrue(session.overlaid_image_path.exists())

    def test_session_duration_tracked(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = make_pipeline(Path(tmp))
            session = pipeline.run_single_session()
            self.assertGreater(session.duration_seconds(), 0)

    def test_camera_error_leads_to_error_status(self):
        import tempfile
        broken_camera = MockCamera()
        broken_camera.capture = MagicMock(side_effect=RuntimeError("Camera disconnected"))

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = make_pipeline(Path(tmp), camera=broken_camera)
            session = pipeline.run_single_session()
            self.assertEqual(session.status, SessionStatus.ERROR)
            self.assertIn("Camera disconnected", session.error)

    def test_printer_failure_leads_to_error_status(self):
        import tempfile
        broken_printer = MockPrinter()
        broken_printer.print_image = MagicMock(
            return_value=PrintResult(
                job=PrintJob(image_path=Path("x.jpg")),
                success=False,
                message="Out of paper",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = make_pipeline(Path(tmp), printer=broken_printer)
            session = pipeline.run_single_session()
            self.assertEqual(session.status, SessionStatus.ERROR)

    def test_with_wedding_overlay(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = make_pipeline(
                Path(tmp),
                overlay_renderer=WeddingOverlayRenderer(),
                config={
                    "countdown_seconds": 0,
                    "cooldown_seconds": 0,
                    "error_cooldown_seconds": 0,
                    "overlay_context": {
                        "couple_names": "Test & Test",
                        "date": "2025-01-01",
                    },
                },
            )
            session = pipeline.run_single_session()
            self.assertEqual(session.status, SessionStatus.COMPLETE)

    def test_with_pillow_anime_enhancer(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = make_pipeline(Path(tmp), enhancer=PillowAnimeEnhancer())
            session = pipeline.run_single_session()
            self.assertEqual(session.status, SessionStatus.COMPLETE)


class TestConfigFactory(unittest.TestCase):
    def test_load_defaults(self):
        from utils.config import load_config
        config = load_config("nonexistent.yaml")
        self.assertIn("hardware", config)
        self.assertEqual(config["hardware"]["camera"], "mock")

    def test_build_mock_components(self):
        from utils.config import load_config, build_components
        config = load_config("nonexistent.yaml")
        components = build_components(config)
        self.assertIn("camera", components)
        self.assertIn("printer", components)
        self.assertIn("motion_detector", components)
        self.assertIn("enhancer", components)
        self.assertIsInstance(components["enhancer"], PillowAnimeEnhancer)
        self.assertIn("overlay_renderer", components)
        self.assertIn("display", components)


if __name__ == "__main__":
    unittest.main(verbosity=2)
