"""
core/pipeline.py
----------------
The main orchestrator. Receives all components via dependency injection —
never imports a concrete implementation. Fully testable with mocks.

Flow per session:
  1. Wait for MotionEvent
  2. Countdown display
  3. Capture image
  4. Enhance image
  5. Apply overlay
  6. Save to disk
  7. Print
  8. Show result to guest
  9. Reset
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from core.interfaces import (
    BaseCamera, BasePrinter, BaseMotionDetector,
    BaseEnhancer, BaseOverlayRenderer, BaseDisplay,
    SessionStatus, PrintJob,
)
from core.session import Session

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Dependency-injected photo pipeline.

    All components are passed in — swap any of them without touching
    this class. Add a new printer? Just pass it in. New AI model?
    Pass a different enhancer.
    """

    def __init__(
        self,
        camera: BaseCamera,
        printer: BasePrinter,
        motion_detector: BaseMotionDetector,
        enhancer: BaseEnhancer,
        overlay_renderer: BaseOverlayRenderer,
        display: BaseDisplay,
        output_dir: Path,
        config: dict,
    ) -> None:
        self.camera = camera
        self.printer = printer
        self.motion_detector = motion_detector
        self.enhancer = enhancer
        self.overlay_renderer = overlay_renderer
        self.display = display
        self.output_dir = output_dir
        self.config = config

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._running = False
        self._session_count = 0

        logger.info(
            "Pipeline initialised | camera=%s printer=%s detector=%s enhancer=%s overlay=%s",
            camera.name, printer.name, motion_detector.name,
            enhancer.name, overlay_renderer.name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main event loop. Runs until stop() is called."""
        self._running = True
        logger.info("Pipeline started — waiting for motion")

        with self.camera, self.printer:
            self.motion_detector.start()
            self.display.show_idle()

            try:
                while self._running:
                    self._run_one_session()
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            finally:
                self.motion_detector.stop()
                self.display.close()
                logger.info("Pipeline stopped after %d sessions", self._session_count)

    def stop(self) -> None:
        self._running = False

    def run_single_session(self) -> Session:
        """Run exactly one session and return it. Useful for testing."""
        with self.camera, self.printer:
            self.motion_detector.start()
            try:
                return self._run_one_session()
            finally:
                self.motion_detector.stop()

    # ------------------------------------------------------------------
    # Internal session logic
    # ------------------------------------------------------------------

    def _run_one_session(self) -> Session:
        session = Session()
        self._session_count += 1
        logger.info("[%s] Session #%d started", session.session_id, self._session_count)

        try:
            self._wait_for_motion(session)
            self._countdown(session)
            self._capture(session)
            self._enhance(session)
            self._overlay(session)
            self._print(session)
            self._complete(session)
        except Exception as exc:
            self._handle_error(session, exc)

        logger.info("[%s] Session ended: %s (%.1fs)",
                    session.session_id, session.status.name, session.duration_seconds())
        return session

    def _wait_for_motion(self, session: Session) -> None:
        session.transition(SessionStatus.IDLE)
        self.display.show_idle()

        timeout = self.config.get("motion_timeout_seconds")
        event = self.motion_detector.wait_for_motion(timeout=timeout)

        if event is None:
            raise TimeoutError("No motion detected within timeout")

        session.motion_event = event
        session.transition(SessionStatus.MOTION_DETECTED)
        logger.info("[%s] Motion detected (confidence=%.2f)", session.session_id, event.confidence)

    def _countdown(self, session: Session) -> None:
        countdown = self.config.get("countdown_seconds", 3)
        for i in range(countdown, 0, -1):
            self.display.show_countdown(i)
            time.sleep(1)

    def _capture(self, session: Session) -> None:
        session.transition(SessionStatus.CAPTURING)
        result = self.camera.capture()
        session.capture_result = result
        logger.info("[%s] Captured image %dx%d",
                    session.session_id, result.image.width, result.image.height)

    def _enhance(self, session: Session) -> None:
        session.transition(SessionStatus.PROCESSING)
        assert session.capture_result

        enhanced = self.enhancer.enhance(session.capture_result.image)
        path = self.output_dir / f"{session.session_id}_enhanced.jpg"
        enhanced.save(path, quality=95)
        session.enhanced_image_path = path
        logger.info("[%s] Enhanced → %s", session.session_id, path)

    def _overlay(self, session: Session) -> None:
        assert session.enhanced_image_path
        img = Image.open(session.enhanced_image_path)

        context = self.config.get("overlay_context", {})
        composited = self.overlay_renderer.apply(img, context)

        path = self.output_dir / f"{session.session_id}_final.jpg"
        composited.save(path, quality=95)
        session.overlaid_image_path = path

        self.display.show_preview(composited)
        logger.info("[%s] Overlay applied → %s", session.session_id, path)

    def _print(self, session: Session) -> None:
        session.transition(SessionStatus.PRINTING)
        assert session.overlaid_image_path

        self.display.show_printing()
        job = PrintJob(
            image_path=session.overlaid_image_path,
            copies=self.config.get("print_copies", 1),
            job_id=session.session_id,
        )
        result = self.printer.print_image(job)
        session.print_result = result

        if not result.success:
            raise RuntimeError(f"Print failed: {result.message}")
        logger.info("[%s] Print job submitted: %s", session.session_id, result.message)

    def _complete(self, session: Session) -> None:
        session.transition(SessionStatus.COMPLETE)
        self.display.show_done()
        cooldown = self.config.get("cooldown_seconds", 5)
        time.sleep(cooldown)

    def _handle_error(self, session: Session, exc: Exception) -> None:
        session.error = str(exc)
        session.transition(SessionStatus.ERROR)
        self.display.show_error(str(exc))
        logger.exception("[%s] Session error: %s", session.session_id, exc)
        time.sleep(self.config.get("error_cooldown_seconds", 3))
