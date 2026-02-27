"""
hardware/printers.py
--------------------
Printer implementations.

  MockPrinter   — logs the print job, no hardware needed
  FilePrinter   — saves the image to a directory (great for dev/testing)
  CUPSPrinter   — real printing via CUPS (Linux/macOS)
  DNPPrinter    — DNP photo printer (common at wedding/event booths)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from core.interfaces import BasePrinter, PrintJob, PrintResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock printer — no hardware
# ---------------------------------------------------------------------------

class MockPrinter(BasePrinter):
    """Simulates a printer. Logs the job and returns success. No hardware."""

    def __init__(self, simulated_delay: float = 1.5) -> None:
        self.simulated_delay = simulated_delay
        self.jobs: list[PrintJob] = []

    def connect(self) -> None:
        logger.info("MockPrinter connected")

    def print_image(self, job: PrintJob) -> PrintResult:
        import time
        self.jobs.append(job)
        logger.info("MockPrinter: printing %s (x%d) [simulating %.1fs]",
                    job.image_path, job.copies, self.simulated_delay)
        time.sleep(self.simulated_delay)
        return PrintResult(job=job, success=True, message=f"Mock print complete: {job.image_path.name}")

    def disconnect(self) -> None:
        logger.info("MockPrinter disconnected (%d total jobs)", len(self.jobs))


# ---------------------------------------------------------------------------
# File printer — saves to a folder
# ---------------------------------------------------------------------------

class FilePrinter(BasePrinter):
    """
    'Prints' by copying the image to an z_output folder.
    Useful during development to inspect print-ready images without hardware.
    """

    def __init__(self, output_dir: Path = Path("printed_output")) -> None:
        self.output_dir = Path(output_dir)

    def connect(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("FilePrinter connected → %s", self.output_dir)

    def print_image(self, job: PrintJob) -> PrintResult:
        dest = self.output_dir / f"print_{job.job_id or uuid.uuid4().hex[:6]}_{job.image_path.name}"
        shutil.copy2(job.image_path, dest)
        logger.info("FilePrinter: saved %s", dest)
        return PrintResult(job=job, success=True, message=f"Saved to {dest}")

    def disconnect(self) -> None:
        logger.info("FilePrinter disconnected")


# ---------------------------------------------------------------------------
# CUPS printer — real printing on Linux/macOS
# ---------------------------------------------------------------------------

class CUPSPrinter(BasePrinter):
    """
    Submits print jobs to a CUPS printer queue.
    Works on Linux and macOS. Windows users should use FilePrinter for now.

    printer_name: the CUPS queue name (run `lpstat -p` to list queues)
    media:        paper size string (e.g. "4x6in", "Letter", "A4")
    """

    def __init__(
        self,
        printer_name: str,
        media: str = "4x6in",
        resolution_dpi: int = 300,
    ) -> None:
        self.printer_name = printer_name
        self.media = media
        self.resolution_dpi = resolution_dpi

    def connect(self) -> None:
        result = subprocess.run(["lpstat", "-p", self.printer_name],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CUPS printer '{self.printer_name}' not found. "
                               f"Run `lpstat -p` to list available printers.")
        logger.info("CUPSPrinter connected: %s", self.printer_name)

    def print_image(self, job: PrintJob) -> PrintResult:
        cmd = [
            "lp",
            "-d", self.printer_name,
            "-n", str(job.copies),
            "-o", f"media={self.media}",
            "-o", f"Resolution={self.resolution_dpi}dpi",
            "-o", "fit-to-page",
            str(job.image_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return PrintResult(job=job, success=False, message=result.stderr.strip())

        job_id = result.stdout.strip()
        logger.info("CUPSPrinter submitted job: %s", job_id)
        return PrintResult(job=job, success=True, message=job_id)

    def disconnect(self) -> None:
        logger.info("CUPSPrinter disconnected")


# ---------------------------------------------------------------------------
# DNP printer — photo booth standard
# ---------------------------------------------------------------------------

class DNPPrinter(BasePrinter):
    """
    Controls a DNP photo printer (DS620, DS820, etc.)
    via the dnpds40 driver or gutenprint.

    DNP printers are common at wedding booths and produce lab-quality prints.
    This wraps CUPS with DNP-specific media and color settings.
    """

    DNP_MEDIA_SIZES = {
        "4x6": "w288h432",
        "5x7": "w360h504",
        "6x8": "w432h576",
        "6x9": "w432h648",
    }

    def __init__(self, printer_name: str, size: str = "4x6") -> None:
        self.printer_name = printer_name
        self.size = size
        if size not in self.DNP_MEDIA_SIZES:
            raise ValueError(f"Unsupported DNP size: {size}. Choose from {list(self.DNP_MEDIA_SIZES)}")

    def connect(self) -> None:

        result = subprocess.run(["lpstat", "-p", self.printer_name], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"DNP printer '{self.printer_name}' not found in CUPS")
        logger.info("DNPPrinter connected: %s (%s)", self.printer_name, self.size)

    def print_image(self, job: PrintJob) -> PrintResult:
        media_code = self.DNP_MEDIA_SIZES[self.size]
        cmd = [
            "lp",
            "-d", self.printer_name,
            "-n", str(job.copies),
            "-o", f"PageSize={media_code}",
            "-o", "ColorModel=CMYK",
            "-o", "Quality=2",   # 2 = high quality on DNP
            str(job.image_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return PrintResult(job=job, success=False, message=result.stderr.strip())
        logger.info("DNPPrinter submitted: %s", result.stdout.strip())
        return PrintResult(job=job, success=True, message=result.stdout.strip())

    def disconnect(self) -> None:
        logger.info("DNPPrinter disconnected")
