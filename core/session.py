"""
core/session.py
---------------
Immutable-ish session record for one photo cycle.
Tracks every state transition with timestamps — useful for logging,
analytics, and post-event review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

from core.interfaces import SessionStatus, CaptureResult, PrintResult, MotionEvent


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)

    status: SessionStatus = SessionStatus.IDLE
    transitions: list[tuple[SessionStatus, datetime]] = field(default_factory=list)

    motion_event: Optional[MotionEvent] = None
    capture_result: Optional[CaptureResult] = None
    enhanced_image_path: Optional[Path] = None
    overlaid_image_path: Optional[Path] = None
    print_result: Optional[PrintResult] = None
    error: Optional[str] = None

    def transition(self, new_status: SessionStatus) -> None:
        self.transitions.append((self.status, datetime.now()))
        self.status = new_status

    def duration_seconds(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status.name,
            "duration_s": round(self.duration_seconds(), 2),
            "transitions": [(s.name, t.isoformat()) for s, t in self.transitions],
            "error": self.error,
        }
