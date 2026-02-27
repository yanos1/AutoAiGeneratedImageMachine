"""
utils/config.py
---------------
Configuration loading and component factory.

Reads config.yaml (and optional env overrides), then instantiates the
correct concrete classes for each component.

To add a new implementation, just register it in the relevant factory dict.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from core.interfaces import (
    BaseCamera, BasePrinter, BaseMotionDetector,
    BaseEnhancer, BaseOverlayRenderer, BaseDisplay,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "hardware": {
        "camera": "mock",
        "printer": "file",
        "motion_detector": "mock",
    },
    "ai": {
        "enhancer": "pillow_anime",
        "overlay": "wedding",
    },
    "pipeline": {
        "countdown_seconds": 3,
        "cooldown_seconds": 5,
        "error_cooldown_seconds": 3,
        "motion_timeout_seconds": None,
        "print_copies": 1,
    },
    "output_dir": "z_output",
    "overlay_context": {
        "couple_names": "Sarah & James",
        "date": "June 14, 2025",
        "event_name": "Wedding Reception",
    },
    "display": "log",
    "logging": {"level": "INFO"},
}


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load YAML config, merging over defaults. Env vars WIN over file values."""
    config = _deep_merge(DEFAULTS, {})

    path = Path(path)
    if path.exists():
        with open(path) as f:
            file_cfg = yaml.safe_load(f) or {}
        config = _deep_merge(config, file_cfg)

    # Env var overrides (e.g. WEDDING_CAMERA=webcam)
    env_map = {
        "WEDDING_CAMERA": ("hardware", "camera"),
        "WEDDING_PRINTER": ("hardware", "printer"),
        "WEDDING_MOTION": ("hardware", "motion_detector"),
        "WEDDING_ENHANCER": ("ai", "enhancer"),
        "WEDDING_OVERLAY": ("ai", "overlay"),
        "WEDDING_DISPLAY": ("display",),
        "WEDDING_OUTPUT_DIR": ("output_dir",),
    }
    for env_key, cfg_path in env_map.items():
        val = os.environ.get(env_key)
        if val:
            node = config
            for part in cfg_path[:-1]:
                node = node[part]
            node[cfg_path[-1]] = val

    return config


def build_components(config: dict) -> dict:
    """
    Factory: read config and return instantiated components.
    Returns a dict with keys: camera, printer, motion_detector,
    enhancer, overlay_renderer, display.
    """
    hw = config.get("hardware", {})
    ai = config.get("ai", {})

    return {
        "camera": _build_camera(hw, config),
        "printer": _build_printer(hw, config),
        "motion_detector": _build_motion_detector(hw, config),
        "enhancer": _build_enhancer(ai, config),
        "overlay_renderer": _build_overlay(ai, config),
        "display": _build_display(config),
    }


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------

def _build_camera(hw: dict, config: dict) -> BaseCamera:
    from hardware.cameras import MockCamera, WebcamCamera, GPhoto2Camera, CanonCamera

    kind = hw.get("camera", "mock").lower()
    cam_cfg = config.get("camera_options", {})

    registry = {
        "mock": lambda: MockCamera(resolution=tuple(cam_cfg.get("resolution", [1920, 1080]))),
        "webcam": lambda: WebcamCamera(device_index=cam_cfg.get("device_index", 0)),
        "gphoto2": lambda: GPhoto2Camera(capture_target=cam_cfg.get("capture_target", "memory card")),
        "canon": lambda: CanonCamera(
            host=cam_cfg.get("host", "192.168.1.2"),
            port=cam_cfg.get("port", 8080),
        ),
    }
    if kind not in registry:
        raise ValueError(f"Unknown camera: '{kind}'. Options: {list(registry)}")
    return registry[kind]()


def _build_printer(hw: dict, config: dict) -> BasePrinter:
    from hardware.printers import MockPrinter, FilePrinter, CUPSPrinter, DNPPrinter

    kind = hw.get("printer", "file").lower()
    p_cfg = config.get("printer_options", {})
    output_dir = Path(config.get("output_dir", "z_output"))

    registry = {
        "mock": lambda: MockPrinter(simulated_delay=p_cfg.get("simulated_delay", 1.5)),
        "file": lambda: FilePrinter(output_dir=output_dir / "printed"),
        "cups": lambda: CUPSPrinter(
            printer_name=p_cfg.get("printer_name", ""),
            media=p_cfg.get("media", "4x6in"),
            resolution_dpi=p_cfg.get("resolution_dpi", 300),
        ),
        "dnp": lambda: DNPPrinter(
            printer_name=p_cfg.get("printer_name", ""),
            size=p_cfg.get("size", "4x6"),
        ),
    }
    if kind not in registry:
        raise ValueError(f"Unknown printer: '{kind}'. Options: {list(registry)}")
    return registry[kind]()


def _build_motion_detector(hw: dict, config: dict) -> BaseMotionDetector:
    from hardware.motion import MockMotionDetector, OpenCVMotionDetector, KeyboardMotionDetector

    kind = hw.get("motion_detector", "mock").lower()
    m_cfg = config.get("motion_options", {})

    registry = {
        "mock": lambda: MockMotionDetector(
            delay_seconds=m_cfg.get("delay_seconds", 2.0),
            confidence=m_cfg.get("confidence", 0.95),
        ),
        "opencv": lambda: OpenCVMotionDetector(
            device_index=m_cfg.get("device_index", 0),
            sensitivity=m_cfg.get("sensitivity", 0.03),
            min_confidence=m_cfg.get("min_confidence", 0.6),
            cooldown=m_cfg.get("cooldown", 2.0),
        ),
        "keyboard": lambda: KeyboardMotionDetector(),
    }
    if kind not in registry:
        raise ValueError(f"Unknown motion detector: '{kind}'. Options: {list(registry)}")
    return registry[kind]()


def _build_enhancer(ai: dict, config: dict) -> BaseEnhancer:
    from ai.enhancers import (
        MockEnhancer,
        PillowAnimeEnhancer,
        StableDiffusionEnhancer,
        OpenAIAnimeEnhancer,
    )

    kind = ai.get("enhancer", "pillow_anime").lower()
    e_cfg = config.get("enhancer_options", {})

    registry = {
        # No-op — for tests
        "mock": lambda: MockEnhancer(),

        # Fast offline anime approximation (default)
        "pillow_anime": lambda: PillowAnimeEnhancer(
            color_levels=e_cfg.get("color_levels", 12),
            edge_strength=e_cfg.get("edge_strength", 0.75),
            saturation=e_cfg.get("saturation", 1.8),
            smoothing=e_cfg.get("smoothing", 5),
        ),

        # Local Stable Diffusion img2img (needs GPU + model weights)
        "stable_diffusion": lambda: StableDiffusionEnhancer(
            model_id=e_cfg.get("model_id", "andite/anything-v4.0"),
            prompt=e_cfg.get("prompt", StableDiffusionEnhancer.DEFAULT_PROMPT),
            negative_prompt=e_cfg.get("negative_prompt", StableDiffusionEnhancer.DEFAULT_NEGATIVE),
            strength=e_cfg.get("strength", 0.65),
            guidance_scale=e_cfg.get("guidance_scale", 9.0),
            steps=e_cfg.get("steps", 25),
            device=e_cfg.get("device", "cpu"),
        ),

        # OpenAI GPT-Image-1 (cloud, highest quality)
        "openai": lambda: OpenAIAnimeEnhancer(
            prompt=e_cfg.get("prompt", OpenAIAnimeEnhancer.DEFAULT_PROMPT),
            api_key=e_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            quality=e_cfg.get("quality", "high"),
            size=e_cfg.get("size", "1024x1024"),
        ),
    }
    if kind not in registry:
        raise ValueError(f"Unknown enhancer: '{kind}'. Options: {list(registry)}")
    return registry[kind]()


def _build_overlay(ai: dict, config: dict) -> BaseOverlayRenderer:
    from ai.overlays import NoOverlay, WeddingOverlayRenderer, MinimalOverlayRenderer

    kind = ai.get("overlay", "wedding").lower()
    registry = {
        "none": NoOverlay,
        "wedding": WeddingOverlayRenderer,
        "minimal": MinimalOverlayRenderer,
    }
    if kind not in registry:
        raise ValueError(f"Unknown overlay: '{kind}'. Options: {list(registry)}")
    return registry[kind]()


def _build_display(config: dict) -> BaseDisplay:
    from ui.display import LogDisplay, OpenCVDisplay

    kind = config.get("display", "log").lower()
    d_cfg = config.get("display_options", {})

    registry = {
        "log": lambda: LogDisplay(),
        "opencv": lambda: OpenCVDisplay(
            window_name=d_cfg.get("window_name", "Wedding Photo Booth"),
            fullscreen=d_cfg.get("fullscreen", False),
        ),
    }
    if kind not in registry:
        raise ValueError(f"Unknown display: '{kind}'. Options: {list(registry)}")
    return registry[kind]()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
