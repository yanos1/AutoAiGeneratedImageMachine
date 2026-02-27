#!/usr/bin/env python3
"""
main.py
-------
Entry point for the Wedding Photo Automation App.

Usage:
    python main.py                    # uses config.yaml
    python main.py --mode mock        # all mocks, no hardware
    python main.py --mode dev         # webcam + keyboard trigger + file printer
    python main.py --mode production  # real hardware (configure config.yaml)
    python main.py --config my.yaml   # custom config file
    python main.py --once             # run a single session and exit
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import load_config, build_components
from utils.logger import setup_logging
from core.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Mode presets — override config for quick startup
# ---------------------------------------------------------------------------

MODE_OVERRIDES = {
    "mock": {
        "hardware": {"camera": "mock", "printer": "mock", "motion_detector": "mock"},
        "ai": {"enhancer": "mock", "overlay": "none"},
        "display": "log",
        "pipeline": {"countdown_seconds": 1, "cooldown_seconds": 1},
    },
    "dev": {
        "hardware": {"camera": "webcam", "printer": "file", "motion_detector": "keyboard"},
        "ai": {"enhancer": "pillow_anime", "overlay": "wedding"},
        "display": "opencv",
    },
    "production": {
        "hardware": {"camera": "webcam", "printer": "cups", "motion_detector": "opencv"},
        "ai": {"enhancer": "pillow", "overlay": "wedding"},
        "display": "opencv",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Wedding Photo Automation App")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", choices=["mock", "dev", "production"],
                        help="Quick-start mode (overrides config)")
    parser.add_argument("--once", action="store_true",
                        help="Run a single session then exit")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.mode and args.mode in MODE_OVERRIDES:
        config = _deep_merge(config, MODE_OVERRIDES[args.mode])

    # Setup logging
    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file"),
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Wedding Photo App (mode=%s)", args.mode or "config")

    # Build components
    components = build_components(config)
    logger.info("Components: camera=%s printer=%s detector=%s enhancer=%s overlay=%s display=%s",
                components["camera"].name,
                components["printer"].name,
                components["motion_detector"].name,
                components["enhancer"].name,
                components["overlay_renderer"].name,
                components["display"].name)

    # Build pipeline
    pipeline_config = config.get("pipeline", {})
    pipeline_config["overlay_context"] = config.get("overlay_context", {})

    pipeline = Pipeline(
        camera=components["camera"],
        printer=components["printer"],
        motion_detector=components["motion_detector"],
        enhancer=components["enhancer"],
        overlay_renderer=components["overlay_renderer"],
        display=components["display"],
        output_dir=Path(config.get("output_dir", "z_output")),
        config=pipeline_config,
    )

    # Run
    if args.once:
        session = pipeline.run_single_session()
        logger.info("Session result: %s", session.summary())
        sys.exit(0 if session.status.name == "COMPLETE" else 1)
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
