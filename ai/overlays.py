"""
ai/overlays.py
--------------
Overlay / branding renderer implementations.

  NoOverlay             — returns image unchanged
  WeddingOverlayRenderer— elegant frame + couple names + date
  MinimalOverlayRenderer— just a small date/event watermark
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from core.interfaces import BaseOverlayRenderer

logger = logging.getLogger(__name__)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Try to load a nice font; fall back to PIL default if unavailable."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# No overlay
# ---------------------------------------------------------------------------

class NoOverlay(BaseOverlayRenderer):
    def apply(self, image: Image.Image, context: dict) -> Image.Image:
        return image


# ---------------------------------------------------------------------------
# Wedding overlay — elegant border + names + date
# ---------------------------------------------------------------------------

class WeddingOverlayRenderer(BaseOverlayRenderer):
    """
    Adds a beautiful wedding-themed overlay:
    - Subtle vignette
    - Decorative border
    - Couple's names (large, centered, bottom)
    - Event date below names
    - Optional event name / venue

    context keys:
        couple_names : str  e.g. "Sarah & James"
        date         : str  e.g. "June 14, 2025"
        event_name   : str  e.g. "Wedding Reception" (optional)
        border_color : tuple (R,G,B) default warm gold
        text_color   : tuple (R,G,B) default white
    """

    def apply(self, image: Image.Image, context: dict) -> Image.Image:
        img = image.convert("RGBA")
        w, h = img.size

        # --- Vignette layer ---
        vignette = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        vdraw = ImageDraw.Draw(vignette)
        steps = 60
        for i in range(steps):
            alpha = int(120 * (i / steps) ** 2)
            margin = int((steps - i) * (min(w, h) / (2 * steps)))
            vdraw.rectangle(
                [margin, margin, w - margin, h - margin],
                outline=(0, 0, 0, alpha),
                width=int(min(w, h) / (2 * steps)) + 1,
            )
        img = Image.alpha_composite(img, vignette)

        # --- Decorative border ---
        border_color = context.get("border_color", (212, 175, 55))  # gold
        bdraw = ImageDraw.Draw(img)
        bw = max(4, w // 120)
        margin = max(20, w // 50)
        bdraw.rectangle(
            [margin, margin, w - margin, h - margin],
            outline=border_color + (200,),
            width=bw,
        )
        # Inner thin border
        inner = margin + bw + 6
        bdraw.rectangle(
            [inner, inner, w - inner, h - inner],
            outline=border_color + (120,),
            width=max(1, bw // 2),
        )

        # --- Text panel at bottom ---
        text_color = context.get("text_color", (255, 255, 255))
        couple_names = context.get("couple_names", "")
        date_str = context.get("date", "")
        event_name = context.get("event_name", "")

        draw = ImageDraw.Draw(img)
        panel_h = int(h * 0.18)
        y_start = h - panel_h

        # Semi-transparent dark gradient at bottom
        for y in range(y_start, h):
            alpha = int(200 * ((y - y_start) / panel_h))
            draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))

        cy = y_start + int(panel_h * 0.1)

        if couple_names:
            font_size = max(24, w // 18)
            font = _load_font(font_size, bold=True)
            bbox = draw.textbbox((0, 0), couple_names, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(
                ((w - tw) // 2, cy),
                couple_names,
                font=font,
                fill=(*text_color, 240),
            )
            cy += font_size + int(panel_h * 0.08)

        if date_str:
            font_size = max(16, w // 30)
            font = _load_font(font_size)
            bbox = draw.textbbox((0, 0), date_str, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(
                ((w - tw) // 2, cy),
                date_str,
                font=font,
                fill=(*text_color, 200),
            )
            cy += font_size + int(panel_h * 0.06)

        if event_name:
            font_size = max(12, w // 45)
            font = _load_font(font_size)
            bbox = draw.textbbox((0, 0), event_name, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(
                ((w - tw) // 2, cy),
                event_name,
                font=font,
                fill=(*text_color, 160),
            )

        result = img.convert("RGB")
        logger.debug("WeddingOverlayRenderer: applied overlay for '%s'", couple_names)
        return result


# ---------------------------------------------------------------------------
# Minimal overlay — just a watermark
# ---------------------------------------------------------------------------

class MinimalOverlayRenderer(BaseOverlayRenderer):
    """Small watermark in bottom-right corner."""

    def apply(self, image: Image.Image, context: dict) -> Image.Image:
        img = image.convert("RGBA")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        text = context.get("couple_names") or context.get("event_name") or ""
        date_str = context.get("date", "")
        watermark = f"{text}  {date_str}".strip()
        if not watermark:
            return img.convert("RGB")

        font_size = max(12, w // 60)
        font = _load_font(font_size)
        bbox = draw.textbbox((0, 0), watermark, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x = w - tw - 20
        y = h - th - 20
        draw.text((x + 1, y + 1), watermark, font=font, fill=(0, 0, 0, 120))
        draw.text((x, y), watermark, font=font, fill=(255, 255, 255, 200))

        return img.convert("RGB")
