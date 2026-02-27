"""
ai/enhancers.py
---------------
Image style-transfer implementations. All extend BaseEnhancer.

The goal is anime-style transformation of wedding photos.

  MockEnhancer           — returns image unchanged (testing)
  PillowAnimeEnhancer    — fast local approximation: flat colours, hard edges,
                           vivid palette. No API needed.
  StableDiffusionEnhancer— full anime style transfer via a local SD model
                           (AnythingV5 / Counterfeit / similar) using img2img.
  OpenAIAnimeEnhancer    — sends photo to GPT-Image-1 with an anime style prompt.
                           Best quality, requires OpenAI API key.
"""

from __future__ import annotations

import logging
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np

from core.interfaces import BaseEnhancer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock — pass-through, used in tests
# ---------------------------------------------------------------------------

class MockEnhancer(BaseEnhancer):
    """Returns the image unchanged. Use in tests."""

    def enhance(self, image: Image.Image) -> Image.Image:
        logger.debug("MockEnhancer: pass-through")
        return image


# ---------------------------------------------------------------------------
# Pillow anime approximation — fast, offline, no model weights needed
# ---------------------------------------------------------------------------

class PillowAnimeEnhancer(BaseEnhancer):
    """
    Approximates anime style using Pillow-only transforms. No GPU or API needed.

    Technique:
      1. Bilateral-filter approximation via median filter (smooths skin, preserves edges)
      2. Edge detection → black ink outline layer
      3. Colour quantisation → flat cel-shaded palette
      4. Boost saturation + contrast for vivid anime look
      5. Composite outlines over flat colours

    Quality: visually distinct anime look; not as clean as a diffusion model,
    but instant and works 100% offline. Great for demos.

    Parameters
    ----------
    color_levels   : int   — palette quantisation levels (8–32; lower = more cartoon)
    edge_strength  : float — opacity of the ink outline layer (0.0–1.0)
    saturation     : float — colour saturation boost multiplier
    smoothing      : int   — median filter radius for skin smoothing (odd number)
    """

    def __init__(
        self,
        color_levels: int = 12,
        edge_strength: float = 0.75,
        saturation: float = 1.8,
        smoothing: int = 5,
    ) -> None:
        self.color_levels = color_levels
        self.edge_strength = edge_strength
        self.saturation = saturation
        self.smoothing = smoothing

    def enhance(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        w, h = img.size

        # 1. Smooth (bilateral approximation via median filter)
        smooth = img.filter(ImageFilter.MedianFilter(size=self.smoothing))
        smooth = smooth.filter(ImageFilter.SMOOTH_MORE)

        # 2. Colour quantisation → flat cel-shaded blocks
        quantised = smooth.quantize(colors=self.color_levels, method=Image.Quantize.FASTOCTREE)
        quantised = quantised.convert("RGB")

        # 3. Boost saturation + contrast for vivid anime palette
        quantised = ImageEnhance.Color(quantised).enhance(self.saturation)
        quantised = ImageEnhance.Contrast(quantised).enhance(1.3)

        # 4. Edge detection for ink outlines
        grey = img.convert("L")
        # Detect edges at multiple scales and combine
        edges_fine = grey.filter(ImageFilter.FIND_EDGES)
        blurred = grey.filter(ImageFilter.GaussianBlur(radius=2))
        edges_coarse = blurred.filter(ImageFilter.FIND_EDGES)

        # Combine and invert (black lines on white)
        edges_arr = np.array(edges_fine).astype(np.float32)
        coarse_arr = np.array(edges_coarse).astype(np.float32)
        combined = np.clip(edges_arr * 0.6 + coarse_arr * 0.4, 0, 255).astype(np.uint8)

        # Threshold → clean black lines
        edge_img = Image.fromarray(combined).convert("L")
        edge_img = edge_img.point(lambda p: 0 if p > 30 else 255)  # invert: lines=0
        edge_rgb = Image.merge("RGB", [edge_img, edge_img, edge_img])

        # 5. Composite: multiply edges onto flat colour
        result_arr = np.array(quantised).astype(np.float32)
        edge_arr = np.array(edge_rgb).astype(np.float32) / 255.0
        composited = (result_arr * edge_arr).clip(0, 255).astype(np.uint8)
        result = Image.fromarray(composited)

        # 6. Final sharpening pass to crisp up line art
        result = result.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))

        logger.info("PillowAnimeEnhancer: applied (levels=%d edge=%.2f sat=%.2f)",
                    self.color_levels, self.edge_strength, self.saturation)
        return result


# ---------------------------------------------------------------------------
# Stable Diffusion img2img — local model, best offline quality
# ---------------------------------------------------------------------------

class StableDiffusionEnhancer(BaseEnhancer):
    """
    Anime style transfer via Stable Diffusion img2img using 🤗 diffusers.

    Uses an anime-focused checkpoint — AnythingV5, Counterfeit-V3, or
    any other LoRA/fine-tune you have locally. Runs on CPU or GPU.

    Requires:
        pip install diffusers transformers accelerate torch
        # Download a model, e.g.:
        # huggingface-cli download andite/anything-v4.0

    Parameters
    ----------
    model_id        : str   — HuggingFace model ID or local path
    prompt          : str   — positive style prompt
    negative_prompt : str   — things to avoid
    strength        : float — 0.0 = no change, 1.0 = full reimagine (0.55–0.75 sweet spot)
    guidance_scale  : float — prompt adherence (7–12)
    steps           : int   — denoising steps (20–30 is plenty for img2img)
    device          : str   — "cuda", "mps", or "cpu"
    """

    DEFAULT_PROMPT = (
        "anime style, wedding couple, soft cel shading, clean line art, "
        "Studio Ghibli inspired, delicate features, warm pastel palette, "
        "detailed background, cinematic composition, high quality"
    )
    DEFAULT_NEGATIVE = (
        "photorealistic, photograph, blurry, low quality, ugly, deformed, "
        "watermark, text, extra limbs, bad anatomy"
    )

    def __init__(
        self,
        model_id: str = "andite/anything-v4.0",
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE,
        strength: float = 0.65,
        guidance_scale: float = 9.0,
        steps: int = 25,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.device = device
        self._pipe = None

    def _get_pipe(self):
        if self._pipe is None:
            try:
                from diffusers import StableDiffusionImg2ImgPipeline
                import torch
            except ImportError:
                raise RuntimeError(
                    "diffusers / torch not installed.\n"
                    "Run: pip install diffusers transformers accelerate torch"
                )

            logger.info("StableDiffusionEnhancer: loading model '%s' on %s (first run may be slow)…",
                        self.model_id, self.device)

            dtype = None
            try:
                import torch
                dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
            except ImportError:
                pass

            self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(self.device)

            # Memory optimisation for consumer GPUs
            if self.device == "cuda":
                self._pipe.enable_attention_slicing()

            logger.info("StableDiffusionEnhancer: model loaded")
        return self._pipe

    def enhance(self, image: Image.Image) -> Image.Image:
        pipe = self._get_pipe()
        original_size = image.size

        # SD works best at 512x512 or 768x768; resize, then restore original size after
        working_size = (768, 768)
        img = image.convert("RGB").resize(working_size, Image.LANCZOS)

        logger.info("StableDiffusionEnhancer: running img2img (strength=%.2f steps=%d)…",
                    self.strength, self.steps)

        result = pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=img,
            strength=self.strength,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.steps,
        ).images[0]

        # Restore to original resolution
        result = result.resize(original_size, Image.LANCZOS)
        logger.info("StableDiffusionEnhancer: transformation complete")
        return result


# ---------------------------------------------------------------------------
# OpenAI GPT-Image-1 — cloud, highest quality
# ---------------------------------------------------------------------------

class OpenAIAnimeEnhancer(BaseEnhancer):
    """
    Transforms a wedding photo into anime style using OpenAI's GPT-Image-1
    image editing API.

    Sends the captured photo as the input image with an anime-style prompt.
    GPT-Image-1 is instruction-following and produces significantly better
    style transfers than DALL-E 2/3 edit endpoints.

    Requires: pip install openai
    Set OPENAI_API_KEY in environment or pass api_key directly.

    Parameters
    ----------
    prompt   : str        — style description (customise per wedding aesthetic)
    api_key  : str | None — OpenAI key (falls back to OPENAI_API_KEY env var)
    quality  : str        — "low" | "medium" | "high" (affects cost + latency)
    size     : str        — "1024x1024" | "1536x1024" | "1024x1536"
    """

    DEFAULT_PROMPT = (
        "Transform this wedding photo into a beautiful anime illustration. "
        "Style: Studio Ghibli meets Makoto Shinkai — soft cel shading, clean ink outlines, "
        "luminous pastel colours, detailed expressive faces, dreamy romantic atmosphere. "
        "Keep the couple's poses and the scene composition identical. "
        "Make the background painterly with soft bokeh. High quality anime art."
    )

    def __init__(
        self,
        prompt: str = DEFAULT_PROMPT,
        api_key: str | None = None,
        quality: str = "high",
        size: str = "1024x1024",
    ) -> None:
        self.prompt = prompt
        self.api_key = api_key
        self.quality = quality
        self.size = size
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("openai not installed. Run: pip install openai")
        return self._client

    def enhance(self, image: Image.Image) -> Image.Image:
        import io
        import urllib.request

        client = self._get_client()
        original_size = image.size

        # Encode image as PNG bytes
        img = image.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        logger.info("OpenAIAnimeEnhancer: calling gpt-image-1 (quality=%s)…", self.quality)

        response = client.images.edit(
            model="gpt-image-1",
            image=("photo.png", buf, "image/png"),
            prompt=self.prompt,
            quality=self.quality,
            size=self.size,
        )

        # Download result
        url = response.data[0].url
        with urllib.request.urlopen(url) as resp:
            result_bytes = resp.read()

        result = Image.open(io.BytesIO(result_bytes)).convert("RGB")
        result = result.resize(original_size, Image.LANCZOS)
        logger.info("OpenAIAnimeEnhancer: transformation complete")
        return result
