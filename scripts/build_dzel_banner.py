"""Build a subtle looping D.Z.E.L. banner from supplied source artwork."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "assets" / "dzel-block-source.png"
OUTPUT_PATH = ROOT / "assets" / "dzel-block.gif"
CONFIG_PATH = ROOT / "config" / "windows.json"

OUTPUT_WIDTH = 1200
FPS = 9
DURATION = 7
SNOW_DENSITY = 0.00012
FLICKER_PROBABILITY = 0.035
NOISE_STRENGTH = 0.0
SEED = 24017

WINDOW_COLOR = (190, 226, 232)
SNOW_COLOR = (207, 222, 225)


def load_windows(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Window configuration must be a JSON object.")

    windows: dict[str, list[dict[str, Any]]] = {}
    for group in ("d", "z", "e", "l", "ambient"):
        entries = config.get(group, [])
        if not isinstance(entries, list):
            raise ValueError(f"The '{group}' window group must be a list.")
        for entry in entries:
            validate_window(group, entry)
        windows[group] = entries
    return windows


def validate_window(group: str, window: Any) -> None:
    if not isinstance(window, dict):
        raise ValueError(f"Each '{group}' window must be an object.")

    for key in ("x", "y", "width", "height"):
        if key not in window or not isinstance(window[key], (int, float)):
            raise ValueError(f"Each '{group}' window needs numeric {key} values.")
    if window["width"] <= 0 or window["height"] <= 0:
        raise ValueError(f"Each '{group}' window needs positive dimensions.")

    for key, default in (
        ("probability", FLICKER_PROBABILITY),
        ("brightness", 0.75),
        ("flicker_strength", 0.25),
    ):
        value = window.get(key, default)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Window {key} must be between 0 and 1.")


def snow_particles(width: int, height: int) -> list[tuple[float, float, int, float]]:
    rng = random.Random(SEED + 1)
    count = max(1, round(width * height * SNOW_DENSITY))
    return [
        (
            rng.uniform(0, width),
            rng.uniform(0, height),
            rng.choice((1, 1, 1, 2)),
            rng.choice((1, 2)),
        )
        for _ in range(count)
    ]


def add_windows(
    frame: Image.Image,
    windows: dict[str, list[dict[str, Any]]],
    scale_x: float,
    scale_y: float,
    rng: random.Random,
) -> None:
    for group, entries in windows.items():
        is_letter = group in {"d", "z", "e", "l"}
        for window in entries:
            brightness = float(window.get("brightness", 0.75))
            probability = float(window.get("probability", FLICKER_PROBABILITY))
            strength = float(window.get("flicker_strength", 0.25))
            max_loss = min(strength, 0.28 if is_letter else 0.8)
            loss = 0.0

            if rng.random() < probability:
                loss = rng.uniform(max_loss * 0.35, max_loss)

            x = round(float(window["x"]) * scale_x)
            y = round(float(window["y"]) * scale_y)
            right = round((float(window["x"]) + float(window["width"])) * scale_x)
            bottom = round((float(window["y"]) + float(window["height"])) * scale_y)
            bounds = (x, y, max(x + 1, right), max(y + 1, bottom))
            region = frame.crop(bounds)
            if loss:
                region = ImageEnhance.Brightness(region).enhance(1.0 - loss)

            tint = Image.new("RGB", region.size, WINDOW_COLOR)
            frame.paste(Image.blend(region, tint, 0.08 * brightness), bounds)


def add_snow(
    frame: Image.Image,
    particles: list[tuple[float, float, int, float]],
    frame_index: int,
    frame_count: int,
) -> None:
    width, height = frame.size
    draw = ImageDraw.Draw(frame, "RGBA")
    phase = 2 * math.pi * frame_index / frame_count

    for x_start, y_start, radius, cycles in particles:
        x = (x_start + 1.5 * math.sin(phase + y_start)) % width
        y = (y_start + cycles * height * frame_index / frame_count) % height
        alpha = 90 if radius == 1 else 125
        draw.ellipse(
            (round(x), round(y), round(x) + radius, round(y) + radius),
            fill=(*SNOW_COLOR, alpha),
        )


def add_analog_noise(frame: Image.Image, frame_index: int) -> Image.Image:
    if NOISE_STRENGTH <= 0:
        return frame

    rng = np.random.default_rng(SEED + 1000 + frame_index)
    noise = rng.normal(0, NOISE_STRENGTH, (frame.height, frame.width, 1))
    pixels = np.asarray(frame, dtype=np.int16) + noise.astype(np.int16)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def build_banner(source: Path, output: Path, config_path: Path) -> None:
    windows = load_windows(config_path)
    with Image.open(source) as source_image:
        original = source_image.convert("RGB")

    original_width, original_height = original.size
    if original_width == 0 or original_height == 0:
        raise ValueError("Source artwork must have non-zero dimensions.")

    output_height = round(OUTPUT_WIDTH * original_height / original_width)
    base = original.resize((OUTPUT_WIDTH, output_height), Image.Resampling.LANCZOS)
    scale_x = OUTPUT_WIDTH / original_width
    scale_y = output_height / original_height
    frame_count = FPS * DURATION
    particles = snow_particles(OUTPUT_WIDTH, output_height)
    frames: list[Image.Image] = []

    for frame_index in range(frame_count):
        # Match the final frame to the first so the loop closes without a jump.
        phase_index = 0 if frame_index == frame_count - 1 else frame_index
        frame = base.copy()
        add_windows(
            frame,
            windows,
            scale_x,
            scale_y,
            random.Random(SEED + phase_index),
        )
        add_snow(frame, particles, phase_index, frame_count)

        breathing = 1.0 + 0.012 * math.sin(2 * math.pi * phase_index / frame_count)
        if breathing != 1.0:
            frame = ImageEnhance.Brightness(frame).enhance(breathing)
        frame = add_analog_noise(frame, phase_index)
        frames.append(frame)

    palette = frames[0].quantize(colors=128, method=Image.Quantize.FASTOCTREE)
    indexed_frames = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    indexed_frames[0].save(
        output,
        save_all=True,
        append_images=indexed_frames[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"Built {output} ({OUTPUT_WIDTH}x{output_height}, {frame_count} frames).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        print(
            "Source artwork is missing. Place the architectural source image at "
            f"{args.source} and run this script again.",
            file=sys.stderr,
        )
        return 1
    if not args.config.is_file():
        print(f"Window configuration is missing: {args.config}", file=sys.stderr)
        return 1

    try:
        build_banner(args.source, args.output, args.config)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not build the D.Z.E.L. banner: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())