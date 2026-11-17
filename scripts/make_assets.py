"""Render README/social-preview images (run locally when assets change).

Not part of the package or CI — Pillow is only needed here:
    uvx --with pillow python scripts/make_assets.py

Outputs:
    docs/assets/terminal-predict.png   — README terminal screenshot
    .github/assets/social-preview.png  — GitHub link-unfurl image (1280x640)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent

# GitHub-dark palette
BG = (13, 17, 23)
FG = (201, 209, 217)
GREEN = (63, 185, 80)
CYAN = (121, 192, 255)
YELLOW = (227, 179, 65)
RED = (248, 81, 73)
GRAY = (110, 118, 129)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def _rounded_panel(draw: ImageDraw.ImageDraw, xy: tuple, radius: int, fill) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def terminal_png(out: Path) -> None:
    mono = _font("consola.ttf", 22)
    bold = _font("consolab.ttf", 22)
    pad, line_h = 34, 34
    lines: list[tuple[str, object, bool | None]] = [
        # (text, color, bold)  — real output of `vv download-model` / `vv predict`
        ("$ vv download-model", GREEN, True),
        ("model saved to    : .../voiceveritas-prosody-lgbm-v0.1.1.joblib", GRAY, False),
        ("", FG, False),
        ("$ vv predict model.joblib examples/example_read.wav", GREEN, True),
        ("model            : prosody_lgbm", FG, False),
        ("duration         : 12.0s", FG, False),
        ("P(read)          : 0.800", YELLOW, True),
        ("verdict@0.5      : READ", FG, False),
        ("estimated SNR    : 60.0 dB", FG, False),
        ("top contributors :", FG, False),
        ("  speechseg_mean           +0.585", CYAN, False),
        ("  spec_flatness_mean       +0.577", CYAN, False),
        ("  zcr_std                  -0.574", CYAN, False),
        ("  zcr_mean                 -0.417", CYAN, False),
    ]
    w, h = 900, pad * 2 + 18 + 18 + line_h * len(lines)
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    # window chrome
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse((pad + i * 26, 18, pad + 14 + i * 26, 32), fill=c)
    y = 50
    for text, color, is_bold in lines:
        draw.text((pad, y), text, font=bold if is_bold else mono, fill=color)
        y += line_h
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out} ({w}x{h})")


def social_png(out: Path) -> None:
    w, h = 1280, 640
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    # subtle vertical gradient
    for y in range(h):
        t = y / h
        draw.line([(0, y), (w, y)], fill=(13 + int(12 * t), 17 + int(14 * t), 23 + int(20 * t)))
    # waveform band behind the title (fixed pattern, no rng needed)
    import math

    bars, x0, cy = 90, 90, 455
    for i in range(bars):
        t = i / (bars - 1)
        amp = 18 + 52 * abs(math.sin(i * 0.55)) * (1 - t * 0.3)
        x = x0 + int(t * (w - 2 * x0))
        r, g, b = int(45 + 60 * t), int(212 - 90 * t), int(191 - 40 * t)
        draw.rounded_rectangle((x, cy - amp, x + 8, cy + amp), radius=4, fill=(r, g, b))
    # glow: blurred copy composited under text
    glow = img.filter(ImageFilter.GaussianBlur(18))
    img = Image.composite(glow, img, Image.new("L", (w, h), 40))
    draw = ImageDraw.Draw(img)

    title = _font("seguisb.ttf", 96)
    small = _font("consola.ttf", 30)
    draw.text((88, 110), "VoiceVeritas", font=title, fill=(240, 246, 252))
    # tagline: shrink-to-fit so it can never run off the canvas
    tagline = "Detect scripted vs. spontaneous interview speech"
    size = 40
    while size > 24 and draw.textlength(tagline, font=_font("segoeui.ttf", size)) > w - 180:
        size -= 2
    draw.text((92, 240), tagline, font=_font("segoeui.ttf", size), fill=FG)
    # chips
    chip_font = _font("segoeui.ttf", 28)
    x = 92
    for label in ["audio-only", "P(read) in ~250 ms", "prosody + wav2vec2", "Apache-2.0"]:
        tw = draw.textlength(label, font=chip_font)
        draw.rounded_rectangle((x, 330, x + tw + 36, 386), radius=16, outline=(90, 100, 115))
        draw.text((x + 18, 340), label, font=chip_font, fill=(160, 175, 190))
        x += int(tw) + 56
    draw.text((92, 584), "github.com/StanleyChanH/VoiceVeritas", font=small, fill=GRAY)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out} ({w}x{h})")


if __name__ == "__main__":
    terminal_png(ROOT / "docs" / "assets" / "terminal-predict.png")
    social_png(ROOT / ".github" / "assets" / "social-preview.png")
