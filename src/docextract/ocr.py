"""OCR with the preprocessing settled in scripts/test_crop2.py.

Measured on 10 training documents, annotated-word recall at IoU>=0.3:
    raw image, psm 11, 2x            ...  30.5% (text found)
    + CLAHE + 3x upscale             ...  67.8% (text found)
    + crop, psm 11, IoU>=0.3         ...  73.4% (positional)

The crop uses a heavy Gaussian blur first: the woven-mat backgrounds in CORD
photographs are high-frequency, so blurring averages them to mid-grey while the
receipt stays bright. A plain Otsu threshold without the blur selects the whole
frame, because the mat's white strands are as bright as the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image


@dataclass(frozen=True)
class OcrConfig:
    tesseract_exe: str | None = None
    lang: str = "ind+eng"
    psm: int = 11
    scale: int = 3
    clahe_clip: float = 3.0
    clahe_grid: int = 8
    min_confidence: int = 30
    crop: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "OcrConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass(frozen=True)
class Word:
    text: str
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in original pixels
    confidence: int


def configure(cfg: OcrConfig) -> None:
    if cfg.tesseract_exe:
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_exe


def find_receipt(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the receipt. Falls back to the full frame if implausible."""
    h, w = gray.shape
    kernel_size = max(31, (min(h, w) // 20) | 1)
    smooth = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    _, mask = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h

    x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if not 0.05 < (cw * ch) / (w * h) < 0.95:
        return 0, 0, w, h

    pad = 8
    x, y = max(0, x - pad), max(0, y - pad)
    return x, y, min(w - x, cw + 2 * pad), min(h - y, ch + 2 * pad)


def enhance(gray: np.ndarray, cfg: OcrConfig) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
    )
    out = clahe.apply(gray)
    out = cv2.resize(out, None, fx=cfg.scale, fy=cfg.scale,
                     interpolation=cv2.INTER_CUBIC)
    return cv2.bilateralFilter(out, 5, 50, 50)


def is_noise(text: str) -> bool:
    """Background texture reads as stray punctuation; real fields never do.

    The woven-mat backgrounds in CORD photographs produce tokens like "|", "~"
    and "==". Dropping them is cheaper than raising min_confidence, which
    discards faint but genuine text: at min_confidence 50 the p95 sequence
    length fits in 512 tokens but label coverage falls from 65% to 59%.
    Anything containing a letter or digit is kept, so "-45" and "1 x" survive.
    """
    if len(text) == 1 and not text.isalnum():
        return True
    return not any(char.isalnum() for char in text)


def read(image: Image.Image, cfg: OcrConfig) -> list[Word]:
    """OCR one document, returning words with boxes in ORIGINAL image pixels."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

    if cfg.crop:
        ox, oy, cw, ch = find_receipt(gray)
        gray = gray[oy:oy + ch, ox:ox + cw]
    else:
        ox = oy = 0

    processed = enhance(gray, cfg)
    data = pytesseract.image_to_data(
        processed,
        lang=cfg.lang,
        config=f"--psm {cfg.psm}",
        output_type=pytesseract.Output.DICT,
    )

    words: list[Word] = []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        if not text:
            continue
        if is_noise(text):
            continue
        conf = int(data["conf"][i])
        if conf < cfg.min_confidence:
            continue
        x = data["left"][i] / cfg.scale + ox
        y = data["top"][i] / cfg.scale + oy
        w = data["width"][i] / cfg.scale
        h = data["height"][i] / cfg.scale
        words.append(Word(text=text, box=(x, y, x + w, y + h), confidence=conf))

    words.sort(key=lambda t: (t.box[1], t.box[0]))
    return words


def normalise_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> list[int]:
    """LayoutLMv3 expects boxes on a 0-1000 grid."""
    x1, y1, x2, y2 = box
    scaled = [
        int(1000 * x1 / width),
        int(1000 * y1 / height),
        int(1000 * x2 / width),
        int(1000 * y2 / height),
    ]
    return [min(1000, max(0, v)) for v in scaled]