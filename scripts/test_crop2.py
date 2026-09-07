"""Fix the crop (blur kills mat texture), then measure POSITIONAL recall.

This is the number that decides OCR pipeline vs OCR-free.
Run:  python scripts/test_crop2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from datasets import load_dataset
from PIL import Image

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

LANG = "ind+eng"
N_DOCS = 10
PSMS = [4, 6, 11]
SCALE = 3


def quad_to_box(quad: dict) -> list[float]:
    xs = [quad[f"x{i}"] for i in range(1, 5)]
    ys = [quad[f"y{i}"] for i in range(1, 5)]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def annotated_words(gt: dict) -> list[dict]:
    return [
        {"text": w["text"], "box": quad_to_box(w["quad"]), "category": line["category"]}
        for line in gt["valid_line"]
        for w in line["words"]
    ]


def find_receipt(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Heavy blur averages the woven mat to mid-grey; the receipt stays bright."""
    h, w = gray.shape
    k = max(31, (min(h, w) // 20) | 1)          # odd kernel, scales with image
    smooth = cv2.GaussianBlur(gray, (k, k), 0)

    _, mask = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h
    x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))

    frac = (cw * ch) / (w * h)
    if not (0.05 < frac < 0.95):                 # implausible -> keep full frame
        return 0, 0, w, h
    pad = 8
    x, y = max(0, x - pad), max(0, y - pad)
    return x, y, min(w - x, cw + 2 * pad), min(h - y, ch + 2 * pad)


def enhance(gray: np.ndarray) -> np.ndarray:
    out = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    out = cv2.resize(out, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    return cv2.bilateralFilter(out, 5, 50, 50)


def ocr(img: np.ndarray, psm: int, ox: int, oy: int) -> list[dict]:
    data = pytesseract.image_to_data(
        img, lang=LANG, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    words = []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        if not text or int(data["conf"][i]) < 0:
            continue
        x = data["left"][i] / SCALE + ox
        y = data["top"][i] / SCALE + oy
        w = data["width"][i] / SCALE
        h = data["height"][i] / SCALE
        words.append({"text": text, "box": [x, y, x + w, y + h]})
    return words


def main() -> None:
    ds = load_dataset("naver-clova-ix/cord-v2")
    thresholds = [0.5, 0.4, 0.3, 0.2]

    stats = {psm: {"text": 0, "tokens": 0, **{t: 0 for t in thresholds}}
             for psm in PSMS}
    total_ann = 0
    crop_fracs = []

    for idx in range(N_DOCS):
        s = ds["train"][idx]
        gt = json.loads(s["ground_truth"])
        anns = annotated_words(gt)
        total_ann += len(anns)

        gray_full = cv2.cvtColor(np.array(s["image"].convert("RGB")), cv2.COLOR_RGB2GRAY)
        x, y, w, h = find_receipt(gray_full)
        crop_fracs.append((w * h) / gray_full.size)
        proc = enhance(gray_full[y:y + h, x:x + w])

        if idx == 0:
            Image.fromarray(proc).save("crop2_after.png")
            print(f"doc 0: frame {gray_full.shape[1]}x{gray_full.shape[0]} "
                  f"-> crop {w}x{h} at ({x},{y})")

        for psm in PSMS:
            toks = ocr(proc, psm, x, y)
            texts = [t["text"] for t in toks]
            stats[psm]["tokens"] += len(toks)
            for ann in anns:
                if ann["text"] in texts:
                    stats[psm]["text"] += 1
                best = max((iou(ann["box"], t["box"]) for t in toks), default=0.0)
                for t in thresholds:
                    if best >= t:
                        stats[psm][t] += 1

    print(f"\nmean crop = {100 * np.mean(crop_fracs):.0f}% of frame "
          f"(was 100% = crop failed)")
    print(f"{N_DOCS} docs, {total_ann} annotated words\n")

    head = f"{'psm':>5}{'text':>8}" + "".join(f"{'IoU' + str(t):>9}" for t in thresholds)
    print(head + f"{'tok/doc':>10}")
    print("-" * len(head + "   tok/doc"))
    for psm in PSMS:
        st = stats[psm]
        row = f"{psm:>5}{100 * st['text'] / total_ann:>7.1f}%"
        for t in thresholds:
            row += f"{100 * st[t] / total_ann:>8.1f}%"
        row += f"{st['tokens'] / N_DOCS:>10.1f}"
        print(row)


if __name__ == "__main__":
    main()