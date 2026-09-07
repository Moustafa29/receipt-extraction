"""Does cropping the receipt out of the background fix OCR detection?

Run:  python scripts/test_crop.py
Writes crop_before.png / crop_after.png for eyeballing.
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


def quad_to_box(quad: dict) -> list[float]:
    xs = [quad[f"x{i}"] for i in range(1, 5)]
    ys = [quad[f"y{i}"] for i in range(1, 5)]
    return [min(xs), min(ys), max(xs), max(ys)]


def annotated_words(gt: dict) -> list[dict]:
    return [
        {"text": w["text"], "box": quad_to_box(w["quad"]), "category": line["category"]}
        for line in gt["valid_line"]
        for w in line["words"]
    ]


def find_receipt(bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of the largest bright blob = the receipt."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, bgr.shape[1], bgr.shape[0]
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)

    # Reject implausible crops (tiny or nearly the whole frame)
    frac = (w * h) / (bgr.shape[0] * bgr.shape[1])
    if frac < 0.05:
        return 0, 0, bgr.shape[1], bgr.shape[0]
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(bgr.shape[1] - x, w + 2 * pad)
    h = min(bgr.shape[0] - y, h + 2 * pad)
    return x, y, w, h


def enhance(gray: np.ndarray, scale: int = 3) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    out = clahe.apply(gray)
    out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out = cv2.bilateralFilter(out, 5, 50, 50)
    return out


def ocr(img: np.ndarray, psm: int, scale: float, ox: int, oy: int) -> list[dict]:
    data = pytesseract.image_to_data(
        img, lang=LANG, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    words = []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        if not text or int(data["conf"][i]) < 0:
            continue
        x = data["left"][i] / scale + ox
        y = data["top"][i] / scale + oy
        w = data["width"][i] / scale
        h = data["height"][i] / scale
        words.append({"text": text, "box": [x, y, x + w, y + h]})
    return words


def main() -> None:
    ds = load_dataset("naver-clova-ix/cord-v2")
    scale = 3

    results = {psm: {"found": 0, "tokens": 0} for psm in PSMS}
    baseline_found = 0
    total_ann = 0

    for idx in range(N_DOCS):
        s = ds["train"][idx]
        gt = json.loads(s["ground_truth"])
        anns = annotated_words(gt)
        total_ann += len(anns)
        ann_texts = [a["text"] for a in anns]

        bgr = cv2.cvtColor(np.array(s["image"].convert("RGB")), cv2.COLOR_RGB2BGR)
        x, y, w, h = find_receipt(bgr)
        gray = cv2.cvtColor(bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        proc = enhance(gray, scale)

        if idx == 0:
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save("crop_before.png")
            Image.fromarray(proc).save("crop_after.png")
            print(f"doc 0: frame {bgr.shape[1]}x{bgr.shape[0]} -> crop {w}x{h} "
                  f"at ({x},{y}), processed {proc.shape[1]}x{proc.shape[0]}")

        # Baseline: raw frame, no crop, psm 11, 2x
        g0 = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2)
        base = ocr(g0, 11, 2, 0, 0)
        base_texts = [t["text"] for t in base]
        baseline_found += sum(1 for t in ann_texts if t in base_texts)

        for psm in PSMS:
            toks = ocr(proc, psm, scale, x, y)
            texts = [t["text"] for t in toks]
            results[psm]["found"] += sum(1 for t in ann_texts if t in texts)
            results[psm]["tokens"] += len(toks)

    print(f"\n{N_DOCS} docs, {total_ann} annotated words")
    print(f"\nbaseline (no crop, psm 11, 2x):  "
          f"{100 * baseline_found / total_ann:.1f}% text found")
    print("\ncropped + CLAHE + 3x:")
    print(f"  {'psm':>5}{'text found':>14}{'tok/doc':>10}")
    for psm in PSMS:
        r = results[psm]
        print(f"  {psm:>5}{100 * r['found'] / total_ann:>13.1f}%"
              f"{r['tokens'] / N_DOCS:>10.1f}")


if __name__ == "__main__":
    main()