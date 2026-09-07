"""Is the low match rate a detection problem or a matching-criterion problem?

Run:  python scripts/diagnose_matching.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytesseract
from datasets import load_dataset
from PIL import Image, ImageOps

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

LANG = "ind+eng"
PSM = 11
SCALE = 2
N_DOCS = 10


def quad_to_box(quad: dict) -> list[float]:
    xs = [quad[f"x{i}"] for i in range(1, 5)]
    ys = [quad[f"y{i}"] for i in range(1, 5)]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_inside(inner, outer) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def annotated_words(gt: dict) -> list[dict]:
    return [
        {"text": w["text"], "box": quad_to_box(w["quad"]), "category": line["category"]}
        for line in gt["valid_line"]
        for w in line["words"]
    ]


def ocr_words(img: Image.Image) -> list[dict]:
    g = ImageOps.grayscale(img)
    g = g.resize((g.width * SCALE, g.height * SCALE), Image.LANCZOS)
    data = pytesseract.image_to_data(
        g, lang=LANG, config=f"--psm {PSM}", output_type=pytesseract.Output.DICT
    )
    words = []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        if not text or int(data["conf"][i]) < 0:
            continue
        x, y = data["left"][i] / SCALE, data["top"][i] / SCALE
        w, h = data["width"][i] / SCALE, data["height"][i] / SCALE
        words.append({"text": text, "box": [x, y, x + w, y + h],
                      "conf": int(data["conf"][i])})
    return words


def main() -> None:
    ds = load_dataset("naver-clova-ix/cord-v2")

    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
    hits = {t: 0 for t in thresholds}
    hits_center = 0
    hits_text = 0
    total_ann = 0
    best_ious = []
    ann_dims, ocr_dims = [], []

    for idx in range(N_DOCS):
        s = ds["train"][idx]
        gt = json.loads(s["ground_truth"])
        anns = annotated_words(gt)
        ocr = ocr_words(s["image"])

        ocr_texts = [t["text"] for t in ocr]
        for t in ocr:
            ocr_dims.append((t["box"][2] - t["box"][0], t["box"][3] - t["box"][1]))

        for ann in anns:
            ann_dims.append((ann["box"][2] - ann["box"][0],
                             ann["box"][3] - ann["box"][1]))
            best = max((iou(ann["box"], t["box"]) for t in ocr), default=0.0)
            best_ious.append(best)
            for t in thresholds:
                if best >= t:
                    hits[t] += 1
            if any(center_inside(t["box"], ann["box"]) or
                   center_inside(ann["box"], t["box"]) for t in ocr):
                hits_center += 1
            if ann["text"] in ocr_texts:
                hits_text += 1

        total_ann += len(anns)

    print(f"{N_DOCS} docs, {total_ann} annotated words\n")
    print("recall by matching criterion:")
    for t in thresholds:
        print(f"  IoU >= {t:.1f}        {100 * hits[t] / total_ann:5.1f}%")
    print(f"  center-in-box     {100 * hits_center / total_ann:5.1f}%")
    print(f"  exact text found  {100 * hits_text / total_ann:5.1f}%   "
          "(ignores position entirely)")

    arr = np.array(best_ious)
    print(f"\nbest-IoU distribution per annotated word:")
    for lo, hi in [(0.0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]:
        n = int(((arr >= lo) & (arr < hi)).sum())
        print(f"  [{lo:.2f}, {hi:.2f})   {n:5d}  ({100 * n / len(arr):4.1f}%)")

    ann_arr, ocr_arr = np.array(ann_dims), np.array(ocr_dims)
    print(f"\nbox sizes (median w x h):")
    print(f"  annotation  {np.median(ann_arr[:, 0]):.0f} x {np.median(ann_arr[:, 1]):.0f}")
    print(f"  ocr         {np.median(ocr_arr[:, 0]):.0f} x {np.median(ocr_arr[:, 1]):.0f}")

    # Eyeball doc 0
    s = ds["train"][0]
    gt = json.loads(s["ground_truth"])
    anns = annotated_words(gt)
    ocr = ocr_words(s["image"])
    print(f"\n--- doc 0: {len(anns)} annotated, {len(ocr)} ocr ---")
    print("\nfirst 25 annotations (text @ box):")
    for a in sorted(anns, key=lambda a: (a["box"][1], a["box"][0]))[:25]:
        b = [round(v) for v in a["box"]]
        print(f"  {a['text']:<20} {b}  {a['category']}")
    print("\nfirst 25 ocr tokens (text @ box, conf):")
    for t in sorted(ocr, key=lambda t: (t["box"][1], t["box"][0]))[:25]:
        b = [round(v) for v in t["box"]]
        print(f"  {t['text']:<20} {b}  conf={t['conf']}")


if __name__ == "__main__":
    main()