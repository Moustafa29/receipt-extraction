"""Measure how well Tesseract OCR boxes align to CORD valid_line annotations.

This decides whether we can transfer labels from annotations onto OCR tokens.
Run:  python scripts/probe_alignment.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytesseract
from datasets import load_dataset

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

LANG = "ind+eng"
IOU_THRESHOLD = 0.5
N_DOCS = 20


def quad_to_box(quad: dict) -> list[int]:
    """Four corner points -> axis-aligned [x1, y1, x2, y2]."""
    xs = [quad[f"x{i}"] for i in range(1, 5)]
    ys = [quad[f"y{i}"] for i in range(1, 5)]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a: list[int], b: list[int]) -> float:
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


def ocr_words(image) -> list[dict]:
    """Tesseract word boxes, dropping empty and non-word entries."""
    data = pytesseract.image_to_data(
        image, lang=LANG, output_type=pytesseract.Output.DICT
    )
    words = []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        if not text or int(data["conf"][i]) < 0:
            continue
        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]
        words.append({"text": text, "box": [x, y, x + w, y + h]})
    return words


def annotated_words(gt: dict) -> list[dict]:
    out = []
    for line in gt["valid_line"]:
        for word in line["words"]:
            out.append(
                {
                    "text": word["text"],
                    "box": quad_to_box(word["quad"]),
                    "category": line["category"],
                }
            )
    return out


def main() -> None:
    ds = load_dataset("naver-clova-ix/cord-v2")

    total_ann = 0
    total_matched = 0
    total_ocr = 0
    total_background = 0
    text_exact = 0
    missed_by_cat = Counter()
    ann_by_cat = Counter()

    for idx in range(N_DOCS):
        sample = ds["train"][idx]
        gt = json.loads(sample["ground_truth"])

        anns = annotated_words(gt)
        ocr = ocr_words(sample["image"])

        used = set()
        for ann in anns:
            best_i, best_iou = -1, 0.0
            for i, tok in enumerate(ocr):
                if i in used:
                    continue
                score = iou(ann["box"], tok["box"])
                if score > best_iou:
                    best_i, best_iou = i, score

            ann_by_cat[ann["category"]] += 1
            if best_iou >= IOU_THRESHOLD:
                used.add(best_i)
                total_matched += 1
                if ocr[best_i]["text"] == ann["text"]:
                    text_exact += 1
            else:
                missed_by_cat[ann["category"]] += 1

        total_ann += len(anns)
        total_ocr += len(ocr)
        total_background += len(ocr) - len(used)

    print(f"documents probed:        {N_DOCS}")
    print(f"annotated words:         {total_ann}")
    print(f"OCR tokens found:        {total_ocr}")
    print(f"matched (IoU>={IOU_THRESHOLD}):     {total_matched}"
          f"  ({100 * total_matched / total_ann:.1f}% recall)")
    print(f"  of which text agrees:  {text_exact}"
          f"  ({100 * text_exact / max(total_matched, 1):.1f}%)")
    print(f"background (O) tokens:   {total_background}"
          f"  ({100 * total_background / max(total_ocr, 1):.1f}% of OCR)")
    print(f"tokens per doc:          {total_ocr / N_DOCS:.1f} OCR "
          f"vs {total_ann / N_DOCS:.1f} annotated")

    print("\nworst-aligned categories (miss rate, min 10 occurrences):")
    rows = [
        (cat, missed_by_cat[cat] / n, missed_by_cat[cat], n)
        for cat, n in ann_by_cat.items()
        if n >= 10
    ]
    for cat, rate, missed, n in sorted(rows, key=lambda r: -r[1])[:10]:
        print(f"  {cat:<28} {100 * rate:5.1f}%   ({missed}/{n})")

    # Eyeball one document
    sample = ds["train"][0]
    gt = json.loads(sample["ground_truth"])
    ocr = ocr_words(sample["image"])
    ann_texts = {w["text"] for w in annotated_words(gt)}
    extra = [t["text"] for t in ocr if t["text"] not in ann_texts]
    print(f"\ndoc 0 — OCR text with no annotation match ({len(extra)} tokens):")
    print("  " + " | ".join(extra[:40]))


if __name__ == "__main__":
    main()