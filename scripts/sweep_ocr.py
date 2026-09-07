"""Sweep Tesseract PSM modes x preprocessing to maximise OCR recall on CORD.

Run:  python scripts/sweep_ocr.py
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
IOU_THRESHOLD = 0.5
N_DOCS = 10
PSM_MODES = [3, 4, 6, 11, 12]


def quad_to_box(quad: dict) -> list[int]:
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


def annotated_words(gt: dict) -> list[dict]:
    return [
        {"text": w["text"], "box": quad_to_box(w["quad"]), "category": line["category"]}
        for line in gt["valid_line"]
        for w in line["words"]
    ]


# --- preprocessing variants -------------------------------------------------
# Each returns (image, scale) where scale maps OCR coords back to original space.

def prep_none(img: Image.Image):
    return img, 1.0


def prep_gray_2x(img: Image.Image):
    g = ImageOps.grayscale(img)
    g = g.resize((g.width * 2, g.height * 2), Image.LANCZOS)
    return g, 2.0


def prep_gray_2x_otsu(img: Image.Image):
    g, scale = prep_gray_2x(img)
    arr = np.asarray(g)
    hist = np.bincount(arr.ravel(), minlength=256).astype(float)
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    w_bg = np.cumsum(hist)
    sum_bg = np.cumsum(np.arange(256) * hist)
    w_fg = total - w_bg
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        variance = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    variance = np.nan_to_num(variance)
    threshold = int(np.argmax(variance))
    binary = (arr > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary), scale


def prep_gray_3x(img: Image.Image):
    g = ImageOps.grayscale(img)
    g = g.resize((g.width * 3, g.height * 3), Image.LANCZOS)
    return g, 3.0


PREPROCESSORS = {
    "raw": prep_none,
    "gray2x": prep_gray_2x,
    "gray2x+otsu": prep_gray_2x_otsu,
    "gray3x": prep_gray_3x,
}


def ocr_words(image: Image.Image, psm: int, scale: float) -> list[dict]:
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(
        image, lang=LANG, config=config, output_type=pytesseract.Output.DICT
    )
    words = []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        if not text or int(data["conf"][i]) < 0:
            continue
        x, y = data["left"][i] / scale, data["top"][i] / scale
        w, h = data["width"][i] / scale, data["height"][i] / scale
        words.append({"text": text, "box": [x, y, x + w, y + h]})
    return words


def score(samples, psm: int, prep_name: str) -> tuple[float, float, float]:
    prep = PREPROCESSORS[prep_name]
    total_ann = total_matched = total_ocr = text_ok = 0

    for image, anns in samples:
        img, scale = prep(image)
        try:
            ocr = ocr_words(img, psm, scale)
        except Exception:
            return 0.0, 0.0, 0.0

        used = set()
        for ann in anns:
            best_i, best = -1, 0.0
            for i, tok in enumerate(ocr):
                if i in used:
                    continue
                s = iou(ann["box"], tok["box"])
                if s > best:
                    best_i, best = i, s
            if best >= IOU_THRESHOLD:
                used.add(best_i)
                total_matched += 1
                if ocr[best_i]["text"] == ann["text"]:
                    text_ok += 1

        total_ann += len(anns)
        total_ocr += len(ocr)

    recall = 100 * total_matched / max(total_ann, 1)
    text_acc = 100 * text_ok / max(total_matched, 1)
    return recall, text_acc, total_ocr / len(samples)


def main() -> None:
    ds = load_dataset("naver-clova-ix/cord-v2")
    samples = []
    for idx in range(N_DOCS):
        s = ds["train"][idx]
        gt = json.loads(s["ground_truth"])
        samples.append((s["image"], annotated_words(gt)))

    ann_per_doc = sum(len(a) for _, a in samples) / len(samples)
    print(f"{N_DOCS} docs, {ann_per_doc:.1f} annotated words/doc\n")
    print(f"{'preprocess':<14}{'psm':>5}{'recall%':>10}{'text%':>8}{'tok/doc':>10}")
    print("-" * 47)

    best = None
    for prep_name in PREPROCESSORS:
        for psm in PSM_MODES:
            recall, text_acc, tok = score(samples, psm, prep_name)
            print(f"{prep_name:<14}{psm:>5}{recall:>10.1f}{text_acc:>8.1f}{tok:>10.1f}")
            if best is None or recall > best[0]:
                best = (recall, prep_name, psm)
        print()

    print(f"best: {best[1]} psm={best[2]} -> {best[0]:.1f}% recall")


if __name__ == "__main__":
    main()