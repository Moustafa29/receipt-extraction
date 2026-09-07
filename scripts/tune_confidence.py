"""Pick min_confidence, and check nothing overflows LayoutLMv3's 512 tokens.

Run:  python scripts/tune_confidence.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

from docextract import ocr as ocr_mod
from docextract.align import align, parse_annotations

N_DOCS = 20
THRESHOLDS = [0, 10, 20, 30, 50, 70]


def main() -> None:
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    ds = load_dataset(cfg["dataset"]["hf_name"])["train"]

    base = ocr_mod.OcrConfig.from_dict({**cfg["ocr"], "min_confidence": 0})
    ocr_mod.configure(base)
    align_cfg = cfg["align"]

    # OCR once at confidence 0, then filter in memory for each threshold.
    cached = []
    for i in range(N_DOCS):
        sample = ds[i]
        gt = json.loads(sample["ground_truth"])
        cached.append(
            (
                ocr_mod.read(sample["image"], base),
                parse_annotations(gt),
                sample["image"].size,
            )
        )
        if (i + 1) % 5 == 0:
            print(f"  ocr {i + 1}/{N_DOCS}")

    print(f"\n{'min_conf':>9}{'coverage':>11}{'tok/doc':>10}{'O share':>10}")
    print("-" * 40)

    for threshold in THRESHOLDS:
        matched = total_ann = tokens = background = 0
        for words, anns, _ in cached:
            kept = [w for w in words if w.confidence >= threshold]
            doc = align(
                kept,
                anns,
                iou_threshold=align_cfg["iou_threshold"],
                fallback_text_match=align_cfg["fallback_text_match"],
                fallback_y_tolerance=align_cfg["fallback_y_tolerance"],
            )
            matched += doc.matched
            total_ann += doc.total_annotations
            tokens += len(doc.words)
            background += sum(1 for t in doc.tags if t == "O")

        print(f"{threshold:>9}{100 * matched / max(total_ann, 1):>10.1f}%"
              f"{tokens / N_DOCS:>10.1f}{100 * background / max(tokens, 1):>9.1f}%")

    # Sequence length at every threshold, so the choice accounts for truncation.
    # LayoutLMv3's tokenizer needs boxes alongside the words - it cannot
    # tokenise bare text.
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], apply_ocr=False)

    print(f"\n{'min_conf':>9}{'mean':>8}{'p50':>7}{'p95':>7}{'max':>7}{'over 512':>11}")
    print("-" * 49)

    for threshold in THRESHOLDS:
        lengths = []
        for words, _, (width, height) in cached:
            kept = [w for w in words if w.confidence >= threshold]
            if not kept:
                lengths.append(0)
                continue
            encoded = tokenizer(
                text=[w.text for w in kept],
                boxes=[ocr_mod.normalise_box(w.box, width, height) for w in kept],
            )
            lengths.append(len(encoded["input_ids"]))

        arr = np.array(lengths)
        over = int((arr > 512).sum())
        print(f"{threshold:>9}{arr.mean():>8.0f}{np.percentile(arr, 50):>7.0f}"
              f"{np.percentile(arr, 95):>7.0f}{arr.max():>7}"
              f"{over:>6}/{len(arr)}")

    print("\nPick the largest threshold whose p95 sits comfortably under 512.")


if __name__ == "__main__":
    main()