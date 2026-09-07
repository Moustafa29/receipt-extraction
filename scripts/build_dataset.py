"""OCR every CORD document, align labels, cache the result as JSONL.

    python scripts/build_dataset.py --config configs/base.yaml
    python scripts/build_dataset.py --limit 20          # quick smoke run

Writes data/processed/{train,validation,test}.jsonl and prints a coverage
report. Coverage is the fraction of annotated words whose label survived the
OCR round trip; whatever is lost is label noise and belongs in the README.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset

from docextract import ocr as ocr_mod
from docextract.align import align, parse_annotations
from docextract.labels import CATEGORIES, DROPPED, LABELS


def build_split(dataset, cfg: dict, split: str, limit: int | None) -> tuple[list, dict]:
    ocr_cfg = ocr_mod.OcrConfig.from_dict(cfg["ocr"])
    ocr_mod.configure(ocr_cfg)
    align_cfg = cfg["align"]

    records = []
    stats = {
        "documents": 0,
        "annotations": 0,
        "matched": 0,
        "ocr_tokens": 0,
        "background_tokens": 0,
        "missed_by_category": Counter(),
        "annotations_by_category": Counter(),
    }

    total = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(total):
        sample = dataset[i]
        ground_truth = json.loads(sample["ground_truth"])
        annotations = parse_annotations(ground_truth)
        words = ocr_mod.read(sample["image"], ocr_cfg)

        doc = align(
            words,
            annotations,
            iou_threshold=align_cfg["iou_threshold"],
            fallback_text_match=align_cfg["fallback_text_match"],
            fallback_y_tolerance=align_cfg["fallback_y_tolerance"],
        )

        width, height = sample["image"].size
        records.append(
            {
                "id": f"{split}-{i}",
                "width": width,
                "height": height,
                "words": doc.words,
                "boxes": [ocr_mod.normalise_box(b, width, height) for b in doc.boxes],
                "tags": doc.tags,
            }
        )

        stats["documents"] += 1
        stats["annotations"] += doc.total_annotations
        stats["matched"] += doc.matched
        stats["ocr_tokens"] += len(doc.words)
        stats["background_tokens"] += sum(1 for t in doc.tags if t == "O")
        for ann in annotations:
            stats["annotations_by_category"][ann.category] += 1

        if (i + 1) % 25 == 0:
            done = stats["matched"] / max(stats["annotations"], 1)
            print(f"  {split}: {i + 1}/{total}  coverage so far {100 * done:.1f}%")

    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="documents per split; omit for the full dataset")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["dataset"]["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"label set: {len(LABELS)} labels from {len(CATEGORIES)} categories "
          f"({len(DROPPED)} categories mapped to O)\n")

    ds = load_dataset(cfg["dataset"]["hf_name"])
    grand = {"annotations": 0, "matched": 0, "ocr": 0, "background": 0}

    for split in ("train", "validation", "test"):
        print(f"building {split}...")
        records, stats = build_split(ds[split], cfg, split, args.limit)

        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        coverage = 100 * stats["matched"] / max(stats["annotations"], 1)
        background = 100 * stats["background_tokens"] / max(stats["ocr_tokens"], 1)
        print(
            f"  -> {path}  {stats['documents']} docs, "
            f"{stats['ocr_tokens']} tokens "
            f"({stats['ocr_tokens'] / max(stats['documents'], 1):.0f}/doc)\n"
            f"     label coverage {coverage:.1f}%  "
            f"({stats['matched']}/{stats['annotations']} annotations survived OCR)\n"
            f"     background     {background:.1f}% of tokens are O\n"
        )

        grand["annotations"] += stats["annotations"]
        grand["matched"] += stats["matched"]
        grand["ocr"] += stats["ocr_tokens"]
        grand["background"] += stats["background_tokens"]

    print("=" * 60)
    print(f"overall label coverage: "
          f"{100 * grand['matched'] / max(grand['annotations'], 1):.1f}%")
    print(f"overall background:     "
          f"{100 * grand['background'] / max(grand['ocr'], 1):.1f}% of tokens")
    print("\nLabel loss is OCR miss, not an annotation error. Report it.")


if __name__ == "__main__":
    main()