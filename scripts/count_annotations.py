"""Count CORD annotated words per category per split.

    python scripts/count_annotations.py

Writes docs/annotation_counts.json, the denominator for word-level true
recall. This reads only the ground_truth column - no images, no OCR - so it
takes seconds, and the result is tracked in git so evaluation on Colab needs
no CORD download when training text+layout only.

The denominator must come from CORD rather than from the cached JSONL: the
whole point of true recall is to count annotated words OCR never detected,
which by definition are absent from the OCR output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from docextract.eval.metrics import (
    count_annotated_words,
    count_annotated_words_by_document,
)
from docextract.labels import CATEGORIES, DROPPED


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="docs/annotation_counts.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hf_name = cfg["dataset"]["hf_name"]

    out: dict = {
        "dataset": hf_name,
        "note": (
            "Annotated words per category, measured from CORD ground truth "
            "independently of OCR. 'kept' covers the 20 modelled categories; "
            "'all' includes the 9 mapped to O. 'documents' holds the same "
            "counts per document, so a subset run gets a subset denominator "
            "instead of dividing by the whole split. Denominator for "
            "word-level true recall."
        ),
        "kept_categories": list(CATEGORIES),
        "dropped_categories": sorted(DROPPED),
        "splits": {},
    }

    for split in ("train", "validation", "test"):
        kept = count_annotated_words(hf_name, split, kept_only=True)
        every = count_annotated_words(hf_name, split, kept_only=False)
        per_document = count_annotated_words_by_document(
            hf_name, split, kept_only=True
        )
        kept_total, all_total = sum(kept.values()), sum(every.values())
        out["splits"][split] = {
            "kept_total": kept_total,
            "all_total": all_total,
            "kept": dict(sorted(kept.items())),
            "documents": per_document,
        }
        print(
            f"{split:11s} {all_total:6d} annotated words, "
            f"{kept_total:6d} in kept categories "
            f"({100 * kept_total / max(all_total, 1):.1f}%)"
        )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
