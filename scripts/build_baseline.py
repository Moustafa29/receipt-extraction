"""Build the naive baseline corpus: CORD valid_line boxes, no OCR.

    python scripts/build_baseline.py
    python scripts/build_baseline.py --order annotation

Writes data/processed/baseline_{train,validation,test}.jsonl in the same
schema as the real corpus, so the same Dataset, trainer and metrics apply
without a second code path.

This is the standard approach the README argues against: train a token
classifier on the annotated fields alone. Those annotations cover ~24 words on
a receipt carrying well over a hundred, and 98.9% of them are a field. The
model therefore never sees a background token and is never asked to decide
that something is *not* a field. Evaluated on real OCR output containing the
whole page, it has no way to reject anything.

Nothing here decodes an image. Boxes come from CORD quads, and the width and
height needed to normalise them onto the 0-1000 grid are already in the cached
OCR corpus, keyed by the same f"{split}-{index}" ids.

Word order
    --order reading (default) sorts annotation words by (y, x), matching how
    ocr.read orders real OCR output. LayoutLMv3 has 1D position embeddings, so
    token order is a real input; sorting both corpora the same way keeps the
    comparison about the missing background class rather than about ordering.
    --order annotation keeps CORD's valid_line order instead, which is what a
    naive implementation would do.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset

from docextract.align import align, parse_annotations
from docextract.ocr import Word, normalise_box


def load_dimensions(path: Path) -> dict[str, tuple[int, int]]:
    """Image dimensions from the cached OCR corpus, keyed by document id."""
    dimensions: dict[str, tuple[int, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            dimensions[record["id"]] = (record["width"], record["height"])
    return dimensions


def build_split(dataset, split: str, dimensions: dict, order: str) -> tuple[list, dict]:
    records, stats = [], {"documents": 0, "words": 0, "background": 0, "skipped": 0}

    for index in range(len(dataset)):
        doc_id = f"{split}-{index}"
        if doc_id not in dimensions:
            stats["skipped"] += 1
            continue

        annotations = parse_annotations(json.loads(dataset[index]["ground_truth"]))
        annotations = [a for a in annotations if a.text]
        if not annotations:
            stats["skipped"] += 1
            continue

        # Perfect "OCR": every annotation is its own token, boxes exact. align
        # then assigns BIO through the same code the real corpus uses, so the
        # two corpora cannot disagree about how B- and I- are opened.
        words = [
            Word(text=a.text, box=a.box, confidence=100) for a in annotations
        ]
        if order == "reading":
            words.sort(key=lambda w: (w.box[1], w.box[0]))

        document = align(words, annotations)
        width, height = dimensions[doc_id]
        records.append(
            {
                "id": doc_id,
                "width": width,
                "height": height,
                "words": document.words,
                "boxes": [normalise_box(b, width, height) for b in document.boxes],
                "tags": document.tags,
            }
        )
        stats["documents"] += 1
        stats["words"] += len(document.words)
        stats["background"] += sum(1 for t in document.tags if t == "O")

    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--order", default="reading",
                        choices=("reading", "annotation"),
                        help="token order; reading matches real OCR output")
    parser.add_argument("--prefix", default="baseline_")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["dataset"]["cache_dir"])
    dataset = load_dataset(cfg["dataset"]["hf_name"])

    print(f"token order: {args.order}\n")
    for split in ("train", "validation", "test"):
        source = out_dir / f"{split}.jsonl"
        if not source.is_file():
            raise SystemExit(
                f"{source} is missing; the baseline reuses its image "
                "dimensions. Run scripts/build_dataset.py first."
            )

        records, stats = build_split(
            dataset[split], split, load_dimensions(source), args.order
        )
        path = out_dir / f"{args.prefix}{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        background = 100 * stats["background"] / max(stats["words"], 1)
        print(
            f"{split:11s} -> {path}\n"
            f"    {stats['documents']} docs, {stats['words']} tokens "
            f"({stats['words'] / max(stats['documents'], 1):.0f}/doc), "
            f"background {background:.1f}%"
            + (f", {stats['skipped']} skipped" if stats["skipped"] else "")
        )

    print(
        "\nBackground here is the share of annotated words in dropped "
        "categories.\nThe real corpus is 87.7% background. That gap is the "
        "train/serve mismatch\nthis baseline exists to measure."
    )


if __name__ == "__main__":
    main()
