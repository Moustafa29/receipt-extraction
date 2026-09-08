"""Evaluate a trained checkpoint and emit the README results table.

    python scripts/evaluate.py --checkpoint outputs/layoutlmv3-ocr/best
    python scripts/evaluate.py --checkpoint <dir> --split validation
    python scripts/evaluate.py --checkpoint <dir> --limit 10   # smoke

Writes markdown to --out (default docs/results.md) and prints the same thing.

Scoring goes through docextract.train.evaluate, the exact function training
used. Reimplementing the merge here would let evaluation and training drift
apart, and a divergence in how overlapping windows collapse would show up as a
metric change that looks like a model change.

Two numbers, and the gap between them is the finding:

    seqeval entity F1   scored over OCR tokens - what a published CORD number
                        measures, and blind to fields OCR never detected
    true recall         scored over every annotated word on the receipt,
                        including the ones OCR missed

The denominator for the second comes from CORD ground truth via
docs/annotation_counts.json, restricted to the documents actually evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LayoutLMv3ForTokenClassification

from docextract.dataset import build_dataset
from docextract.eval.metrics import (
    load_annotation_counts,
    results_table,
    true_recall,
)
from docextract.train import evaluate, resolve_amp


def run(
    config_path: str,
    checkpoint: str,
    split: str,
    limit: int | None,
    out_path: str | None,
) -> str:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_cfg, train_cfg = cfg["model"], cfg["train"]

    checkpoint_dir = Path(checkpoint)
    if not (checkpoint_dir / "config.json").is_file():
        raise SystemExit(
            f"{checkpoint_dir} is not a checkpoint directory "
            "(no config.json). Point --checkpoint at the 'best' folder."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Evaluation under autocast matches training numerics; on CPU it degrades
    # to full precision exactly as the training loop does.
    use_amp, amp_dtype, amp_reason = resolve_amp(bool(train_cfg["fp16"]), device)
    print(f"device: {device} - {amp_reason}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, add_prefix_space=True)
    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint_dir).to(device)

    jsonl = Path(cfg["dataset"]["cache_dir"]) / f"{split}.jsonl"
    dataset = build_dataset(
        jsonl,
        tokenizer,
        max_length=model_cfg["max_length"],
        overlap_words=model_cfg["overlap_words"],
        limit=limit,
    )
    print(f"{split}: {len(dataset.documents)} documents, {len(dataset)} windows")

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    metrics, loss, predicted = evaluate(model, loader, dataset, device, amp_dtype)

    counts = load_annotation_counts(
        train_cfg["annotation_counts"],
        split,
        doc_ids=[document.id for document in dataset.documents],
    )
    recall = true_recall([d.tags for d in dataset.documents], predicted, counts)

    selection = checkpoint_dir / "selection.json"
    provenance = ""
    if selection.is_file():
        chosen = json.loads(selection.read_text(encoding="utf-8"))
        provenance = (
            f"\nCheckpoint: epoch {chosen.get('epoch')}, selected on "
            f"{chosen.get('selected_on')}. The checkpoint maximises validation "
            f"F1, not the true-recall figure reported as the headline.\n"
        )

    table = results_table(
        metrics, recall, title=f"Results - {split} split"
    ) + f"\n\nMean loss {loss:.4f} over {len(dataset)} windows.\n" + provenance

    print()
    print(table)

    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table + "\n", encoding="utf-8")
        print(f"\n-> {path}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="outputs/layoutlmv3-ocr/best")
    parser.add_argument("--split", default="test",
                        choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="docs/results.md",
                        help="markdown destination; empty string to skip")
    args = parser.parse_args()

    run(args.config, args.checkpoint, args.split, args.limit, args.out or None)


if __name__ == "__main__":
    main()
