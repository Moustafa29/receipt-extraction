"""Fine-tune LayoutLMv3 for token classification on the OCR-aligned corpus.

    python -m docextract.train --config configs/base.yaml
    python -m docextract.train --limit 10 --epochs 1     # CPU smoke run

Written as an explicit loop rather than transformers.Trainer. Evaluation here
is not a per-batch reduction: predictions have to be merged across the
overlapping windows of a document before any metric is meaningful, and the
window bookkeeping that makes that possible has to be carried alongside the
batch and stripped before the forward pass. That is awkward to express through
Trainer's callbacks and trivial in a loop.

Checkpoint selection
    Early stopping watches seqeval micro F1, not the headline true-recall
    number and not validation loss. Loss keeps improving after F1 plateaus
    because 88.4% of tokens are O and the model gets steadily more confident
    about them. True recall ignores precision, so stopping on it would select
    a model that over-predicts. The consequence is explicit and belongs in the
    README: the saved checkpoint maximises F1, not the number reported as the
    headline.

The validation curve
    Validation is 100 documents, ~2167 annotated words. A 1% F1 move is around
    20 entities, so the curve is noisy enough that a stopping point can be an
    artefact. Every epoch is appended to metrics.csv as it completes, and the
    run ends by reporting epoch-to-epoch volatility against the size of the
    improvement that selected the checkpoint. If the swing is the same size as
    the gain, the number to trust is neither.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    LayoutLMv3ForTokenClassification,
    get_linear_schedule_with_warmup,
)

from .dataset import build_dataset, merge_predictions, split_batch
from .eval.metrics import (
    EntityMetrics,
    entity_metrics,
    ids_to_tags,
    load_annotation_counts,
    results_table,
    true_recall,
)
from .labels import ID2LABEL, LABEL2ID, LABELS

CSV_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_micro_f1",
    "val_micro_precision",
    "val_micro_recall",
    "val_macro_f1",
    "val_true_recall",
    "val_ocr_ceiling",
    "val_tagger_accuracy",
    "learning_rate",
    "epoch_seconds",
    "is_best",
)


def resolve_amp(fp16: bool, device: torch.device) -> tuple[bool, object, str]:
    """Decide whether mixed precision runs, and say why in words.

    fp16 autocast is a CUDA path. On CPU it is unsupported, so the config flag
    degrades to full precision rather than raising - otherwise every local
    smoke run would fail on a setting that is correct for Colab. The returned
    reason exists so a test can assert the CPU path was *skipped* rather than
    silently passing as though fp16 had been exercised.
    """
    if not fp16:
        return False, None, "disabled by config (train.fp16: false)"
    if device.type != "cuda":
        return False, None, (
            f"fp16 autocast requires CUDA; device is {device.type}. "
            "Mixed precision is UNVERIFIED by this run - exercise it on a GPU "
            "before trusting a Colab session."
        )
    return True, torch.float16, "fp16 autocast with gradient scaling"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_micro_f1: float
    val_micro_precision: float
    val_micro_recall: float
    val_macro_f1: float
    val_true_recall: float
    val_ocr_ceiling: float
    val_tagger_accuracy: float
    learning_rate: float
    epoch_seconds: float
    is_best: bool


class RunLog:
    """CSV always, Weights & Biases when it is available and asked for.

    The CSV is flushed every epoch: a Colab session that dies at epoch 14 must
    still leave 13 plottable rows behind.
    """

    def __init__(self, output_dir: Path, wandb_cfg: dict, config: dict) -> None:
        self.path = output_dir / "metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._handle.flush()

        self.wandb = None
        if wandb_cfg.get("enabled"):
            try:
                import wandb

                wandb.init(project=wandb_cfg.get("project", "receipt-extraction"),
                           config=config)
                self.wandb = wandb
                print("logging to Weights & Biases")
            except Exception as exc:
                print(f"W&B unavailable ({exc}); using {self.path} only")

    def append(self, record: EpochRecord) -> None:
        self._writer.writerow(asdict(record))
        self._handle.flush()
        if self.wandb is not None:
            self.wandb.log(asdict(record), step=record.epoch)

    def close(self) -> None:
        self._handle.close()
        if self.wandb is not None:
            self.wandb.finish()


def evaluate(
    model,
    loader: DataLoader,
    dataset,
    device: torch.device,
    amp_dtype,
) -> tuple[EntityMetrics, float, list[list[str]]]:
    """Run the model over every window, merge, then score at document level.

    Returns the entity metrics, mean validation loss, and the merged per-word
    tag sequences. True recall is left to the caller, which owns the CORD
    annotation census that supplies its denominator.
    """
    model.eval()

    logit_chunks: list[np.ndarray] = []
    word_index_chunks: list[np.ndarray] = []
    doc_indices: list[int] = []
    total_loss, batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            inputs, meta = split_batch(batch)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                outputs = model(**inputs)

            total_loss += outputs.loss.detach().item()
            batches += 1
            # float() before numpy: fp16 logits would lose resolution in the
            # softmax averaging that merging performs.
            logit_chunks.append(outputs.logits.float().cpu().numpy())
            word_index_chunks.append(meta["word_index"].numpy())
            doc_indices.extend(
                dataset.windows[int(i)].doc_index for i in meta["window_id"].tolist()
            )

    logits = np.concatenate(logit_chunks, axis=0)
    word_index = np.concatenate(word_index_chunks, axis=0)

    merged = merge_predictions(logits, word_index, doc_indices, dataset.word_counts)
    predicted = [ids_to_tags(doc) for doc in merged]
    gold = [document.tags for document in dataset.documents]

    return entity_metrics(gold, predicted), total_loss / max(batches, 1), predicted


def curve_verdict(f1_by_epoch: list[float], best: float) -> str:
    """Is the selected checkpoint a real peak or a bounce?

    Compares the typical epoch-to-epoch swing against how far the best epoch
    stands above the runner-up. When the swing is the larger of the two, the
    stopping point is not distinguishable from noise on a 100-document split.
    """
    if len(f1_by_epoch) < 3:
        return "too few epochs to judge curve stability"

    # Measure the swing over the plateau, not the opening climb. Early epochs
    # improve by large genuine steps; including them inflates the estimate of
    # routine noise and would flag a healthy run as unstable.
    tail = f1_by_epoch[len(f1_by_epoch) // 2:] if len(f1_by_epoch) >= 6 else f1_by_epoch
    deltas = [abs(b - a) for a, b in zip(tail, tail[1:])]
    if not deltas:
        return "too few epochs to judge curve stability"
    swing = statistics.median(deltas)
    runner_up = max(f for f in f1_by_epoch if f != best) if len(set(f1_by_epoch)) > 1 else best
    margin = best - runner_up

    verdict = (
        f"median epoch-to-epoch F1 swing {100 * swing:.2f} points; "
        f"best epoch beats the runner-up by {100 * margin:.2f} points"
    )
    if margin < swing:
        return (
            verdict + "\n  WARNING: the winning margin is smaller than the "
            "routine swing. Early stopping is selecting noise - treat the "
            "checkpoint as one draw from a plateau, not a peak."
        )
    return verdict + "\n  The peak is larger than the routine swing."


def train(
    cfg: dict,
    limit: int | None = None,
    epochs_override: int | None = None,
    data_prefix: str = "",
) -> Path:
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    set_seed(train_cfg["seed"])
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, amp_dtype, amp_reason = resolve_amp(bool(train_cfg["fp16"]), device)
    print(f"device: {device}  mixed precision: {use_amp} - {amp_reason}")

    if model_cfg.get("use_images"):
        raise NotImplementedError(
            "use_images is the item-5 ablation; the main path is text+layout. "
            "Set model.use_images: false."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], add_prefix_space=True)
    cache_dir = Path(cfg["dataset"]["cache_dir"])

    datasets = {}
    for split in ("train", "validation"):
        datasets[split] = build_dataset(
            cache_dir / f"{data_prefix}{split}.jsonl",
            tokenizer,
            max_length=model_cfg["max_length"],
            overlap_words=model_cfg["overlap_words"],
            limit=limit,
        )
        print(f"{split}: {len(datasets[split].documents)} documents, "
              f"{len(datasets[split])} windows")

    # Denominator restricted to the documents actually loaded, so true recall
    # stays correct under --limit instead of dividing by the whole split.
    annotation_counts = load_annotation_counts(
        train_cfg["annotation_counts"],
        "validation",
        doc_ids=[document.id for document in datasets["validation"].documents],
    )
    print(f"validation denominator: {sum(annotation_counts.values())} "
          f"annotated words across {len(datasets['validation'].documents)} documents")

    train_loader = DataLoader(
        datasets["train"],
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        drop_last=False,
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=train_cfg["batch_size"],
        shuffle=False,  # merging maps windows back by window_id, but a stable
        num_workers=train_cfg["num_workers"],  # order keeps runs comparable
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_cfg["name"],
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    epochs = epochs_override or train_cfg["epochs"]
    accum = train_cfg["grad_accum"]
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_cfg["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log = RunLog(output_dir, train_cfg.get("wandb", {}), cfg)
    best_f1, best_epoch, patience_left = -1.0, 0, train_cfg["patience"]
    f1_curve: list[float] = []

    print(f"\n{len(train_loader)} batches/epoch, accumulating {accum} "
          f"-> effective batch {train_cfg['batch_size'] * accum}, "
          f"{total_steps} optimiser steps over {epochs} epochs\n")

    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        running_loss, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            inputs, _ = split_batch(batch)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                outputs = model(**inputs)
                # Scale so accumulated gradients average rather than sum.
                loss = outputs.loss / accum

            scaler.scale(loss).backward()
            # detach first: float() on a grad-tracking tensor warns, and the
            # running total must not hold a reference to the graph.
            running_loss += outputs.loss.detach().item()
            seen += 1

            is_last = step == len(train_loader)
            if step % accum == 0 or is_last:
                # Unscale before clipping, or the threshold applies to
                # fp16-scaled gradients and does nothing predictable.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_cfg["max_grad_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        metrics, val_loss, predicted = evaluate(
            model, val_loader, datasets["validation"], device, amp_dtype
        )
        recall = true_recall(
            [d.tags for d in datasets["validation"].documents],
            predicted,
            annotation_counts,
        )

        f1_curve.append(metrics.micro_f1)
        is_best = metrics.micro_f1 > best_f1
        if is_best:
            best_f1, best_epoch = metrics.micro_f1, epoch
            patience_left = train_cfg["patience"]
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
            (output_dir / "best" / "selection.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "selected_on": "seqeval micro F1 (validation)",
                        "val_micro_f1": metrics.micro_f1,
                        "val_true_recall": recall.recall,
                        "note": "checkpoint maximises F1, not true recall",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (output_dir / "best" / "results.md").write_text(
                results_table(metrics, recall,
                              title=f"Validation, epoch {epoch}"),
                encoding="utf-8",
            )
        else:
            patience_left -= 1

        record = EpochRecord(
            epoch=epoch,
            train_loss=running_loss / max(seen, 1),
            val_loss=val_loss,
            val_micro_f1=metrics.micro_f1,
            val_micro_precision=metrics.micro_precision,
            val_micro_recall=metrics.micro_recall,
            val_macro_f1=metrics.macro_f1,
            val_true_recall=recall.recall,
            val_ocr_ceiling=recall.ocr_ceiling,
            val_tagger_accuracy=recall.tagger_accuracy,
            learning_rate=scheduler.get_last_lr()[0],
            epoch_seconds=time.time() - started,
            is_best=is_best,
        )
        log.append(record)

        print(
            f"epoch {epoch:2d}/{epochs}  "
            f"train_loss {record.train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  "
            f"micro_F1 {100 * metrics.micro_f1:5.2f}  "
            f"macro_F1 {100 * metrics.macro_f1:5.2f}  "
            f"true_recall {100 * recall.recall:5.2f}%  "
            f"{'*best*' if is_best else f'patience {patience_left}'}  "
            f"{record.epoch_seconds:.0f}s"
        )

        if patience_left <= 0:
            print(f"\nno micro-F1 gain in {train_cfg['patience']} epochs; stopping")
            break

    log.close()

    print(f"\nbest epoch {best_epoch}, validation micro F1 {100 * best_f1:.2f}")
    print(curve_verdict(f1_curve, best_f1))
    print(f"checkpoint: {output_dir / 'best'}")
    print(f"curve:      {output_dir / 'metrics.csv'}")
    return output_dir / "best"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="documents per split; for smoke runs")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override train.epochs")
    parser.add_argument("--data-prefix", default="",
                        help="corpus filename prefix, e.g. baseline_")
    parser.add_argument("--output-dir", default=None,
                        help="override train.output_dir")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    train(cfg, limit=args.limit, epochs_override=args.epochs,
          data_prefix=args.data_prefix)


if __name__ == "__main__":
    main()
