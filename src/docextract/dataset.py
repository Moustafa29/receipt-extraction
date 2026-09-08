"""LayoutLMv3 encoding for the OCR-aligned receipt corpus.

Reads the JSONL written by scripts/build_dataset.py and turns it into model
inputs. Three things here are not the obvious implementation, and each is a
consequence of a measurement in docs/measurements.md.

Subword labels
    LayoutLMv3 tokenises "Sub-Total" into several subwords, but the label
    belongs to the word, not to its pieces. Only the first subword carries the
    label; the rest get -100, which torch's cross-entropy ignores. Labelling
    every subword would train the model to re-predict the same field for each
    fragment and would inflate token counts for common multi-piece prices.

Word-level windows
    p95 sequence length is 638 subwords and the longest document is 1080
    (docs/measurements.md, "Confidence threshold"), against a 512-token limit.
    Truncating cuts the bottom of the receipt, which is exactly where
    total.total_price lives, so long documents are split into overlapping
    windows instead.

    The windows are planned over words, not subwords. The tokeniser's own
    stride option slices mid-word, which would strand a word's continuation
    subwords at the start of the next window where they look like a first
    subword. Packing whole words keeps "first subword" unambiguous and gives
    every window a contiguous word range.

Image branch
    LayoutLMv3 is multimodal, but pixel_values is optional and the cached JSONL
    holds no images. Text+layout is the default. Passing an image_loader turns
    the visual branch back on; CordImages recovers the originals from Hugging
    Face, since record ids are f"{split}-{index}" against the source dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import LABEL2ID, LABELS, OUTSIDE

# torch's cross-entropy skips positions holding this, so continuation subwords
# and padding contribute no gradient.
IGNORE_INDEX = -100

# Position carries no word label (special token, padding, or continuation
# subword). Kept distinct from IGNORE_INDEX: this indexes words, not classes.
NO_WORD = -1

# Keys the model accepts. Everything else in a batch is bookkeeping for merging
# and must be stripped before the forward pass; see split_batch.
MODEL_KEYS = ("input_ids", "attention_mask", "bbox", "labels", "pixel_values")


@dataclass(frozen=True)
class Document:
    """One receipt: OCR words, boxes on the 0-1000 grid, and BIO tags."""

    id: str
    words: list[str]
    boxes: list[list[int]]
    tags: list[str]
    width: int
    height: int

    def __post_init__(self) -> None:
        n = len(self.words)
        if not (len(self.boxes) == len(self.tags) == n):
            raise ValueError(
                f"{self.id}: {n} words, {len(self.boxes)} boxes, "
                f"{len(self.tags)} tags - must match"
            )
        for tag in self.tags:
            if tag not in LABEL2ID:
                raise ValueError(f"{self.id}: unknown tag {tag!r}")
        for box in self.boxes:
            if len(box) != 4:
                raise ValueError(f"{self.id}: box {box!r} is not 4 numbers")
            if not all(0 <= v <= 1000 for v in box):
                raise ValueError(
                    f"{self.id}: box {box!r} is off the 0-1000 grid - "
                    "boxes must already be normalised by ocr.normalise_box"
                )

    @property
    def label_ids(self) -> list[int]:
        return [LABEL2ID[tag] for tag in self.tags]


def load_documents(path: str | Path, limit: int | None = None) -> list[Document]:
    """Read one split's JSONL. Empty documents are dropped, not silently kept.

    A document with no OCR tokens would encode to nothing but special tokens
    and contribute a window of pure -100, which is a wasted forward pass.

    limit caps the number of documents, for smoke runs.
    """
    documents: list[Document] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            try:
                if not record["words"]:
                    continue
                documents.append(
                    Document(
                        id=record["id"],
                        words=record["words"],
                        boxes=record["boxes"],
                        tags=record["tags"],
                        width=record["width"],
                        height=record["height"],
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"{path}:{line_number}: record is missing {exc}"
                ) from exc
            if limit is not None and len(documents) >= limit:
                break
    if not documents:
        raise ValueError(f"{path}: no usable documents")
    return documents


@dataclass(frozen=True)
class WindowSpan:
    """Half-open word range [start, end) of one document."""

    doc_index: int
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


def plan_windows(
    subword_counts: Sequence[int],
    budget: int,
    overlap_words: int,
) -> list[tuple[int, int]]:
    """Pack words into overlapping ranges that each fit inside `budget`.

    `budget` is subwords available for content, i.e. max_length minus the
    special tokens the tokeniser adds. Returns half-open (start, end) word
    ranges covering every word at least once.

    A word whose own subword count exceeds the budget gets a window to itself
    and will be truncated by the tokeniser; that is a pathological OCR string,
    not a receipt field, but it must not wedge the packer in an infinite loop.
    """
    n = len(subword_counts)
    if budget < 1:
        raise ValueError(f"budget must be positive, got {budget}")
    if overlap_words < 0:
        raise ValueError(f"overlap_words must not be negative, got {overlap_words}")
    if n == 0:
        return []

    windows: list[tuple[int, int]] = []
    start = 0
    while start < n:
        used = 0
        end = start
        while end < n and used + subword_counts[end] <= budget:
            used += subword_counts[end]
            end += 1

        if end == start:
            # Single word larger than the whole budget.
            end = start + 1

        windows.append((start, end))
        if end >= n:
            break

        # Step back by the overlap, but always make forward progress.
        step_back = min(overlap_words, end - start - 1)
        start = end - step_back

    return windows


def _first_subword_positions(word_ids: Sequence[int | None]) -> list[int]:
    """Index of the word each position labels, or NO_WORD.

    Only the first subword of a word is marked. Because windows are planned
    over whole words, position 0 of a window is never a mid-word continuation,
    so comparing against the previous position is sufficient.
    """
    out: list[int] = []
    previous: int | None = None
    for word_id in word_ids:
        if word_id is None:
            out.append(NO_WORD)
        elif word_id != previous:
            out.append(word_id)
        else:
            out.append(NO_WORD)
        previous = word_id
    return out


class ReceiptDataset(Dataset):
    """Windows of encoded receipts, one window per item.

    Each item is a dict of tensors:
        input_ids, attention_mask, bbox, labels   - model inputs
        pixel_values                              - only when image_loader is set
        word_index                                - document word each position
                                                    labels, or -1
        window_id                                 - row in self.windows

    word_index and window_id are bookkeeping for merge_predictions and are not
    model arguments; use split_batch to separate them.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        tokenizer,
        max_length: int = 512,
        overlap_words: int = 64,
        image_loader: Callable[[Document], "object"] | None = None,
        image_processor=None,
    ) -> None:
        if image_loader is not None and image_processor is None:
            raise ValueError("image_loader needs an image_processor to encode with")

        self.documents = list(documents)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap_words = overlap_words
        self.image_loader = image_loader
        self.image_processor = image_processor

        # Special tokens (<s>, </s>) occupy positions that content cannot use.
        self.budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        if self.budget < 1:
            raise ValueError(f"max_length {max_length} leaves no room for content")

        self.windows: list[WindowSpan] = []
        self._encoded: list[dict[str, torch.Tensor]] = []
        self._build()

    def _subword_counts(self, words: Sequence[str]) -> list[int]:
        """Subwords each word costs, measured with the real tokeniser.

        add_special_tokens=False so the per-word cost excludes <s>/</s>, which
        are accounted for once per window in self.budget.
        """
        # A list of words plus boxes is pre-split by definition for this
        # tokeniser; passing is_split_into_words is an error.
        # verbose=False: this measures a whole document, which is routinely
        # longer than 512, and the tokeniser's length warning is not relevant
        # here - the result is never fed to the model, only counted.
        encoded = self.tokenizer(
            list(words),
            boxes=[[0, 0, 0, 0]] * len(words),
            add_special_tokens=False,
            verbose=False,
        )
        word_ids = encoded.word_ids(0)
        counts = [0] * len(words)
        for word_id in word_ids:
            if word_id is not None:
                counts[word_id] += 1
        # A word the tokeniser maps to nothing still needs a slot, or the
        # packer would place infinitely many of them in one window.
        return [max(1, c) for c in counts]

    def _build(self) -> None:
        for doc_index, document in enumerate(self.documents):
            counts = self._subword_counts(document.words)
            for start, end in plan_windows(counts, self.budget, self.overlap_words):
                span = WindowSpan(doc_index=doc_index, start=start, end=end)
                self.windows.append(span)
                self._encoded.append(self._encode(document, span))

    def _encode(self, document: Document, span: WindowSpan) -> dict[str, torch.Tensor]:
        words = document.words[span.start : span.end]
        boxes = document.boxes[span.start : span.end]
        label_ids = document.label_ids[span.start : span.end]

        encoding = self.tokenizer(
            words,
            boxes=boxes,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        positions = _first_subword_positions(encoding.word_ids(0))
        labels = [
            IGNORE_INDEX if p == NO_WORD else label_ids[p] for p in positions
        ]
        # Store document-relative word indices so merging needs no span lookup.
        word_index = [
            NO_WORD if p == NO_WORD else span.start + p for p in positions
        ]

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
            "bbox": torch.tensor(encoding["bbox"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "word_index": torch.tensor(word_index, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self._encoded[index])
        item["window_id"] = torch.tensor(index, dtype=torch.long)

        if self.image_loader is not None:
            document = self.documents[self.windows[index].doc_index]
            image = self.image_loader(document)
            processed = self.image_processor(images=image, return_tensors="pt")
            item["pixel_values"] = processed["pixel_values"][0]

        return item

    @property
    def window_doc_indices(self) -> list[int]:
        return [w.doc_index for w in self.windows]

    @property
    def word_counts(self) -> list[int]:
        return [len(d.words) for d in self.documents]

    def windows_per_document(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for span in self.windows:
            doc_id = self.documents[span.doc_index].id
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts


def split_batch(batch: dict) -> tuple[dict, dict]:
    """Separate model kwargs from merge bookkeeping.

    The default collate keeps every key, but LayoutLMv3.forward would reject
    word_index and window_id.
    """
    model_inputs = {k: v for k, v in batch.items() if k in MODEL_KEYS}
    meta = {k: v for k, v in batch.items() if k not in MODEL_KEYS}
    return model_inputs, meta


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def merge_predictions(
    logits,
    word_index,
    window_doc_indices: Sequence[int],
    word_counts: Sequence[int],
) -> list[list[int]]:
    """Collapse overlapping window predictions into one label per word.

    logits      (n_windows, seq_len, n_labels)
    word_index  (n_windows, seq_len), document word index or -1

    A word inside an overlap is predicted twice, from two different contexts.
    Probabilities are averaged rather than taking either window arbitrarily:
    softmax first, then mean, so a window that is confident outweighs one that
    is not. Averaging raw logits instead would let an unnormalised scale
    difference between windows decide the outcome.

    Words no window covered - only possible when a single word overflowed the
    whole budget and was truncated - fall back to O, which is the honest
    default given O is 88.4% of tokens.
    """
    logits = np.asarray(logits, dtype=np.float64)
    word_index = np.asarray(word_index)

    if logits.ndim != 3:
        raise ValueError(f"logits must be (windows, seq, labels), got {logits.shape}")
    if word_index.shape != logits.shape[:2]:
        raise ValueError(
            f"word_index {word_index.shape} does not match logits {logits.shape[:2]}"
        )
    if len(window_doc_indices) != logits.shape[0]:
        raise ValueError(
            f"{len(window_doc_indices)} window mappings for {logits.shape[0]} windows"
        )

    n_labels = logits.shape[-1]
    totals = [np.zeros((n, n_labels)) for n in word_counts]
    seen = [np.zeros(n, dtype=bool) for n in word_counts]

    for window in range(logits.shape[0]):
        doc = window_doc_indices[window]
        mask = word_index[window] >= 0
        if not mask.any():
            continue
        targets = word_index[window][mask]
        np.add.at(totals[doc], targets, _softmax(logits[window][mask]))
        seen[doc][targets] = True

    outside = LABEL2ID[OUTSIDE]
    predictions: list[list[int]] = []
    for doc, count in enumerate(word_counts):
        predicted = np.full(count, outside, dtype=int)
        covered = seen[doc]
        if covered.any():
            predicted[covered] = totals[doc][covered].argmax(axis=-1)
        predictions.append(predicted.tolist())
    return predictions


class CordImages:
    """Recover source images for the visual branch.

    build_dataset.py writes ids as f"{split}-{index}", where index is the row
    in the Hugging Face split, so the original photograph is addressable
    without caching pixels alongside the JSONL. Splits load on first use.
    """

    def __init__(self, hf_name: str = "naver-clova-ix/cord-v2") -> None:
        self.hf_name = hf_name
        self._splits: dict[str, object] = {}

    def __call__(self, document: Document):
        split, _, index = document.id.rpartition("-")
        if not split or not index.isdigit():
            raise ValueError(
                f"cannot recover an image from id {document.id!r}; "
                "expected '<split>-<index>'"
            )
        if split not in self._splits:
            from datasets import load_dataset

            self._splits[split] = load_dataset(self.hf_name, split=split)
        return self._splits[split][int(index)]["image"].convert("RGB")


def build_dataset(
    jsonl_path: str | Path,
    tokenizer,
    max_length: int = 512,
    overlap_words: int = 64,
    image_loader: Callable[[Document], "object"] | None = None,
    image_processor=None,
    limit: int | None = None,
) -> ReceiptDataset:
    return ReceiptDataset(
        load_documents(jsonl_path, limit=limit),
        tokenizer=tokenizer,
        max_length=max_length,
        overlap_words=overlap_words,
        image_loader=image_loader,
        image_processor=image_processor,
    )
