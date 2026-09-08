"""Naive-baseline corpus tests. Run with: pytest -q

The baseline exists to be worse than the real pipeline in a specific way, so
these check it is wrong for the intended reason - no background class - and
right about everything else. A baseline that differs in token order, label
schema or box normalisation would confound the comparison it exists to make.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from docextract.dataset import load_documents
from docextract.eval.metrics import load_annotation_counts
from docextract.labels import LABEL2ID

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "data" / "processed"


def baseline(split: str):
    path = PROCESSED / f"baseline_{split}.jsonl"
    if not path.is_file():
        pytest.skip("run scripts/build_baseline.py first")
    return load_documents(path)


def test_baseline_token_count_matches_the_cord_census():
    """Every annotated word must appear exactly once, or the baseline is
    training on a different corpus than the one true recall divides by."""
    census = REPO / "docs" / "annotation_counts.json"
    if not census.is_file():
        pytest.skip("no census")

    data = json.loads(census.read_text(encoding="utf-8"))
    for split in ("train", "validation", "test"):
        docs = baseline(split)
        tokens = sum(len(d.words) for d in docs)
        assert tokens == data["splits"][split]["all_total"], split


def test_baseline_has_almost_no_background():
    """~1% O against the real corpus's 87.7% - the mismatch being measured."""
    docs = baseline("test")
    tags = [t for d in docs for t in d.tags]
    share = sum(1 for t in tags if t == "O") / len(tags)

    assert share < 0.05, f"background {share:.1%} is too high to be annotations"
    assert share > 0.0, "dropped categories should still produce some O"


def test_real_corpus_is_overwhelmingly_background():
    """The control for the test above."""
    path = PROCESSED / "test.jsonl"
    if not path.is_file():
        pytest.skip("no OCR corpus")

    tags = [t for d in load_documents(path) for t in d.tags]
    share = sum(1 for t in tags if t == "O") / len(tags)
    assert share > 0.8, f"expected ~88% background, got {share:.1%}"


def test_baseline_documents_are_far_shorter_than_ocr_documents():
    ocr_path = PROCESSED / "test.jsonl"
    if not ocr_path.is_file():
        pytest.skip("no OCR corpus")

    base = baseline("test")
    ocr = load_documents(ocr_path)
    base_len = sum(len(d.words) for d in base) / len(base)
    ocr_len = sum(len(d.words) for d in ocr) / len(ocr)

    assert base_len < ocr_len / 3, f"{base_len:.0f} vs {ocr_len:.0f} words/doc"


def test_baseline_words_are_in_reading_order():
    """Token order must match how ocr.read sorts, or the comparison also
    varies the 1D position embeddings."""
    for document in baseline("test")[:20]:
        tops = [box[1] for box in document.boxes]
        assert tops == sorted(tops), f"{document.id} is not sorted by y"


def test_baseline_boxes_are_on_the_grid():
    for document in baseline("test")[:20]:
        for box in document.boxes:
            assert all(0 <= v <= 1000 for v in box)
            assert box[0] <= box[2] and box[1] <= box[3]


def test_baseline_uses_the_same_label_schema():
    tags = {t for d in baseline("test") for t in d.tags}
    assert tags <= set(LABEL2ID), tags - set(LABEL2ID)


def test_baseline_bio_never_opens_with_i():
    """Gold must be valid IOB2 or strict-mode seqeval scores it as nothing."""
    for document in baseline("test"):
        previous = "O"
        for tag in document.tags:
            if tag.startswith("I-"):
                assert previous.endswith(tag[2:]) and previous != "O", (
                    f"{document.id}: {tag} after {previous}"
                )
            previous = tag


def test_baseline_ids_match_the_ocr_corpus():
    """Shared ids let the census supply a denominator for either corpus."""
    ocr_path = PROCESSED / "test.jsonl"
    if not ocr_path.is_file():
        pytest.skip("no OCR corpus")

    base_ids = {d.id for d in baseline("test")}
    ocr_ids = {d.id for d in load_documents(ocr_path)}
    assert base_ids == ocr_ids


def test_baseline_ceiling_is_effectively_total():
    """Evaluated in annotation space every annotated word is present, so the
    OCR ceiling is ~100% and true recall collapses to tagger accuracy. That is
    the world a published CORD number is measured in."""
    from docextract.eval.metrics import true_recall

    census = REPO / "docs" / "annotation_counts.json"
    if not census.is_file():
        pytest.skip("no census")

    docs = baseline("test")
    gold = [d.tags for d in docs]
    counts = load_annotation_counts(census, "test", doc_ids=[d.id for d in docs])
    recall = true_recall(gold, gold, counts)

    assert recall.ocr_ceiling == pytest.approx(1.0, abs=0.001)
    assert recall.recall == pytest.approx(1.0, abs=0.001)
