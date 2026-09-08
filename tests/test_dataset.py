"""Dataset encoding tests. Run with: pytest -q

Split in two. Everything about window planning and prediction merging is pure
arithmetic and runs unconditionally - those are the parts that would silently
corrupt training if wrong. The encoding tests need layoutlmv3-base and skip
with a reason when it is not available offline.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from docextract.dataset import (
    IGNORE_INDEX,
    NO_WORD,
    CordImages,
    Document,
    ReceiptDataset,
    _first_subword_positions,
    load_documents,
    merge_predictions,
    plan_windows,
    split_batch,
)
from docextract.labels import LABEL2ID


# --------------------------------------------------------------------------
# Window planning
# --------------------------------------------------------------------------


def test_short_document_is_one_window():
    assert plan_windows([2, 2, 2], budget=510, overlap_words=64) == [(0, 3)]


def test_empty_document_plans_nothing():
    assert plan_windows([], budget=510, overlap_words=64) == []


def test_windows_cover_every_word():
    counts = [3] * 100
    windows = plan_windows(counts, budget=30, overlap_words=2)

    covered = set()
    for start, end in windows:
        covered.update(range(start, end))
    assert covered == set(range(100))


def test_no_window_exceeds_budget():
    counts = [1, 5, 2, 8, 3, 1, 4, 2, 7, 1]
    for start, end in plan_windows(counts, budget=10, overlap_words=1):
        assert sum(counts[start:end]) <= 10


def test_consecutive_windows_overlap_by_requested_words():
    counts = [1] * 50
    windows = plan_windows(counts, budget=10, overlap_words=3)

    assert len(windows) > 1
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert prev_end - next_start == 3


def test_overlap_is_capped_so_windows_advance():
    # Overlap wider than the window itself must not stall the packer.
    counts = [5] * 20
    windows = plan_windows(counts, budget=10, overlap_words=64)

    assert len(windows) < 100  # terminated rather than looping
    for (prev_start, _), (next_start, _) in zip(windows, windows[1:]):
        assert next_start > prev_start


def test_word_larger_than_budget_gets_its_own_window():
    # A pathological OCR string must not wedge the packer.
    counts = [2, 900, 2]
    windows = plan_windows(counts, budget=510, overlap_words=0)

    assert (1, 2) in windows
    covered = set()
    for start, end in windows:
        covered.update(range(start, end))
    assert covered == {0, 1, 2}


def test_zero_overlap_produces_disjoint_windows():
    counts = [1] * 20
    windows = plan_windows(counts, budget=5, overlap_words=0)
    assert windows == [(0, 5), (5, 10), (10, 15), (15, 20)]


@pytest.mark.parametrize("budget,overlap", [(0, 4), (-1, 4), (10, -1)])
def test_plan_windows_rejects_nonsense_arguments(budget, overlap):
    with pytest.raises(ValueError):
        plan_windows([1, 1, 1], budget=budget, overlap_words=overlap)


# --------------------------------------------------------------------------
# Subword label alignment
# --------------------------------------------------------------------------


def test_only_first_subword_of_each_word_is_marked():
    # <s> w0 w0 w0 w1 w2 w2 </s>
    word_ids = [None, 0, 0, 0, 1, 2, 2, None]
    assert _first_subword_positions(word_ids) == [
        NO_WORD, 0, NO_WORD, NO_WORD, 1, 2, NO_WORD, NO_WORD
    ]


def test_repeated_word_id_after_gap_is_still_marked():
    # Defensive: a None between two pieces of the same word id must not merge
    # them into one span.
    word_ids = [0, None, 0]
    assert _first_subword_positions(word_ids) == [0, NO_WORD, 0]


# --------------------------------------------------------------------------
# Document validation
# --------------------------------------------------------------------------


def make_doc(doc_id="train-0", words=None, tags=None):
    words = words or ["Nasi", "75,000"]
    tags = tags or ["B-menu.nm", "B-menu.price"]
    # Boxes must stay on the 0-1000 grid however long the document is.
    boxes = [[10 * (i % 100), 10, 10 * (i % 100) + 5, 20] for i in range(len(words))]
    return Document(id=doc_id, words=words, boxes=boxes, tags=tags,
                    width=100, height=200)


def test_document_label_ids_follow_the_schema():
    doc = make_doc()
    assert doc.label_ids == [LABEL2ID["B-menu.nm"], LABEL2ID["B-menu.price"]]


def test_document_rejects_length_mismatch():
    with pytest.raises(ValueError, match="must match"):
        Document(id="x", words=["a", "b"], boxes=[[0, 0, 1, 1]],
                 tags=["O", "O"], width=10, height=10)


def test_document_rejects_unknown_tag():
    with pytest.raises(ValueError, match="unknown tag"):
        Document(id="x", words=["a"], boxes=[[0, 0, 1, 1]],
                 tags=["B-menu.vatyn"], width=10, height=10)


def test_document_rejects_unnormalised_box():
    # Raw pixel boxes are the likely mistake; they must not reach the model.
    with pytest.raises(ValueError, match="0-1000 grid"):
        Document(id="x", words=["a"], boxes=[[0, 0, 1500, 40]],
                 tags=["O"], width=2000, height=100)


def test_load_documents_skips_blank_lines_and_empty_documents(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"id": "train-0", "width": 100, "height": 100, "words": ["Nasi"],
         "boxes": [[0, 0, 10, 10]], "tags": ["B-menu.nm"]},
        {"id": "train-1", "width": 100, "height": 100, "words": [],
         "boxes": [], "tags": []},
    ]
    path.write_text(
        "\n\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    documents = load_documents(path)
    assert [d.id for d in documents] == ["train-0"]


def test_load_documents_reports_the_failing_line(tmp_path):
    good = {"id": "train-0", "width": 100, "height": 100, "words": ["Nasi"],
            "boxes": [[0, 0, 10, 10]], "tags": ["B-menu.nm"]}
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(good) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_documents(path)


def test_load_documents_reports_a_missing_field(tmp_path):
    path = tmp_path / "short.jsonl"
    path.write_text('{"id": "train-0", "words": ["Nasi"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_documents(path)


# --------------------------------------------------------------------------
# Prediction merging
# --------------------------------------------------------------------------


def one_hot_logits(rows, n_labels=4, strength=10.0):
    """(seq, n_labels) logits peaked at the given label per position."""
    out = np.zeros((len(rows), n_labels))
    for i, label in enumerate(rows):
        if label is not None:
            out[i, label] = strength
    return out


def test_merge_round_trips_a_single_window():
    gold = [1, 2, 3, 0]
    logits = np.stack([one_hot_logits([1, 2, 3, 0, None])])
    word_index = np.array([[0, 1, 2, 3, NO_WORD]])

    merged = merge_predictions(logits, word_index, [0], [4])
    assert merged == [gold]


def test_merge_averages_probabilities_across_an_overlap():
    # Word 2 appears in both windows: weakly label 1, strongly label 3.
    window_a = one_hot_logits([0, 1, 1], strength=0.5)
    window_b = one_hot_logits([3, 0, 0], strength=8.0)
    logits = np.stack([window_a, window_b])
    word_index = np.array([[0, 1, 2], [2, 3, 4]])

    merged = merge_predictions(logits, word_index, [0, 0], [5])
    assert merged[0][2] == 3


def test_merge_keeps_documents_separate():
    logits = np.stack([one_hot_logits([1, 1]), one_hot_logits([2, 2])])
    word_index = np.array([[0, 1], [0, 1]])

    merged = merge_predictions(logits, word_index, [0, 1], [2, 2])
    assert merged == [[1, 1], [2, 2]]


def test_uncovered_word_falls_back_to_outside():
    logits = np.stack([one_hot_logits([1, None])])
    word_index = np.array([[0, NO_WORD]])

    merged = merge_predictions(logits, word_index, [0], [3])
    assert merged == [[1, LABEL2ID["O"], LABEL2ID["O"]]]


def test_merge_returns_one_prediction_per_word():
    logits = np.stack([one_hot_logits([1, 2, 3])])
    word_index = np.array([[0, 1, 2]])

    merged = merge_predictions(logits, word_index, [0], [7])
    assert len(merged[0]) == 7


@pytest.mark.parametrize(
    "logits,word_index,doc_map,counts",
    [
        (np.zeros((2, 3)), np.zeros((2, 3), int), [0, 0], [3]),      # 2-D logits
        (np.zeros((1, 3, 4)), np.zeros((1, 2), int), [0], [3]),      # shape clash
        (np.zeros((2, 3, 4)), np.zeros((2, 3), int), [0], [3]),      # short map
    ],
)
def test_merge_rejects_mismatched_shapes(logits, word_index, doc_map, counts):
    with pytest.raises(ValueError):
        merge_predictions(logits, word_index, doc_map, counts)


# --------------------------------------------------------------------------
# Batch plumbing
# --------------------------------------------------------------------------


def test_split_batch_keeps_bookkeeping_away_from_the_model():
    import torch

    batch = {
        "input_ids": torch.zeros(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "bbox": torch.zeros(2, 4, 4, dtype=torch.long),
        "labels": torch.zeros(2, 4, dtype=torch.long),
        "word_index": torch.zeros(2, 4, dtype=torch.long),
        "window_id": torch.zeros(2, dtype=torch.long),
    }
    model_inputs, meta = split_batch(batch)

    assert set(model_inputs) == {"input_ids", "attention_mask", "bbox", "labels"}
    assert set(meta) == {"word_index", "window_id"}


def test_cord_images_rejects_an_unparseable_id():
    doc = make_doc(doc_id="no-index-here")
    with pytest.raises(ValueError, match="expected"):
        CordImages()(doc)


# --------------------------------------------------------------------------
# Encoding - needs layoutlmv3-base
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "microsoft/layoutlmv3-base", add_prefix_space=True
        )
    except Exception as exc:  # network off and nothing cached
        pytest.skip(f"layoutlmv3-base unavailable: {exc}")


def test_first_subword_carries_the_label_rest_are_ignored(tokenizer):
    doc = make_doc(words=["Sub-Total", "75,000"],
                   tags=["B-sub_total.subtotal_price", "I-sub_total.subtotal_price"])
    data = ReceiptDataset([doc], tokenizer, max_length=32)
    item = data[0]

    labelled = [
        (int(w), int(lab))
        for w, lab in zip(item["word_index"], item["labels"])
        if int(w) != NO_WORD
    ]
    assert labelled == [
        (0, LABEL2ID["B-sub_total.subtotal_price"]),
        (1, LABEL2ID["I-sub_total.subtotal_price"]),
    ]
    # Every unlabelled position, including the subwords of "Sub-Total".
    assert all(
        int(lab) == IGNORE_INDEX
        for w, lab in zip(item["word_index"], item["labels"])
        if int(w) == NO_WORD
    )


def test_multi_subword_word_produces_exactly_one_label(tokenizer):
    doc = make_doc(words=["Kembalian"], tags=["B-total.changeprice"])
    data = ReceiptDataset([doc], tokenizer, max_length=32)
    item = data[0]

    real = (item["labels"] != IGNORE_INDEX).sum().item()
    assert real == 1


def test_special_tokens_are_never_labelled(tokenizer):
    doc = make_doc()
    data = ReceiptDataset([doc], tokenizer, max_length=32)
    item = data[0]

    specials = set(tokenizer.all_special_ids)
    for token_id, label in zip(item["input_ids"], item["labels"]):
        if int(token_id) in specials:
            assert int(label) == IGNORE_INDEX


def test_boxes_survive_encoding_on_the_0_1000_grid(tokenizer):
    doc = Document(id="train-0", words=["Nasi"], boxes=[[12, 34, 56, 78]],
                   tags=["B-menu.nm"], width=100, height=100)
    data = ReceiptDataset([doc], tokenizer, max_length=32)
    item = data[0]

    position = int((item["word_index"] == 0).nonzero()[0])
    assert item["bbox"][position].tolist() == [12, 34, 56, 78]


def test_long_document_windows_instead_of_truncating(tokenizer):
    # 400 words cannot fit in 64 subwords, so the tail must survive in a
    # later window rather than being cut.
    words = [f"item{i}" for i in range(400)]
    tags = ["O"] * 399 + ["B-total.total_price"]
    doc = make_doc(words=words, tags=tags)
    data = ReceiptDataset([doc], tokenizer, max_length=64, overlap_words=4)

    assert len(data) > 1
    covered = set()
    for i in range(len(data)):
        covered.update(int(w) for w in data[i]["word_index"] if int(w) != NO_WORD)
    assert 399 in covered, "the total line was truncated away"
    assert covered == set(range(400))


def test_every_window_fits_the_sequence_limit(tokenizer):
    words = [f"item{i}" for i in range(300)]
    doc = make_doc(words=words, tags=["O"] * 300)
    data = ReceiptDataset([doc], tokenizer, max_length=64, overlap_words=8)

    for i in range(len(data)):
        assert data[i]["input_ids"].shape[0] == 64


def test_windows_of_one_document_all_map_back_to_it(tokenizer):
    words = [f"item{i}" for i in range(200)]
    doc = make_doc(words=words, tags=["O"] * 200)
    data = ReceiptDataset([doc], tokenizer, max_length=64, overlap_words=4)

    assert set(data.window_doc_indices) == {0}
    assert data.windows_per_document() == {"train-0": len(data)}


def test_word_longer_than_the_budget_does_not_lose_its_neighbours(tokenizer):
    """OCR garbage must not truncate the rest of the receipt away.

    The longest real word in the corpus is 12 subwords, so this is a safety
    net rather than a live path - but an unbounded word would otherwise either
    hang the packer or swallow the words after it.
    """
    garbage = "qx7#" * 400  # ~1600 subwords, well past the 510 budget
    doc = make_doc(words=["Nasi", garbage, "75,000"],
                   tags=["B-menu.nm", "O", "B-menu.price"])
    data = ReceiptDataset([doc], tokenizer, max_length=512, overlap_words=64)

    covered = set()
    for i in range(len(data)):
        item = data[i]
        assert item["input_ids"].shape[0] == 512
        covered.update(int(w) for w in item["word_index"] if int(w) != NO_WORD)

    assert covered == {0, 1, 2}, "a word was dropped around the oversized token"


def test_merge_round_trips_real_windows(tokenizer):
    """End to end: gold labels -> synthetic logits -> merge -> gold labels.

    Catches an off-by-one between window word offsets and document indices,
    which unit-testing the pieces separately would not.
    """
    words = [f"item{i}" for i in range(120)]
    tags = ["O"] * 120
    tags[0] = "B-menu.nm"
    tags[119] = "B-total.total_price"
    doc = make_doc(words=words, tags=tags)
    data = ReceiptDataset([doc], tokenizer, max_length=64, overlap_words=4)

    n_labels = len(LABEL2ID)
    seq_len = data[0]["input_ids"].shape[0]
    logits = np.zeros((len(data), seq_len, n_labels))
    word_index = np.full((len(data), seq_len), NO_WORD)

    for w in range(len(data)):
        item = data[w]
        word_index[w] = item["word_index"].numpy()
        for position, label in enumerate(item["labels"].tolist()):
            if label != IGNORE_INDEX:
                logits[w, position, label] = 10.0

    merged = merge_predictions(logits, word_index, data.window_doc_indices,
                               data.word_counts)
    assert merged[0] == doc.label_ids
