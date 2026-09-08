"""End-to-end smoke test. Run before spending Colab time:

    pytest -m slow -q

The point is to fail here rather than on the first Colab cell. Everything the
full run touches - encoding, windowing, merging, the training step, metrics,
the CSV - is exercised on 10 documents on CPU.

What this cannot cover is stated rather than papered over: fp16 autocast is a
CUDA path, so on a CPU box the mixed-precision leg SKIPS with a message naming
what is left unverified. A green run on CPU is not evidence that fp16 works.
"""
from __future__ import annotations

import csv
import math

import pytest
import torch

from docextract.dataset import NO_WORD, ReceiptDataset, build_dataset
from docextract.labels import LABELS
from docextract.train import resolve_amp, train


CPU = torch.device("cpu")
CUDA = torch.device("cuda")


# --------------------------------------------------------------------------
# Mixed precision: exercised on GPU, loudly skipped on CPU
# --------------------------------------------------------------------------


def test_amp_is_disabled_on_cpu_with_an_explicit_reason():
    enabled, dtype, reason = resolve_amp(True, CPU)

    assert enabled is False
    assert dtype is None
    assert "UNVERIFIED" in reason, "a silent CPU fallback is the failure mode"
    assert "CUDA" in reason


def test_amp_config_flag_is_honoured():
    enabled, _, reason = resolve_amp(False, CPU)
    assert enabled is False
    assert "config" in reason


def test_amp_enabled_on_cuda():
    enabled, dtype, reason = resolve_amp(True, CUDA)

    assert enabled is True
    assert dtype is torch.float16
    assert "gradient scaling" in reason


@pytest.mark.slow
def test_fp16_autocast_actually_runs():
    """The real thing, on a real GPU. Skips loudly everywhere else."""
    if not torch.cuda.is_available():
        pytest.skip(
            "NO GPU: fp16 autocast and gradient scaling are UNVERIFIED. "
            "The Colab run will be the first time this code path executes. "
            "Run `pytest -m slow` on a CUDA box to cover it."
        )

    device = torch.device("cuda")
    enabled, dtype, _ = resolve_amp(True, device)
    layer = torch.nn.Linear(64, 41).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)

    with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled):
        out = layer(torch.randn(8, 64, device=device))
        assert out.dtype is torch.float16, "autocast did not take effect"
        loss = out.float().pow(2).mean()

    scaler.scale(loss).backward()
    scaler.unscale_(torch.optim.SGD(layer.parameters(), lr=0.0))
    assert all(
        torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None
    ), "gradients went non-finite under fp16"


# --------------------------------------------------------------------------
# Windowing: the most likely silent corruption
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "microsoft/layoutlmv3-base", add_prefix_space=True
        )
    except Exception as exc:
        pytest.skip(f"layoutlmv3-base unavailable: {exc}")


# Small enough that the whole document fits one 512-token window (the control
# case), large enough that a 64-token window forces several.
N_WORDS = 120


def make_long_doc(n_words=N_WORDS):
    from docextract.dataset import Document

    words = [f"item{i}" for i in range(n_words)]
    tags = ["O"] * n_words
    tags[0] = "B-menu.nm"
    tags[n_words // 2] = "B-menu.price"
    tags[-1] = "B-total.total_price"
    boxes = [[10 * (i % 100), 10, 10 * (i % 100) + 5, 20] for i in range(n_words)]
    return Document(id="train-0", words=words, boxes=boxes, tags=tags,
                    width=100, height=200)


def test_multi_window_document_covers_the_same_words_as_a_single_window(tokenizer):
    """A document split into windows must expose exactly the words a
    single-window encoding of the same document does - no duplicates, no gaps.

    This is the assertion that catches windowing corruption. If the word
    offsets drift by one between windows, coverage still looks plausible per
    window while the merged document is silently wrong.
    """
    doc = make_long_doc()

    single = ReceiptDataset([doc], tokenizer, max_length=512, overlap_words=0)
    multi = ReceiptDataset([doc], tokenizer, max_length=64, overlap_words=8)

    assert len(single) == 1, "control case must fit one window"
    assert len(multi) > 1, "test case must actually split"

    def covered(dataset):
        out = []
        for i in range(len(dataset)):
            out += [int(w) for w in dataset[i]["word_index"] if int(w) != NO_WORD]
        return out

    single_words = covered(single)
    multi_words = covered(multi)

    assert sorted(set(multi_words)) == sorted(set(single_words))
    assert set(single_words) == set(range(N_WORDS))
    # Overlaps duplicate words by design; every word must still appear.
    assert len(multi_words) >= len(single_words)


def test_merged_prediction_count_is_identical_across_window_layouts(tokenizer):
    """Merging must return one label per word regardless of how many windows
    the document was cut into."""
    import numpy as np

    from docextract.dataset import merge_predictions

    doc = make_long_doc()
    layouts = {
        "single": ReceiptDataset([doc], tokenizer, max_length=512, overlap_words=0),
        "multi": ReceiptDataset([doc], tokenizer, max_length=64, overlap_words=8),
        "heavy-overlap": ReceiptDataset([doc], tokenizer, max_length=64,
                                        overlap_words=24),
    }

    merged_lengths = {}
    for name, data in layouts.items():
        seq_len = data[0]["input_ids"].shape[0]
        logits = np.zeros((len(data), seq_len, len(LABELS)))
        word_index = np.stack([data[i]["word_index"].numpy() for i in range(len(data))])
        merged = merge_predictions(logits, word_index, data.window_doc_indices,
                                   data.word_counts)
        assert len(merged) == 1
        merged_lengths[name] = len(merged[0])

    assert set(merged_lengths.values()) == {N_WORDS}, merged_lengths


def test_gold_and_prediction_sequences_stay_aligned(tokenizer):
    """-100 must never reach a metric: every merged prediction is a real tag."""
    import numpy as np

    from docextract.dataset import merge_predictions
    from docextract.eval.metrics import ids_to_tags

    doc = make_long_doc()
    data = ReceiptDataset([doc], tokenizer, max_length=64, overlap_words=8)

    seq_len = data[0]["input_ids"].shape[0]
    logits = np.random.RandomState(0).randn(len(data), seq_len, len(LABELS))
    word_index = np.stack([data[i]["word_index"].numpy() for i in range(len(data))])

    merged = merge_predictions(logits, word_index, data.window_doc_indices,
                               data.word_counts)
    tags = ids_to_tags(merged[0])

    assert len(tags) == len(doc.tags)
    assert all(tag in LABELS for tag in tags)
    assert -100 not in merged[0]


# --------------------------------------------------------------------------
# The full run
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_training_runs_end_to_end_on_cpu(tmp_path):
    """10 documents, 2 epochs, CPU. Catches what unit tests cannot: config
    wiring, the optimiser step, the metric plumbing, and the CSV."""
    import yaml
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((repo / "configs" / "base.yaml").read_text(encoding="utf-8"))

    if not (repo / cfg["dataset"]["cache_dir"] / "validation.jsonl").is_file():
        pytest.skip("run scripts/build_dataset.py first")

    cfg["dataset"]["cache_dir"] = str(repo / cfg["dataset"]["cache_dir"])
    cfg["train"]["annotation_counts"] = str(repo / cfg["train"]["annotation_counts"])
    cfg["train"]["output_dir"] = str(tmp_path / "run")
    cfg["train"]["batch_size"] = 2
    cfg["train"]["grad_accum"] = 2
    cfg["train"]["patience"] = 99  # never stop early in a 2-epoch run
    cfg["train"]["wandb"] = {"enabled": False}

    best = train(cfg, limit=10, epochs_override=2)

    assert (best / "config.json").is_file(), "no checkpoint written"
    assert (best / "selection.json").is_file()
    assert (best / "results.md").is_file()

    rows = list(csv.DictReader((tmp_path / "run" / "metrics.csv").open(encoding="utf-8")))
    assert len(rows) == 2, f"expected one CSV row per epoch, got {len(rows)}"

    for row in rows:
        for column, value in row.items():
            if column == "is_best":
                continue
            number = float(value)
            assert not math.isnan(number), f"{column} is NaN"
            assert math.isfinite(number), f"{column} is not finite"

    # The model must actually be learning something, not just executing.
    assert float(rows[1]["train_loss"]) < float(rows[0]["train_loss"]), (
        "training loss did not decrease across two epochs"
    )

    # A denominator restricted to the 10 loaded documents, not the full split.
    ceiling = float(rows[0]["val_ocr_ceiling"])
    assert 0.4 < ceiling < 0.9, (
        f"OCR ceiling {ceiling:.3f} is implausible - the true-recall "
        "denominator is probably the whole split rather than the subset"
    )


@pytest.mark.slow
def test_dataset_builds_from_the_tracked_corpus(tokenizer):
    """The tracked JSONL must encode without the CORD dataset present, which
    is what the Colab text+layout run depends on."""
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    path = repo / "data" / "processed" / "validation.jsonl"
    if not path.is_file():
        pytest.skip("corpus not present")

    data = build_dataset(path, tokenizer, max_length=512, overlap_words=64, limit=10)

    assert len(data.documents) == 10
    assert len(data) >= 10, "windows should never be fewer than documents"
    assert all(len(d.words) == len(d.tags) for d in data.documents)
