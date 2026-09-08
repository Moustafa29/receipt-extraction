"""Metric tests. Run with: pytest -q

The two numbers must not be allowed to drift into agreeing with each other.
seqeval scores what OCR found; true recall scores against the whole receipt.
Several tests exist purely to pin the gap between them open.
"""
from __future__ import annotations

import json

import pytest

from docextract.eval.metrics import (
    count_annotated_words,
    entity_metrics,
    ids_to_tags,
    load_annotation_counts,
    results_table,
    true_recall,
)
from docextract.labels import LABEL2ID


# --------------------------------------------------------------------------
# Entity-level scoring
# --------------------------------------------------------------------------


def test_perfect_prediction_scores_one():
    gold = [["B-menu.nm", "I-menu.nm", "O", "B-menu.price"]]
    metrics = entity_metrics(gold, gold)

    assert metrics.micro_f1 == 1.0
    assert metrics.macro_f1 == 1.0


def test_partial_span_scores_zero_for_that_entity():
    # Entity level: getting two of three words right is not partial credit.
    gold = [["B-menu.nm", "I-menu.nm", "I-menu.nm"]]
    pred = [["B-menu.nm", "I-menu.nm", "O"]]
    metrics = entity_metrics(gold, pred)

    assert metrics.micro_f1 == 0.0


def test_support_counts_entities_not_tokens():
    # Three words, one entity.
    gold = [["B-menu.nm", "I-menu.nm", "I-menu.nm"]]
    metrics = entity_metrics(gold, gold)

    assert metrics.field("menu.nm").support == 1


def test_two_adjacent_entities_are_counted_separately():
    gold = [["B-menu.nm", "B-menu.nm"]]
    metrics = entity_metrics(gold, gold)

    assert metrics.field("menu.nm").support == 2


def test_strict_mode_rejects_an_entity_that_opens_with_i():
    gold = [["B-menu.nm", "O"]]
    pred = [["I-menu.nm", "O"]]

    strict = entity_metrics(gold, pred, strict=True)
    lenient = entity_metrics(gold, pred, strict=False)

    assert strict.micro_f1 == 0.0
    # The lenient default treats the stray I- as a valid entity start, which
    # is exactly the inflation strict mode exists to prevent.
    assert lenient.micro_f1 == 1.0


def test_predicting_all_outside_scores_zero_not_high():
    # 88.4% of tokens are O, so token accuracy would read ~0.88 here.
    gold = [["B-menu.nm", "O", "O", "O", "O", "O", "O", "O", "B-menu.price"]]
    pred = [["O"] * 9]
    metrics = entity_metrics(gold, pred)

    assert metrics.micro_f1 == 0.0


def test_per_field_scores_are_independent():
    gold = [["B-menu.nm", "B-menu.price"]]
    pred = [["B-menu.nm", "O"]]
    metrics = entity_metrics(gold, pred)

    assert metrics.field("menu.nm").f1 == 1.0
    assert metrics.field("menu.price").f1 == 0.0


def test_entity_metrics_rejects_ragged_input():
    with pytest.raises(ValueError, match="gold tags against"):
        entity_metrics([["O", "O"]], [["O"]])

    with pytest.raises(ValueError, match="gold sequences against"):
        entity_metrics([["O"], ["O"]], [["O"]])


def test_ids_to_tags_round_trips():
    tags = ["O", "B-menu.nm", "I-menu.nm"]
    ids = [LABEL2ID[t] for t in tags]
    assert ids_to_tags(ids) == tags


# --------------------------------------------------------------------------
# True recall
# --------------------------------------------------------------------------


def test_true_recall_counts_ocr_misses_as_misses():
    # OCR found 2 of 10 annotated menu.nm words and both were tagged right.
    gold = [["B-menu.nm", "B-menu.nm"]]
    pred = [["B-menu.nm", "B-menu.nm"]]
    recall = true_recall(gold, pred, {"menu.nm": 10})

    assert recall.correct == 2
    assert recall.ocr_recovered == 2
    assert recall.annotated == 10
    assert recall.recall == pytest.approx(0.2)
    assert recall.ocr_ceiling == pytest.approx(0.2)
    assert recall.tagger_accuracy == pytest.approx(1.0)


def test_true_recall_factors_into_ceiling_times_accuracy():
    gold = [["B-menu.nm", "B-menu.nm", "B-menu.nm", "B-menu.nm"]]
    pred = [["B-menu.nm", "B-menu.nm", "O", "O"]]
    recall = true_recall(gold, pred, {"menu.nm": 8})

    assert recall.recall == pytest.approx(recall.ocr_ceiling * recall.tagger_accuracy)
    assert recall.recall == pytest.approx(2 / 8)


def test_true_recall_requires_an_exact_tag_match():
    # Right field, wrong boundary: not a correctly labelled word.
    gold = [["B-menu.nm", "I-menu.nm"]]
    pred = [["B-menu.nm", "B-menu.nm"]]
    recall = true_recall(gold, pred, {"menu.nm": 2})

    assert recall.correct == 1
    assert recall.recall == pytest.approx(0.5)


def test_background_tokens_are_not_in_the_denominator():
    # O tokens are OCR noise, not annotated words.
    gold = [["O", "O", "B-menu.nm"]]
    pred = [["O", "O", "B-menu.nm"]]
    recall = true_recall(gold, pred, {"menu.nm": 1})

    assert recall.ocr_recovered == 1
    assert recall.recall == pytest.approx(1.0)


def test_false_positive_on_background_does_not_change_true_recall():
    # Recall is a recall: precision damage shows up in seqeval instead.
    gold = [["O", "B-menu.nm"]]
    pred = [["B-menu.price", "B-menu.nm"]]
    recall = true_recall(gold, pred, {"menu.nm": 1, "menu.price": 3})

    assert recall.correct == 1
    assert recall.annotated == 4


def test_per_field_true_recall_uses_the_field_denominator():
    gold = [["B-menu.nm", "B-total.total_price"]]
    pred = [["B-menu.nm", "B-total.total_price"]]
    recall = true_recall(gold, pred, {"menu.nm": 4, "total.total_price": 1})

    assert recall.field_recall("menu.nm") == pytest.approx(0.25)
    assert recall.field_recall("total.total_price") == pytest.approx(1.0)


def test_category_absent_from_the_split_reports_zero_not_error():
    recall = true_recall([["O"]], [["O"]], {"menu.nm": 0})
    assert recall.field_recall("menu.nm") == 0.0
    assert recall.recall == 0.0


def test_recovering_more_words_than_cord_annotates_is_an_error():
    # Guards against pairing predictions with the wrong split's census.
    gold = [["B-menu.nm", "B-menu.nm", "B-menu.nm"]]
    with pytest.raises(ValueError, match="denominator is wrong"):
        true_recall(gold, gold, {"menu.nm": 2})


def test_unknown_category_in_gold_is_an_error():
    gold = [["B-menu.price"]]
    with pytest.raises(ValueError, match="census and the corpus disagree"):
        true_recall(gold, gold, {"menu.nm": 5})


def test_true_recall_rejects_ragged_input():
    with pytest.raises(ValueError, match="gold tags against"):
        true_recall([["O", "O"]], [["O"]], {"menu.nm": 1})


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_results_table_leads_with_true_recall_and_labels_seqeval():
    gold = [["B-menu.nm", "O", "B-menu.price"]]
    pred = [["B-menu.nm", "O", "O"]]
    table = results_table(
        entity_metrics(gold, pred),
        true_recall(gold, pred, {"menu.nm": 4, "menu.price": 4}),
    )

    assert "true recall (headline)" in table
    assert "easier number" in table
    assert "| menu.nm |" in table


# --------------------------------------------------------------------------
# Census
# --------------------------------------------------------------------------


def write_census(tmp_path):
    path = tmp_path / "counts.json"
    path.write_text(json.dumps({
        "splits": {
            "test": {
                "kept_total": 9,
                "kept": {"menu.nm": 6, "menu.price": 3},
                "documents": {
                    "test-0": {"menu.nm": 4, "menu.price": 1},
                    "test-1": {"menu.nm": 2, "menu.price": 2},
                },
            }
        }
    }), encoding="utf-8")
    return path


def test_load_annotation_counts_reads_one_split(tmp_path):
    path = write_census(tmp_path)

    assert load_annotation_counts(path, "test") == {"menu.nm": 6, "menu.price": 3}
    with pytest.raises(KeyError, match="no census for split"):
        load_annotation_counts(path, "train")


def test_subset_denominator_counts_only_the_given_documents(tmp_path):
    """A --limit run must not divide by the whole split.

    Dividing 4 correct words by the full 2167-word denominator reports ~0.2%
    where the real figure is ~62%. That is a number wrong enough to be
    believed.
    """
    path = write_census(tmp_path)
    counts = load_annotation_counts(path, "test", doc_ids=["test-0"])

    assert counts == {"menu.nm": 4, "menu.price": 1}


def test_subset_denominator_keeps_absent_categories_at_zero(tmp_path):
    path = write_census(tmp_path)
    path.write_text(json.dumps({
        "splits": {"test": {
            "kept": {"menu.nm": 6, "total.total_price": 2},
            "documents": {"test-0": {"menu.nm": 4}},
        }}
    }), encoding="utf-8")

    counts = load_annotation_counts(path, "test", doc_ids=["test-0"])
    assert counts == {"menu.nm": 4, "total.total_price": 0}


def test_subset_denominator_rejects_an_unknown_document(tmp_path):
    path = write_census(tmp_path)
    with pytest.raises(KeyError, match="census and corpus disagree"):
        load_annotation_counts(path, "test", doc_ids=["test-99"])


def test_subset_denominator_needs_a_census_with_documents(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "splits": {"test": {"kept": {"menu.nm": 3}}}
    }), encoding="utf-8")

    with pytest.raises(KeyError, match="no per-document counts"):
        load_annotation_counts(path, "test", doc_ids=["test-0"])


def test_tracked_census_subset_is_consistent_with_the_split_total():
    """Summing every document must reproduce the split total."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "docs" / "annotation_counts.json"
    if not path.is_file():
        pytest.skip("run scripts/count_annotations.py to build the census")

    data = json.loads(path.read_text(encoding="utf-8"))
    for split in ("train", "validation", "test"):
        ids = list(data["splits"][split]["documents"])
        subset = load_annotation_counts(path, split, doc_ids=ids)
        assert subset == load_annotation_counts(path, split), split
        assert sum(subset.values()) == data["splits"][split]["kept_total"], split


def test_tracked_census_matches_the_documented_coverage():
    """docs/annotation_counts.json must agree with docs/measurements.md.

    measurements.md reports coverage as 1468/2356 on test and 1471/2186 on
    validation. Those denominators are all-category counts; if the census
    drifts from them, one of the two documents is lying.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "docs" / "annotation_counts.json"
    if not path.is_file():
        pytest.skip("run scripts/count_annotations.py to build the census")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["splits"]["train"]["all_total"] == 19367
    assert data["splits"]["validation"]["all_total"] == 2186
    assert data["splits"]["test"]["all_total"] == 2356


@pytest.mark.slow
def test_count_annotated_words_matches_the_tracked_census():
    """Recompute test-split counts from CORD. Skipped without the dataset."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "docs" / "annotation_counts.json"
    if not path.is_file():
        pytest.skip("no tracked census to compare against")

    try:
        counts = count_annotated_words("naver-clova-ix/cord-v2", "test")
    except Exception as exc:
        pytest.skip(f"CORD unavailable: {exc}")

    expected = load_annotation_counts(path, "test")
    assert counts == expected
