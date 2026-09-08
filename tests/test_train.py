"""Training-loop unit tests. Run with: pytest -q

The end-to-end smoke run lives separately; this covers the pieces that decide
what gets reported, where a silent error would be indistinguishable from a
real result.
"""
from __future__ import annotations

from docextract.train import CSV_COLUMNS, EpochRecord, curve_verdict


def test_noisy_plateau_is_flagged_as_noise():
    # Bounces ~7 points; the best epoch wins by 1.
    curve = [0.30, 0.41, 0.36, 0.44, 0.38, 0.45, 0.39, 0.46]
    verdict = curve_verdict(curve, max(curve))

    assert "WARNING" in verdict
    assert "selecting noise" in verdict


def test_clean_convergence_is_not_flagged():
    curve = [0.10, 0.30, 0.45, 0.52, 0.53, 0.54, 0.56, 0.62]
    verdict = curve_verdict(curve, max(curve))

    assert "WARNING" not in verdict


def test_opening_climb_does_not_count_as_noise():
    """A run that climbs hard then settles must not be called unstable.

    Measuring the swing across the whole curve would let the early genuine
    improvements masquerade as epoch-to-epoch jitter.
    """
    curve = [0.05, 0.35, 0.60, 0.70, 0.71, 0.71, 0.72, 0.78]
    assert "WARNING" not in curve_verdict(curve, max(curve))


def test_short_runs_refuse_to_judge():
    assert "too few epochs" in curve_verdict([0.1, 0.2], 0.2)


def test_flat_curve_reports_a_zero_margin():
    curve = [0.4, 0.4, 0.4, 0.4]
    verdict = curve_verdict(curve, 0.4)
    assert "0.00 points" in verdict


def test_epoch_record_matches_the_csv_header():
    """A field added to the record but not the header would be dropped."""
    record = EpochRecord(
        epoch=1, train_loss=0.0, val_loss=0.0, val_micro_f1=0.0,
        val_micro_precision=0.0, val_micro_recall=0.0, val_macro_f1=0.0,
        val_true_recall=0.0, val_ocr_ceiling=0.0, val_tagger_accuracy=0.0,
        learning_rate=0.0, epoch_seconds=0.0, is_best=False,
    )
    assert tuple(vars(record)) == CSV_COLUMNS
