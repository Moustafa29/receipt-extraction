"""README-against-artifact consistency. Run with: pytest -q

The README quotes numbers that scripts/evaluate.py generates into docs/. Those
two can drift silently: rerun the evaluation, commit the new docs/results.md,
and the README keeps quoting the old figures with nothing to flag it. Since
the headline claims of this project are numbers, a stale one is the most
damaging kind of error it can carry.

These tests parse both and compare.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
RESULTS = REPO / "docs" / "results.md"
BASELINE_ANNOTATIONS = REPO / "docs" / "results_baseline_annotations.md"
BASELINE_OCR = REPO / "docs" / "results_baseline_ocr.md"


def rows(markdown: str) -> dict[str, list[str]]:
    """Field name -> numeric cells, for every data row of a results table."""
    out: dict[str, list[str]] = {}
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("*", "") for c in line.strip("|").split("|")]
        if len(cells) < 4 or not cells[0] or set(cells[0]) <= set("- "):
            continue
        if cells[0] in ("field",):
            continue
        values = [c for c in cells[1:] if c]
        if values and all(re.fullmatch(r"[\d.]+", v) for v in values):
            out[cells[0]] = values
    return out


def readme_section(title: str) -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index(f"\n## {title}\n")
    rest = text[start + 1:]
    end = rest.find("\n## ", 1)
    return rest[:end] if end != -1 else rest


def headline(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    return {
        "true_recall": float(re.search(r"true recall \(headline\): \*\*([\d.]+)%", text).group(1)),
        "ceiling": float(re.search(r"OCR ceiling: ([\d.]+)%", text).group(1)),
        "tagger": float(re.search(r"Tagger accuracy on recovered words: ([\d.]+)%", text).group(1)),
        "micro_f1": float(re.search(r"micro \*\*([\d.]+)\*\*", text).group(1)),
        "macro_f1": float(re.search(r"macro ([\d.]+)", text).group(1)),
    }


@pytest.fixture(scope="module")
def results_available():
    for path in (README, RESULTS, BASELINE_ANNOTATIONS, BASELINE_OCR):
        if not path.is_file():
            pytest.skip(f"{path.name} not present; run scripts/evaluate.py")


def test_readme_results_table_matches_the_generated_one(results_available):
    """Every per-field row in the README must equal docs/results.md."""
    generated = rows(RESULTS.read_text(encoding="utf-8"))
    published = rows(readme_section("Results"))

    assert generated, "no rows parsed from docs/results.md"
    missing = set(generated) - set(published)
    assert not missing, f"README omits fields: {sorted(missing)}"

    for field, values in generated.items():
        assert published[field] == values, (
            f"{field}: README says {published[field]}, "
            f"docs/results.md says {values}"
        )


def test_readme_quotes_the_generated_headline_numbers(results_available):
    section = readme_section("Results")
    figures = headline(RESULTS)

    assert f"{figures['true_recall']}%" in section
    assert f"{figures['ceiling']}%" in section
    assert f"micro {figures['micro_f1']}" in section or \
           f"micro **{figures['micro_f1']}**" in section
    assert str(figures["macro_f1"]) in section


def test_baseline_contrast_table_matches_both_evaluations(results_available):
    """The two-row contrast table is the README's central claim; both rows
    must come from the files they claim to summarise."""
    section = readme_section("The naive baseline")
    annotations = headline(BASELINE_ANNOTATIONS)
    ocr = headline(BASELINE_OCR)

    for figures, label in ((annotations, "annotation"), (ocr, "OCR")):
        assert str(figures["micro_f1"]) in section, f"{label} micro F1 missing"
        assert str(figures["macro_f1"]) in section, f"{label} macro F1 missing"
        assert f"{figures['true_recall']}%" in section, f"{label} true recall missing"


def test_baseline_and_pipeline_share_one_ocr_ceiling(results_available):
    """The three-way comparison rests on both models facing the same ceiling.
    If a rebuild changes the corpus, that claim silently stops holding."""
    assert headline(BASELINE_OCR)["ceiling"] == headline(RESULTS)["ceiling"]


def test_annotation_space_ceiling_is_total(results_available):
    """Annotation input contains every annotated word, so true recall and
    tagger accuracy must coincide there. This is what makes 93.0 the easy
    number."""
    figures = headline(BASELINE_ANNOTATIONS)
    assert figures["ceiling"] == 100.0
    assert figures["true_recall"] == figures["tagger"]


def test_readme_test_count_is_current(results_available):
    """The README states a test count; keep it honest."""
    text = README.read_text(encoding="utf-8")
    claimed = re.search(r"(\d+) tests: (\d+) fast, (\d+) behind the slow marker", text)
    assert claimed, "README no longer states a test count in the expected form"

    total, fast, slow = (int(g) for g in claimed.groups())
    assert total == fast + slow, f"{fast} + {slow} != {total}"
