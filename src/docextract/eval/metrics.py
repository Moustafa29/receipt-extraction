"""Entity-level scoring, and the recall figure that does not flatter itself.

Two numbers, and the gap between them is the point of this repository.

seqeval entity F1
    Computed over OCR tokens: the model's predicted BIO sequence against the
    aligned gold sequence. This is the number comparable to published CORD
    results, and it is the easier one - a field OCR never detected is absent
    from both sequences, so failing to extract it costs nothing. Reporting it
    alone would mean scoring the model only on the questions it was asked.

Word-level true recall
    correctly-labelled annotated words / all CORD annotated words, where the
    denominator counts every annotated word on the receipt including the 35.6%
    OCR never found. Those words are unreachable, and they are counted as
    misses because a deployed extractor does miss them.

    It factors exactly:

        true_recall = ocr_ceiling x tagger_accuracy

    ocr_ceiling is the share of annotated words that survived OCR and
    alignment - the hard limit from docs/measurements.md. tagger_accuracy is
    the share of those the model then labelled correctly. A high F1 sitting on
    a low ceiling is what the README argues published CORD numbers conceal.

Denominator
    Kept categories only. Nine categories are mapped to O by design
    (docextract.labels), so counting them as recall misses would penalise the
    model for fields it is built not to predict. That is 27 of 2356 words on
    test, 1.1%; the all-categories figure is reported alongside so the choice
    is visible rather than assumed.

Correctness
    Exact BIO tag match: B-menu.nm scores only against B-menu.nm. Matching on
    category alone would let a model that emits I- everywhere score well while
    producing unusable entity boundaries.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from ..labels import CATEGORIES, ID2LABEL, OUTSIDE

# seqeval aggregate rows, which are not field names.
_AGGREGATE_ROWS = ("micro avg", "macro avg", "weighted avg", "accuracy")


def ids_to_tags(ids: Sequence[int]) -> list[str]:
    return [ID2LABEL[int(i)] for i in ids]


def _category(tag: str) -> str:
    """menu.nm from B-menu.nm. Category names contain dots, prefixes do not."""
    return tag.split("-", 1)[1] if "-" in tag else tag


@dataclass(frozen=True)
class FieldScore:
    field: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EntityMetrics:
    """seqeval entity-level scores over OCR tokens."""

    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_field: tuple[FieldScore, ...]

    def field(self, name: str) -> FieldScore | None:
        for score in self.per_field:
            if score.field == name:
                return score
        return None


def entity_metrics(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    strict: bool = True,
) -> EntityMetrics:
    """Entity-level P/R/F1 per field, plus micro and macro aggregates.

    Entity-level means a menu name spanning three words scores as one item,
    and scores only if every word in the span is right. Token accuracy is
    meaningless on this corpus: 88.4% of tokens are O, so predicting O
    everywhere would read as 88.4% "accurate".

    strict=True with the IOB2 scheme requires an entity to open with B-. The
    lenient default would silently treat a stray I- after O as a valid entity
    start, inflating scores for a model with incoherent boundaries.
    """
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold sequences against {len(pred)} predicted")
    for i, (g, p) in enumerate(zip(gold, pred)):
        if len(g) != len(p):
            raise ValueError(
                f"sequence {i}: {len(g)} gold tags against {len(p)} predicted"
            )

    kwargs = {"mode": "strict", "scheme": IOB2} if strict else {}
    report = classification_report(
        list(gold), list(pred), output_dict=True, zero_division=0, **kwargs
    )

    per_field = tuple(
        FieldScore(
            field=name,
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            f1=float(row["f1-score"]),
            support=int(row["support"]),
        )
        for name, row in sorted(report.items())
        if name not in _AGGREGATE_ROWS
    )

    micro = report.get("micro avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
    macro = report.get("macro avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
    return EntityMetrics(
        micro_precision=float(micro["precision"]),
        micro_recall=float(micro["recall"]),
        micro_f1=float(micro["f1-score"]),
        macro_precision=float(macro["precision"]),
        macro_recall=float(macro["recall"]),
        macro_f1=float(macro["f1-score"]),
        per_field=per_field,
    )


@dataclass(frozen=True)
class TrueRecall:
    """Word-level recall against every annotated word, OCR misses included."""

    correct: int
    ocr_recovered: int
    annotated: int
    per_field: Mapping[str, tuple[int, int, int]] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        """The headline: correctly-labelled words / all annotated words."""
        return self.correct / self.annotated if self.annotated else 0.0

    @property
    def ocr_ceiling(self) -> float:
        """Share of annotated words that survived OCR - the hard limit."""
        return self.ocr_recovered / self.annotated if self.annotated else 0.0

    @property
    def tagger_accuracy(self) -> float:
        """Share of recovered words the model then labelled correctly."""
        return self.correct / self.ocr_recovered if self.ocr_recovered else 0.0

    def field_recall(self, name: str) -> float:
        correct, _, annotated = self.per_field.get(name, (0, 0, 0))
        return correct / annotated if annotated else 0.0

    def field_ceiling(self, name: str) -> float:
        """Share of this field's annotated words that survived OCR.

        The per-field limit. A field can only be extracted as well as it is
        detected, and the fields differ enormously - menu.cnt sits near 47%
        where menu.nm reaches 70%.
        """
        _, recovered, annotated = self.per_field.get(name, (0, 0, 0))
        return recovered / annotated if annotated else 0.0


def true_recall(
    gold: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
    annotated_counts: Mapping[str, int],
) -> TrueRecall:
    """Word-level recall whose denominator includes words OCR never found.

    gold/pred are BIO tags over OCR tokens. annotated_counts maps category to
    the number of annotated words in the split, measured from CORD directly by
    count_annotated_words - independent of what OCR managed to detect, which
    is exactly what makes the denominator honest.

    A gold non-O token is an annotated word that survived OCR. Correct means
    the predicted tag matches exactly.
    """
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold sequences against {len(pred)} predicted")

    recovered: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for i, (gold_doc, pred_doc) in enumerate(zip(gold, pred)):
        if len(gold_doc) != len(pred_doc):
            raise ValueError(
                f"sequence {i}: {len(gold_doc)} gold tags against {len(pred_doc)}"
            )
        for gold_tag, pred_tag in zip(gold_doc, pred_doc):
            if gold_tag == OUTSIDE:
                continue
            category = _category(gold_tag)
            recovered[category] += 1
            if gold_tag == pred_tag:
                correct[category] += 1

    unknown = set(recovered) - set(annotated_counts)
    if unknown:
        raise ValueError(
            f"gold contains categories missing from annotated_counts: "
            f"{sorted(unknown)} - the census and the corpus disagree"
        )
    for category, found in recovered.items():
        if found > annotated_counts[category]:
            raise ValueError(
                f"{category}: OCR recovered {found} words but CORD annotates "
                f"{annotated_counts[category]} - denominator is wrong"
            )

    per_field = {
        category: (correct[category], recovered[category], annotated_counts[category])
        for category in annotated_counts
    }
    return TrueRecall(
        correct=sum(correct.values()),
        ocr_recovered=sum(recovered.values()),
        annotated=sum(annotated_counts.values()),
        per_field=per_field,
    )


def count_annotated_words_by_document(
    hf_name: str,
    split: str,
    kept_only: bool = True,
) -> dict[str, dict[str, int]]:
    """Annotated words per category, per document, straight from CORD.

    Keyed f"{split}-{index}" to match the ids build_dataset.py writes, so a
    subset of documents can be given its own denominator. Reads only the
    ground_truth column, so no image is decoded and the census costs a couple
    of seconds per split rather than an OCR pass.
    """
    from datasets import load_dataset

    from ..align import parse_annotations

    dataset = load_dataset(hf_name, split=split).select_columns(["ground_truth"])
    per_document: dict[str, dict[str, int]] = {}
    for index, row in enumerate(dataset):
        counts: Counter[str] = Counter()
        for annotation in parse_annotations(json.loads(row["ground_truth"])):
            if kept_only and annotation.category not in CATEGORIES:
                continue
            counts[annotation.category] += 1
        per_document[f"{split}-{index}"] = dict(sorted(counts.items()))
    return per_document


def count_annotated_words(
    hf_name: str,
    split: str,
    kept_only: bool = True,
) -> dict[str, int]:
    """Split-level totals per category."""
    counts: Counter[str] = Counter()
    for document in count_annotated_words_by_document(
        hf_name, split, kept_only
    ).values():
        counts.update(document)

    if kept_only:
        # Categories absent from a split still need a zero, or per-field
        # recall would silently omit them.
        for category in CATEGORIES:
            counts.setdefault(category, 0)
    return dict(counts)


def load_annotation_counts(
    path: str | Path,
    split: str,
    doc_ids: Iterable[str] | None = None,
) -> dict[str, int]:
    """Read a census from the JSON written by count_annotations.py.

    doc_ids restricts the denominator to those documents. Without it a subset
    run - a smoke test, a --limit run - would divide by the whole split and
    report a true recall of a few percent when the real figure is around 62%.
    A number that wrong is worse than no number, because it looks like a
    result.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if split not in data["splits"]:
        raise KeyError(f"{path} has no census for split {split!r}")
    entry = data["splits"][split]

    if doc_ids is None:
        return {k: int(v) for k, v in entry["kept"].items()}

    per_document = entry.get("documents")
    if per_document is None:
        raise KeyError(
            f"{path} has no per-document counts; rebuild it with "
            "scripts/count_annotations.py to use a subset denominator"
        )

    doc_ids = list(doc_ids)
    missing = [d for d in doc_ids if d not in per_document]
    if missing:
        raise KeyError(
            f"{path} has no counts for {missing[:3]}"
            f"{'...' if len(missing) > 3 else ''} - census and corpus disagree"
        )

    counts = {category: 0 for category in entry["kept"]}
    for doc_id in doc_ids:
        for category, n in per_document[doc_id].items():
            counts[category] = counts.get(category, 0) + int(n)
    return counts


def results_table(
    entity: EntityMetrics,
    recall: TrueRecall,
    title: str = "Results",
) -> str:
    """Markdown, in the shape the README's results table needs."""
    lines = [
        f"### {title}",
        "",
        f"Word-level true recall (headline): **{100 * recall.recall:.1f}%** "
        f"({recall.correct}/{recall.annotated} annotated words)",
        "",
        f"- OCR ceiling: {100 * recall.ocr_ceiling:.1f}% "
        f"({recall.ocr_recovered}/{recall.annotated} survived OCR)",
        f"- Tagger accuracy on recovered words: "
        f"{100 * recall.tagger_accuracy:.1f}%",
        "",
        f"seqeval entity F1 over OCR tokens (easier number, excludes what OCR "
        f"missed): micro **{100 * entity.micro_f1:.1f}**, "
        f"macro {100 * entity.macro_f1:.1f}",
        "",
        "P/R/F1 and support are entity-level over OCR tokens; OCR ceiling and "
        "true recall are word-level against every annotated word in CORD. The "
        "two halves are different units and do not compare directly.",
        "",
        "| field | P | R | F1 | support | OCR ceiling | true recall |",
        "| ----- | - | - | -- | ------- | ----------- | ----------- |",
    ]
    for score in sorted(entity.per_field, key=lambda s: -s.support):
        lines.append(
            f"| {score.field} | {100 * score.precision:.1f} | "
            f"{100 * score.recall:.1f} | {100 * score.f1:.1f} | "
            f"{score.support} | {100 * recall.field_ceiling(score.field):.1f} | "
            f"{100 * recall.field_recall(score.field):.1f} |"
        )
    lines += [
        f"| **micro** | {100 * entity.micro_precision:.1f} | "
        f"{100 * entity.micro_recall:.1f} | {100 * entity.micro_f1:.1f} | "
        f"{sum(s.support for s in entity.per_field)} | "
        f"{100 * recall.ocr_ceiling:.1f} | {100 * recall.recall:.1f} |",
        f"| **macro** | {100 * entity.macro_precision:.1f} | "
        f"{100 * entity.macro_recall:.1f} | {100 * entity.macro_f1:.1f} | | | |",
    ]
    return "\n".join(lines)
