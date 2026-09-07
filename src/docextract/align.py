"""Transfer CORD annotation labels onto OCR tokens.

CORD's valid_line covers only the labelled fields - roughly 25 words on a page
that holds well over a hundred. Training on those words alone produces a model
that never learns to reject anything, which is why so many published CORD
results sit near 96% F1 and collapse on real OCR input. Running our own OCR
gives us the background (O) class the annotations lack.

Matching is two-pass. IoU>=0.3 handles the bulk; a text-based fallback recovers
annotations whose OCR box is offset enough to fail IoU but whose text was read
correctly. Anything still unmatched is a genuine OCR miss and is reported as
label loss, not silently ignored.
"""
from __future__ import annotations

from dataclasses import dataclass

from .labels import bio
from .ocr import Word

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Annotation:
    text: str
    box: Box
    category: str
    group_id: int
    sub_group_id: int


@dataclass
class AlignedDocument:
    words: list[str]
    boxes: list[Box]
    tags: list[str]
    matched: int
    total_annotations: int

    @property
    def coverage(self) -> float:
        if self.total_annotations == 0:
            return 0.0
        return self.matched / self.total_annotations


def quad_to_box(quad: dict) -> Box:
    """CORD stores four corner points; LayoutLMv3 needs an axis-aligned box.

    This discards skew, which is a real limitation on photographed receipts and
    is why the Phase 4 degradation study tests rotation explicitly.
    """
    xs = [quad[f"x{i}"] for i in range(1, 5)]
    ys = [quad[f"y{i}"] for i in range(1, 5)]
    return (min(xs), min(ys), max(xs), max(ys))


def parse_annotations(ground_truth: dict) -> list[Annotation]:
    out: list[Annotation] = []
    for line in ground_truth.get("valid_line", []):
        # Some CORD samples store sub_total/total as lists rather than dicts,
        # so nothing here assumes a fixed shape.
        category = line.get("category")
        if not category:
            continue
        for word in line.get("words", []):
            quad = word.get("quad")
            if not quad:
                continue
            out.append(
                Annotation(
                    text=word.get("text", "").strip(),
                    box=quad_to_box(quad),
                    category=category,
                    group_id=line.get("group_id", -1),
                    sub_group_id=line.get("sub_group_id", 0),
                )
            )
    return out


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _vertical_centre(box: Box) -> float:
    return (box[1] + box[3]) / 2


def align(
    words: list[Word],
    annotations: list[Annotation],
    iou_threshold: float = 0.3,
    fallback_text_match: bool = True,
    fallback_y_tolerance: float = 40.0,
) -> AlignedDocument:
    """Assign a BIO tag to every OCR token."""
    assignment: dict[int, Annotation] = {}   # ocr index -> annotation
    claimed: set[int] = set()                # ocr indices already used
    matched_ann: set[int] = set()

    # Pass 1: greedy IoU, best pairs first so strong matches win.
    candidates = []
    for ai, ann in enumerate(annotations):
        for wi, word in enumerate(words):
            score = iou(ann.box, word.box)
            if score >= iou_threshold:
                candidates.append((score, ai, wi))
    candidates.sort(reverse=True)

    for _, ai, wi in candidates:
        if ai in matched_ann or wi in claimed:
            continue
        assignment[wi] = annotations[ai]
        claimed.add(wi)
        matched_ann.add(ai)

    # Pass 2: same text on roughly the same line.
    if fallback_text_match:
        for ai, ann in enumerate(annotations):
            if ai in matched_ann or not ann.text:
                continue
            best_wi, best_dy = None, fallback_y_tolerance
            for wi, word in enumerate(words):
                if wi in claimed or word.text != ann.text:
                    continue
                dy = abs(_vertical_centre(word.box) - _vertical_centre(ann.box))
                if dy < best_dy:
                    best_wi, best_dy = wi, dy
            if best_wi is not None:
                assignment[best_wi] = ann
                claimed.add(best_wi)
                matched_ann.add(ai)

    # Emit tags in reading order. B- opens whenever the (category, group)
    # changes, so two adjacent items of the same category stay separate.
    tags: list[str] = []
    previous_key = None
    for wi, word in enumerate(words):
        ann = assignment.get(wi)
        if ann is None:
            tags.append("O")
            previous_key = None
            continue
        key = (ann.category, ann.group_id, ann.sub_group_id)
        tags.append(bio(ann.category, first=key != previous_key))
        previous_key = key if bio(ann.category, True) != "O" else None

    return AlignedDocument(
        words=[w.text for w in words],
        boxes=[w.box for w in words],
        tags=tags,
        matched=len(matched_ann),
        total_annotations=len(annotations),
    )