"""Alignment tests. Run with: pytest -q"""
from __future__ import annotations

from docextract.align import Annotation, align, iou, quad_to_box
from docextract.labels import LABELS, bio
from docextract.ocr import Word, normalise_box


def make_word(text: str, box, conf: int = 90) -> Word:
    return Word(text=text, box=box, confidence=conf)


def make_ann(text: str, box, category: str, group: int = 0) -> Annotation:
    return Annotation(text=text, box=box, category=category,
                      group_id=group, sub_group_id=0)


def test_quad_to_box_handles_rotated_quads():
    quad = {"x1": 10, "y1": 20, "x2": 50, "y2": 18,
            "x3": 52, "y3": 40, "x4": 12, "y4": 42}
    assert quad_to_box(quad) == (10, 18, 52, 42)


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_exact_overlap_transfers_label():
    words = [make_word("Nasi", (10, 10, 60, 30))]
    anns = [make_ann("Nasi", (10, 10, 60, 30), "menu.nm")]
    doc = align(words, anns)
    assert doc.tags == ["B-menu.nm"]
    assert doc.coverage == 1.0


def test_unmatched_ocr_token_becomes_background():
    words = [make_word("Nasi", (10, 10, 60, 30)),
             make_word("TERIMA", (10, 500, 90, 520))]
    anns = [make_ann("Nasi", (10, 10, 60, 30), "menu.nm")]
    doc = align(words, anns)
    assert doc.tags == ["B-menu.nm", "O"]


def test_text_fallback_recovers_offset_box():
    # Same line, box offset enough to fail IoU>=0.3.
    words = [make_word("75,000", (200, 12, 260, 32))]
    anns = [make_ann("75,000", (100, 10, 160, 30), "menu.price")]

    without = align(words, anns, fallback_text_match=False)
    assert without.tags == ["O"]

    with_fallback = align(words, anns, fallback_text_match=True)
    assert with_fallback.tags == ["B-menu.price"]


def test_text_fallback_respects_vertical_tolerance():
    words = [make_word("75,000", (200, 900, 260, 920))]
    anns = [make_ann("75,000", (100, 10, 160, 30), "menu.price")]
    doc = align(words, anns, fallback_text_match=True, fallback_y_tolerance=40)
    assert doc.tags == ["O"]


def test_adjacent_groups_each_open_with_b():
    words = [make_word("Tea", (10, 10, 50, 30)),
             make_word("Kopi", (10, 40, 50, 60))]
    anns = [make_ann("Tea", (10, 10, 50, 30), "menu.nm", group=1),
            make_ann("Kopi", (10, 40, 50, 60), "menu.nm", group=2)]
    doc = align(words, anns)
    assert doc.tags == ["B-menu.nm", "B-menu.nm"]


def test_multiword_group_continues_with_i():
    words = [make_word("Nasi", (10, 10, 50, 30)),
             make_word("Campur", (55, 10, 110, 30))]
    anns = [make_ann("Nasi", (10, 10, 50, 30), "menu.nm", group=1),
            make_ann("Campur", (55, 10, 110, 30), "menu.nm", group=1)]
    doc = align(words, anns)
    assert doc.tags == ["B-menu.nm", "I-menu.nm"]


def test_dropped_category_becomes_background():
    words = [make_word("VAT", (10, 10, 50, 30))]
    anns = [make_ann("VAT", (10, 10, 50, 30), "menu.vatyn")]
    doc = align(words, anns)
    assert doc.tags == ["O"]


def test_label_set_size():
    assert len(LABELS) == 41
    assert LABELS[0] == "O"


def test_bio_returns_outside_for_dropped():
    assert bio("void_menu.price", first=True) == "O"
    assert bio("menu.nm", first=True) == "B-menu.nm"
    assert bio("menu.nm", first=False) == "I-menu.nm"


def test_normalise_box_clamps_to_grid():
    assert normalise_box((0, 0, 100, 100), 100, 100) == [0, 0, 1000, 1000]
    assert normalise_box((-5, -5, 200, 200), 100, 100) == [0, 0, 1000, 1000]


def test_is_noise_drops_texture_fragments():
    from docextract.ocr import is_noise

    for junk in ["|", "~", "==", "---", ".", "«", "///"]:
        assert is_noise(junk), junk


def test_is_noise_keeps_real_fields():
    from docextract.ocr import is_noise

    for real in ["-45", "1", "x", "75,000", "Nasi", "Sub-Total", "PB1"]:
        assert not is_noise(real), real