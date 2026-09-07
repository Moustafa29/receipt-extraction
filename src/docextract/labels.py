"""BIO label schema for CORD.

Category selection is empirical: scripts/profile_labels.py counted words per
category across the 800 training documents. Categories under 100 training words
are mapped to O, because the 100-document test split would give them fewer than
~12 evaluation tokens - an F1 computed on that is noise, not a measurement.

Dropped (train word count): menu.num 94, total.total_etc 69, menu.etc 15,
menu.sub.unitprice 14, sub_total.othersvc_price 6, menu.vatyn 6,
void_menu.nm 3, void_menu.price 1, menu.itemsubtotal 1.

They are mapped to O rather than a shared "other" class: they share no
semantics, so an "other" bucket would be unlearnable and would drag macro-F1
down without carrying information.
"""
from __future__ import annotations

# 20 kept categories, ordered by training frequency.
CATEGORIES: tuple[str, ...] = (
    "menu.nm",
    "menu.price",
    "menu.cnt",
    "total.total_price",
    "sub_total.subtotal_price",
    "total.cashprice",
    "total.changeprice",
    "sub_total.tax_price",
    "menu.unitprice",
    "menu.sub.nm",
    "total.menuqty_cnt",
    "menu.discountprice",
    "total.creditcardprice",
    "sub_total.service_price",
    "sub_total.etc",
    "sub_total.discount_price",
    "menu.sub.cnt",
    "menu.sub.price",
    "total.emoneyprice",
    "total.menutype_cnt",
)

DROPPED: frozenset[str] = frozenset(
    {
        "menu.num",
        "total.total_etc",
        "menu.etc",
        "menu.sub.unitprice",
        "sub_total.othersvc_price",
        "menu.vatyn",
        "void_menu.nm",
        "void_menu.price",
        "menu.itemsubtotal",
    }
)

OUTSIDE = "O"

# 41 labels: O + B-/I- for each of the 20 categories.
LABELS: tuple[str, ...] = (OUTSIDE,) + tuple(
    f"{prefix}-{cat}" for cat in CATEGORIES for prefix in ("B", "I")
)

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}


def is_kept(category: str) -> bool:
    return category in CATEGORIES


def bio(category: str, first: bool) -> str:
    """BIO tag for a word, or O if the category was dropped."""
    if not is_kept(category):
        return OUTSIDE
    return f"{'B' if first else 'I'}-{category}"


def label_id(tag: str) -> int:
    return LABEL2ID[tag]