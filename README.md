# Receipt Field Extraction

Layout-aware extraction of structured fields from photographed receipts, using
LayoutLMv3 token classification on the CORD dataset.

**The headline result is about the data, not the model: only 64.4% of CORD's
annotated fields survive a realistic OCR pipeline.** That figure is the ceiling
on what any downstream tagger can achieve on photographed receipts. Published
CORD results near 96% F1 are computed on pre-filtered ground-truth boxes — an
easier problem than a deployed system faces, because the model is handed the
answer's location and never has to reject anything.

This repository documents how that number was arrived at, and builds the
pipeline that measures against it honestly.

**Status:** data pipeline complete (Phases 1–2). First training run complete
(20 epochs, Colab T4); test-set evaluation and the naive baseline in progress.

---

## The problem with the standard approach

CORD annotates only the fields it cares about. Its `valid_line` structure covers
roughly 25 words on a receipt that carries well over a hundred — the store
header, date, cashier line, and footer text are simply absent from the
annotations.

Training a token classifier on those words alone means the model never sees a
single background token. It is never asked to decide that something is _not_ a
field. At inference time, when it receives real OCR output containing the whole
page, it labels the store name as a menu item, because rejection was never part
of its training signal.

|                         | annotations only             | OCR + alignment (this repo) |
| ----------------------- | ---------------------------- | --------------------------- |
| Tokens at training time | ~25 pre-filtered field words | ~132 words, whole page      |
| Tokens at inference     | whole page from OCR          | whole page from OCR         |
| Background class        | absent                       | 88.4% of tokens             |
| Train/serve mismatch    | severe                       | none                        |

Running our own OCR and transferring labels onto its output is what supplies the
background class. It also exposes how much supervision OCR loses — which is the
finding above.

---

## OCR is the bottleneck, and it is worse than expected

CORD receipts are photographs, not scans: curved thermal paper, low-contrast
print, shot on patterned surfaces.

![Before and after preprocessing](docs/crop_before.png)

At default settings Tesseract read almost nothing. On the first test document it
returned 21 tokens against 135 annotated words, and the tokens it did return
were mostly the woven placemat being read as text (`|`, `ay`, `s1`). The only
genuine reads were the large bold Grand Total line at the bottom.

Measured across 10 documents with no preprocessing: **51% of annotated words had
zero overlapping detection**, and exact-text recall ignoring position entirely
was 31.7%. The failure was detection, not alignment.

### What fixed it

![After contrast enhancement and cropping](docs/crop2_after.png)

| configuration                 | text recall |
| ----------------------------- | ----------- |
| raw frame, psm 11, 2x upscale | 30.5%       |
| + CLAHE + 3x upscale, psm 4   | 60.5%       |
| + CLAHE + 3x upscale, psm 6   | 62.5%       |
| + CLAHE + 3x upscale, psm 11  | 67.8%       |

Contrast enhancement plus upscaling more than doubled recall. Page segmentation
mode 11 (sparse text) wins because the standard layout modes try to find columns
in the background texture.

Cropping needed a second attempt. A plain Otsu threshold returns the entire
frame, because the placemat's white strands are as bright as the paper. Blurring
first solves it: the background is high-frequency and averages to mid-grey under
a large Gaussian kernel, while the receipt stays uniformly bright.

With the crop working, at psm 11:

| matching criterion | recall |
| ------------------ | ------ |
| IoU ≥ 0.5          | 57.1%  |
| IoU ≥ 0.4          | 65.8%  |
| IoU ≥ 0.3          | 73.4%  |
| IoU ≥ 0.2          | 78.2%  |

IoU ≥ 0.3 is used. The 0.5 convention borrowed from object detection is too
strict for word matching: receipt words are around 78×34 px, and Tesseract's
boxes are looser vertically than CORD's tight glyph quads, so a correct match on
the right word routinely scores below 0.5.

---

## Sequence length: no threshold works

Reading the whole page produces long documents. LayoutLMv3 accepts 512 subword
tokens. Raising Tesseract's confidence threshold shortens sequences but discards
faint genuine text along with noise:

| min_confidence | label coverage | tokens/doc | p95 subwords | over 512 |
| -------------- | -------------- | ---------- | ------------ | -------- |
| 0              | 70.8%          | 267        | 1088         | 5/20     |
| 10             | 68.3%          | 236        | 943          | 5/20     |
| 20             | 67.4%          | 207        | 818          | 4/20     |
| 30             | 64.9%          | 162        | 638          | 3/20     |
| 50             | 58.5%          | 76         | 289          | 0/20     |
| 70             | 47.0%          | 33         | 107          | 0/20     |

Every setting that fits the window costs at least six points of label coverage.
The resolution is two changes rather than one threshold:

1. **Filter non-alphanumeric tokens.** Background texture reads as `|`, `~`,
   `==`. Dropping them removed 16% of tokens at a cost of 0.3 points of
   coverage — a far better trade than raising confidence. Real fields such as
   `-45` and `1 x` contain alphanumerics and survive.
2. **Sliding windows** with overlap and prediction merging for documents that
   are still too long. Truncation is not acceptable: it cuts the bottom of the
   receipt, where `total.total_price` lives.

`min_confidence` stays at 30 — the best coverage-per-token point in the table.

---

## Label schema

29 CORD categories reduced to 20, giving 41 BIO labels.

Nine categories were mapped to `O` because they fall below 100 training words:
`menu.num` (94), `total.total_etc` (69), `menu.etc` (15), `menu.sub.unitprice`
(14), `sub_total.othersvc_price` (6), `menu.vatyn` (6), `void_menu.nm` (3),
`void_menu.price` (1), `menu.itemsubtotal` (1).

The binding constraint is the 100-document test split, one eighth the size of
train. A category with 94 training words yields roughly 12 test tokens; an F1
computed over 12 tokens is noise, and the single-document categories may not
appear in test at all, giving undefined rather than low F1.

They map to `O` rather than a shared `other` class deliberately. `void_menu.price`
and `menu.vatyn` have nothing in common, so an `other` bucket would be
unlearnable and would drag macro-F1 down without carrying information.
Background is the honest label.

Sub-item categories (`menu.sub.nm`, `menu.sub.cnt`, `menu.sub.price`) are kept
distinct rather than merged into `menu.*`, because the hierarchy is real and
merging would corrupt line-item reconstruction.

---

## Final dataset

| split       | docs      | tokens      | tokens/doc | label coverage | background |
| ----------- | --------- | ----------- | ---------- | -------------- | ---------- |
| train       | 800       | 105,225     | 132        | 64.3%          | 88.3%      |
| validation  | 100       | 14,791      | 148        | 67.3%          | 90.1%      |
| test        | 100       | 11,789      | 118        | 62.3%          | 87.7%      |
| **overall** | **1,000** | **131,805** | **132**    | **64.4%**      | **88.4%**  |

Coverage stabilised by around 200 documents and stayed between 64% and 65.6%
thereafter, so 64.4% is a property of the pipeline rather than a sampling
artefact.

---

## Training

LayoutLMv3-base, text and layout only — the visual branch is left off for this
run and revisited as an ablation. 20 epochs on a Colab T4: batch 8 with
gradient accumulation to an effective 32, fp16, early stopping on validation
seqeval micro F1 with patience 5. Sequences past 512 subwords are windowed
rather than truncated, so 7.4% of training documents contribute more than one
window and long receipts carry proportionally more gradient — a bias toward
exactly the receipts where `total.total_price` is hardest to reach.

Best epoch 19: validation micro F1 **73.44**, macro F1 **56.11**, word-level
true recall **54.45%**. The per-epoch curve is `outputs/*/metrics.csv`.

Three things about that number that a single figure would hide.

**The stopping criterion is not the headline number.** The checkpoint maximises
micro F1; true recall is what the results table reports. Selecting on true
recall would favour a model that over-predicts, because recall alone carries no
precision penalty. So the saved checkpoint is not the one that maximises the
number quoted as the result, and `best/selection.json` records that.

**The selected epoch is not a clear peak.** The run reports a median
epoch-to-epoch F1 swing of 0.42 points against a winning margin of 0.10 — the
best epoch beats the runner-up by less than the curve's routine variation. On a
100-document validation split a one-point F1 move is roughly 20 entities, so
epoch 19 is one draw from a plateau rather than a maximum.

**Micro plateaued from epoch 15, but macro F1 and true recall were still
climbing at epoch 20.** Micro is dominated by `menu.nm`, which converges early;
the rare categories that drive macro had not settled when the epoch budget ran
out. Stopping on micro may therefore have cut the run short for precisely the
fields with the least training signal. A 40-epoch run would settle it and has
not been run.

---

## What doesn't work

**A third of the supervision is lost.** 35.6% of annotated fields have no
corresponding OCR token, so the model is trained to call them `O` when they are
not. This is label noise and it caps achievable F1 below what a clean pipeline
would reach.

**The test split is the worst of the three.** Coverage is 62.3% on test against
67.3% on validation. Reported test metrics carry more label noise than
validation metrics suggest.

**Skew is discarded.** CORD stores four corner points per word because receipts
are photographed at an angle. LayoutLMv3 requires axis-aligned rectangles, so
the conversion takes the min/max bounding box and throws the rotation away.
Curved receipts lose the most.

**Cropping is partial.** The blur-then-threshold crop recovers about 83% of the
frame on average — it trims the margins but does not isolate the receipt
precisely. Background texture still reaches the OCR stage and is handled by the
noise filter rather than removed at source.

**Indonesian receipts only.** CORD is Indonesian, and the dataset's
anonymisation blurs store names and contact details, so no `store.*` categories
exist. Nothing here supports a claim about other languages or about invoices and
forms, which have different layouts.

**Tesseract is a poor fit and was kept anyway.** It was built for scanned
documents, not photographs of curved thermal paper. A scene-text engine such as
PaddleOCR or EasyOCR would likely raise the ceiling. That comparison has not
been run.

---

## Repository layout

```
configs/base.yaml           pipeline configuration
src/docextract/
  labels.py                 BIO schema, category selection
  ocr.py                    preprocessing, Tesseract, box normalisation
  align.py                  IoU + text-fallback label transfer
scripts/
  profile_labels.py         category distribution
  sweep_ocr.py              20-config preprocessing sweep
  diagnose_matching.py      detection vs alignment diagnosis
  test_crop.py              first crop attempt (failed)
  test_crop2.py             blur-then-Otsu crop
  tune_confidence.py        confidence/length trade
  build_dataset.py          full pipeline -> JSONL
tests/test_align.py         15 unit tests
docs/measurements.md        every number, raw
```

## Setup

Requires Python 3.11 and Tesseract with the Indonesian language pack.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Install Tesseract, then add `ind.traineddata` to its `tessdata` directory and
set `ocr.tesseract_exe` in `configs/base.yaml` to the binary's path.

## Running

```bash
pytest -q                                    # 15 unit tests, no GPU
python scripts/build_dataset.py --limit 20   # smoke run
python scripts/build_dataset.py              # full build, 1-2 hours on CPU
```

The full build writes `data/processed/{train,validation,test}.jsonl` and prints
the coverage report above.

## Data

CORD v2 (Park et al., _CORD: A Consolidated Receipt Dataset for Post-OCR
Parsing_, Document Intelligence Workshop at NeurIPS 2019). Real Indonesian
receipts, anonymised. Downloaded automatically from Hugging Face.

The BIO tagging approach follows Hwang et al., _Post-OCR parsing: building
simple and robust parser via BIO tagging_, from the same group.
