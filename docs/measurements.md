# Measurements

Raw results from the data-pipeline experiments. Scripts in `scripts/`.
Interpretation belongs in the README, not here.

## Dataset

CORD v2 (`naver-clova-ix/cord-v2`), public subset: 800 train / 100 validation /
100 test. Real Indonesian receipts, photographed. Store names and contact
details are blurred out by the dataset's anonymisation, so no `store.*`
categories exist in the release.

`scripts/profile_labels.py`, 800 training documents: 29 categories,
mean 13.7 annotation groups per document, 1.8 words per group.

Top categories by training words: menu.nm 5379, menu.price 2117, menu.cnt 1980,
total.total_price 1695, sub_total.subtotal_price 1185, total.cashprice 1112,
total.changeprice 1044, sub_total.tax_price 1022, menu.unitprice 629,
menu.sub.nm 622, total.menuqty_cnt 513, menu.discountprice 355,
total.creditcardprice 326, sub_total.service_price 274, sub_total.etc 249,
sub_total.discount_price 164, menu.sub.cnt 146, menu.sub.price 126,
total.emoneyprice 115, total.menutype_cnt 105.

Below the 100-word cutoff: menu.num 94, total.total_etc 69, menu.etc 15,
menu.sub.unitprice 14, sub_total.othersvc_price 6, menu.vatyn 6,
void_menu.nm 3, void_menu.price 1, menu.itemsubtotal 1.

`valid_line` covers only annotated fields (~25 words/doc). `dontcare` is
non-empty in 7 of 100 documents, so it is not a usable source of background
tokens.

## OCR configuration sweep

`scripts/sweep_ocr.py`, 10 documents, recall at IoU >= 0.5, no crop.

| preprocess  | psm | recall | text acc | tok/doc |
| ----------- | --- | ------ | -------- | ------- |
| raw         | 3   | 23.0%  | 93.9%    | 20.9    |
| raw         | 11  | 28.6%  | 84.3%    | 69.0    |
| gray2x      | 6   | 30.0%  | 79.4%    | 103.1   |
| gray2x      | 11  | 32.2%  | 80.9%    | 110.6   |
| gray2x+otsu | 11  | 31.7%  | 81.4%    | 114.2   |
| gray3x      | 4   | 31.1%  | 81.1%    | 60.2    |
| gray3x      | 11  | 31.4%  | 80.4%    | 155.8   |

## Matching criterion

`scripts/diagnose_matching.py`, 10 documents, gray2x + psm 11, no crop,
no confidence filter.

| criterion                    | recall |
| ---------------------------- | ------ |
| IoU >= 0.5                   | 32.2%  |
| IoU >= 0.4                   | 34.7%  |
| IoU >= 0.3                   | 38.4%  |
| IoU >= 0.2                   | 43.1%  |
| IoU >= 0.1                   | 46.2%  |
| centre-in-box                | 49.0%  |
| exact text, position ignored | 31.7%  |

Best-IoU distribution per annotated word: 51.0% at [0.00, 0.01), 2.8% at
[0.01, 0.10), 7.8% at [0.10, 0.30), 6.2% at [0.30, 0.50), 32.2% at [0.50, 1.01).

Median box size: annotation 78x34 px, OCR 24x14 px.

Doc 0: 135 annotated words, 21 OCR tokens. Detected text was confined to the
large bold Grand Total line; the item list was not read at all.

## Cropping and contrast

`scripts/test_crop.py` (crop failed — plain Otsu returned the full frame,
because the woven-mat background is as bright as the paper):

| config                | text found |
| --------------------- | ---------- |
| raw frame, psm 11, 2x | 30.5%      |
| CLAHE + 3x, psm 4     | 60.5%      |
| CLAHE + 3x, psm 6     | 62.5%      |
| CLAHE + 3x, psm 11    | 67.8%      |

`scripts/test_crop2.py` (crop fixed via Gaussian blur before Otsu; mean crop
83% of frame), 10 documents:

| psm | text  | IoU 0.5 | IoU 0.4 | IoU 0.3 | IoU 0.2 | tok/doc |
| --- | ----- | ------- | ------- | ------- | ------- | ------- |
| 4   | 52.9% | 53.2%   | 58.5%   | 64.1%   | 70.0%   | 50.9    |
| 6   | 63.3% | 50.7%   | 55.2%   | 60.5%   | 68.6%   | 258.7   |
| 11  | 63.6% | 57.1%   | 65.8%   | 73.4%   | 78.2%   | 238.4   |

Figures: `docs/crop_before.png`, `docs/crop2_after.png`.

## Confidence threshold

`scripts/tune_confidence.py`, 20 documents. Coverage is the share of annotated
words whose label survived alignment; length is LayoutLMv3 subword tokens.

| min_conf | coverage | tok/doc | O share | mean | p50 | p95  | max  | over 512 |
| -------- | -------- | ------- | ------- | ---- | --- | ---- | ---- | -------- |
| 0        | 70.8%    | 266.9   | 91.5%   | 412  | 270 | 1088 | 1875 | 5/20     |
| 10       | 68.3%    | 235.7   | 90.7%   | 362  | 223 | 943  | 1667 | 5/20     |
| 20       | 67.4%    | 206.8   | 89.5%   | 317  | 193 | 818  | 1448 | 4/20     |
| 30       | 64.9%    | 162.4   | 87.2%   | 249  | 142 | 638  | 1080 | 3/20     |
| 50       | 58.5%    | 75.7    | 75.1%   | 116  | 79  | 289  | 332  | 0/20     |
| 70       | 47.0%    | 32.5    | 53.5%   | 53   | 44  | 107  | 128  | 0/20     |

No threshold satisfies both constraints: settings that fit the 512-token window
lose 6+ points of label coverage.

## Noise filter

Dropping OCR tokens containing no alphanumeric character (`is_noise`), at
`min_confidence: 30`, 20 documents:

|        | tok/doc | coverage |
| ------ | ------- | -------- |
| before | 162     | 64.1%    |
| after  | 136     | 63.8%    |

16% fewer tokens for 0.3 points of coverage.

## Final dataset

`scripts/build_dataset.py`, full run. 41 BIO labels from 20 categories.

| split       | docs     | tokens      | tok/doc | label coverage      | O share   |
| ----------- | -------- | ----------- | ------- | ------------------- | --------- |
| train       | 800      | 105,225     | 132     | 64.3% (12448/19367) | 88.3%     |
| validation  | 100      | 14,791      | 148     | 67.3% (1471/2186)   | 90.1%     |
| test        | 100      | 11,789      | 118     | 62.3% (1468/2356)   | 87.7%     |
| **overall** | **1000** | **131,805** | **132** | **64.4%**           | **88.4%** |

Coverage during the training build stabilised by ~200 documents and stayed
between 64% and 65.6% thereafter.
