### Results - test split

Word-level true recall (headline): **94.8%** (2209/2329 annotated words)

- OCR ceiling: 100.0% (2329/2329 survived OCR)
- Tagger accuracy on recovered words: 94.8%

seqeval entity F1 over OCR tokens (easier number, excludes what OCR missed): micro **93.0**, macro 84.2

P/R/F1 and support are entity-level over OCR tokens; OCR ceiling and true recall are word-level against every annotated word in CORD. The two halves are different units and do not compare directly.

| field | P | R | F1 | support | OCR ceiling | true recall |
| ----- | - | - | -- | ------- | ----------- | ----------- |
| menu.nm | 91.9 | 93.0 | 92.4 | 328 | 100.0 | 96.1 |
| menu.price | 95.7 | 98.4 | 97.0 | 247 | 100.0 | 98.4 |
| menu.cnt | 95.6 | 98.2 | 96.9 | 221 | 100.0 | 98.3 |
| total.total_price | 91.1 | 92.9 | 92.0 | 99 | 100.0 | 95.7 |
| total.cashprice | 91.3 | 90.0 | 90.6 | 70 | 100.0 | 92.6 |
| menu.unitprice | 95.6 | 95.6 | 95.6 | 68 | 100.0 | 95.7 |
| sub_total.subtotal_price | 96.9 | 94.0 | 95.5 | 67 | 100.0 | 95.9 |
| total.changeprice | 93.2 | 98.2 | 95.7 | 56 | 100.0 | 99.2 |
| sub_total.tax_price | 91.5 | 93.5 | 92.5 | 46 | 100.0 | 97.7 |
| menu.sub.nm | 92.5 | 84.1 | 88.1 | 44 | 100.0 | 81.5 |
| total.menuqty_cnt | 76.3 | 96.7 | 85.3 | 30 | 100.0 | 98.5 |
| menu.sub.price | 88.9 | 80.0 | 84.2 | 20 | 100.0 | 80.0 |
| menu.sub.cnt | 88.2 | 88.2 | 88.2 | 17 | 100.0 | 88.2 |
| total.creditcardprice | 92.9 | 76.5 | 83.9 | 17 | 100.0 | 88.2 |
| sub_total.service_price | 90.9 | 71.4 | 80.0 | 14 | 100.0 | 92.5 |
| sub_total.etc | 90.9 | 76.9 | 83.3 | 13 | 100.0 | 76.7 |
| menu.discountprice | 60.0 | 60.0 | 60.0 | 10 | 100.0 | 96.7 |
| total.menutype_cnt | 100.0 | 25.0 | 40.0 | 8 | 100.0 | 29.4 |
| sub_total.discount_price | 85.7 | 85.7 | 85.7 | 7 | 100.0 | 87.5 |
| total.emoneyprice | 40.0 | 100.0 | 57.1 | 2 | 100.0 | 100.0 |
| **micro** | 92.6 | 93.4 | 93.0 | 1384 | 100.0 | 94.8 |
| **macro** | 87.5 | 84.9 | 84.2 | | | |

Mean loss 0.3085 over 100 windows.

Checkpoint: epoch 18, selected on seqeval micro F1 (validation). The checkpoint maximises validation F1, not the true-recall figure reported as the headline.

