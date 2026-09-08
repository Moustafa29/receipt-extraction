### Results - test split

Word-level true recall (headline): **49.8%** (1161/2329 annotated words)

- OCR ceiling: 62.4% (1454/2329 survived OCR)
- Tagger accuracy on recovered words: 79.8%

seqeval entity F1 over OCR tokens (easier number, excludes what OCR missed): micro **73.0**, macro 52.8

| field | P | R | F1 | support | true recall |
| ----- | - | - | -- | ------- | ----------- |
| menu.nm | 75.8 | 80.2 | 77.9 | 242 | 63.3 |
| menu.price | 83.7 | 91.1 | 87.3 | 169 | 62.5 |
| menu.cnt | 79.8 | 67.6 | 73.2 | 105 | 32.3 |
| total.total_price | 76.3 | 78.0 | 77.2 | 91 | 51.2 |
| total.cashprice | 67.2 | 60.0 | 63.4 | 65 | 41.2 |
| sub_total.subtotal_price | 72.7 | 81.4 | 76.8 | 59 | 57.2 |
| total.changeprice | 67.3 | 78.7 | 72.5 | 47 | 48.3 |
| menu.unitprice | 83.3 | 88.9 | 86.0 | 45 | 58.0 |
| sub_total.tax_price | 57.1 | 53.3 | 55.2 | 45 | 43.8 |
| menu.sub.nm | 53.1 | 44.7 | 48.6 | 38 | 37.0 |
| total.menuqty_cnt | 34.5 | 38.5 | 36.4 | 26 | 25.4 |
| total.creditcardprice | 40.0 | 46.2 | 42.9 | 13 | 31.4 |
| sub_total.etc | 50.0 | 25.0 | 33.3 | 12 | 13.3 |
| sub_total.service_price | 40.0 | 36.4 | 38.1 | 11 | 47.5 |
| menu.discountprice | 70.0 | 77.8 | 73.7 | 9 | 46.7 |
| menu.sub.cnt | 77.8 | 77.8 | 77.8 | 9 | 41.2 |
| sub_total.discount_price | 40.0 | 33.3 | 36.4 | 6 | 18.8 |
| total.menutype_cnt | 0.0 | 0.0 | 0.0 | 6 | 0.0 |
| menu.sub.price | 0.0 | 0.0 | 0.0 | 3 | 0.0 |
| total.emoneyprice | 0.0 | 0.0 | 0.0 | 2 | 0.0 |
| **micro** | 72.9 | 73.2 | 73.0 | 1003 | 49.8 |
| **macro** | 53.4 | 52.9 | 52.8 | | |

Mean loss 0.1886 over 110 windows.

Checkpoint: epoch 19, selected on seqeval micro F1 (validation). The checkpoint maximises validation F1, not the true-recall figure reported as the headline.

