### Results - test split

Word-level true recall (headline): **46.4%** (1080/2329 annotated words)

- OCR ceiling: 62.4% (1454/2329 survived OCR)
- Tagger accuracy on recovered words: 74.3%

seqeval entity F1 over OCR tokens (easier number, excludes what OCR missed): micro **23.8**, macro 30.6

P/R/F1 and support are entity-level over OCR tokens; OCR ceiling and true recall are word-level against every annotated word in CORD. The two halves are different units and do not compare directly.

| field | P | R | F1 | support | OCR ceiling | true recall |
| ----- | - | - | -- | ------- | ----------- | ----------- |
| menu.nm | 15.1 | 61.6 | 24.3 | 242 | 70.4 | 59.8 |
| menu.price | 52.2 | 85.8 | 64.9 | 169 | 69.0 | 59.3 |
| menu.cnt | 18.0 | 79.0 | 29.4 | 105 | 47.2 | 37.6 |
| total.total_price | 50.0 | 41.8 | 45.5 | 91 | 62.2 | 37.8 |
| total.cashprice | 52.5 | 32.3 | 40.0 | 65 | 60.8 | 31.8 |
| sub_total.subtotal_price | 56.9 | 49.2 | 52.7 | 59 | 64.8 | 47.6 |
| total.changeprice | 32.4 | 25.5 | 28.6 | 47 | 56.7 | 35.8 |
| menu.unitprice | 32.3 | 88.9 | 47.3 | 45 | 65.2 | 58.0 |
| sub_total.tax_price | 38.2 | 28.9 | 32.9 | 45 | 58.6 | 35.9 |
| menu.sub.nm | 1.7 | 47.4 | 3.3 | 38 | 70.4 | 43.2 |
| total.menuqty_cnt | 19.0 | 42.3 | 26.2 | 26 | 49.3 | 31.3 |
| total.creditcardprice | 22.2 | 30.8 | 25.8 | 13 | 51.0 | 37.3 |
| sub_total.etc | 9.1 | 16.7 | 11.8 | 12 | 56.7 | 20.0 |
| sub_total.service_price | 50.0 | 36.4 | 42.1 | 11 | 65.0 | 50.0 |
| menu.discountprice | 42.9 | 33.3 | 37.5 | 9 | 53.3 | 33.3 |
| menu.sub.cnt | 2.0 | 44.4 | 3.8 | 9 | 52.9 | 23.5 |
| sub_total.discount_price | 25.0 | 33.3 | 28.6 | 6 | 56.2 | 37.5 |
| total.menutype_cnt | 0.0 | 0.0 | 0.0 | 6 | 47.1 | 5.9 |
| menu.sub.price | 0.5 | 66.7 | 1.0 | 3 | 15.0 | 10.0 |
| total.emoneyprice | 50.0 | 100.0 | 66.7 | 2 | 100.0 | 100.0 |
| **micro** | 15.0 | 58.0 | 23.8 | 1003 | 62.4 | 46.4 |
| **macro** | 28.5 | 47.2 | 30.6 | | | |

Mean loss 4.2033 over 110 windows.

Checkpoint: epoch 18, selected on seqeval micro F1 (validation). The checkpoint maximises validation F1, not the true-recall figure reported as the headline.

