| image | event | currency | tesseract | vlm | agree | notes |
|---|---|---|---|---|---|---|
| image_01 | event_253 | IDR | 4365000 | 4365000 | yes | tesseract: cached |
| image_02 | event_1442 | INR | 100000.00 | 100000 | yes | tesseract: cached |
| image_03 | event_1545 | INR | 41272.00 | 41272.0 | yes | tesseract: cached |
| image_04 | event_1700 | INR | 2854.00 | 2854.00 | yes | — |
| image_05 | event_1786 | INR | 704.05 | 704.05 | yes | — |
| image_06 | event_3051 | INR | 1995.00 | 1995.00 | yes | — |
| image_07 | event_3231 | INR | 8528.10 | 8528.10 | yes | — |
| image_08 | event_4535 | INR | 15339 | 15339 | yes | — |
| image_09 | event_5170 | INR | 723.00 | 723 | yes | — |
| image_10 | event_6033 | INR | 79679.26 | 79679.26 | yes | — |
| image_11 | event_6859 | INR | 3650.00 | 3650.00 | yes | — |
| image_12 | event_7307 | USD | 40.00 | 40.00 | yes | vlm: model amount rejected, text used |
| image_13 | event_7941 | INR | 2298 | 2298 | yes | — |
| image_14 | event_9421 | INR | — | error | n/a | tesseract: no labelled amount; vlm: model did not return JSON |
| image_15 | event_9806 | INR | 9968.00 | 1142513200 | no | vlm: model amount rejected, text used |
| image_16 | event_10521 | INR | 393.22 | 393.22 | yes | — |

**Images compared:** 16
**Resolved an amount:** tesseract 15/16, vlm 15/16
**Exact agreement:** 14/15
**Cache:** hits 3, misses 29, writes 28

## What the two readers cannot do

**`image_14` is handwriting.** It is a carbon-copy pharmacy invoice whose items and
total are written by hand. A person reads the line items as 1500, 724, 796, 550, 303
and 670, with `TOTAL 4543.00` written beside them. Neither Tesseract (a printed-text
engine) nor Qwen3-VL 2B (a small vision model trained mostly on printed documents)
returns a usable amount, so `event_9421` stays unresolved.

That single unreadable receipt is not a cosmetic gap. The forecaster fails closed on an
unknown amount, so `request_101` (the request that depends on this event) returns
`amount_safe_to_pay = 0`, `not_affordable`, with the explanation "financial evidence
for event_9421 remains unresolved". One handwritten receipt decides a whole request.

The honest options are: use a stronger local vision model, add a human confirmation
step in the TUI for unresolved images, or accept the conservative answer. Guessing
4543.00 from a model that cannot read handwriting is exactly the failure mode the
validator exists to prevent.

**`image_15` shows the validator doing its job.** The VLM answered `1142513200` for an
INR 9968 receipt, roughly 114,000x too large. The structured amount disagreed with the
text parse, was rejected, and the deterministic parser's `9968.00` was used. A system
that trusted the model's JSON would have recorded a 1.14 billion rupee pharmacy bill.

**Formatting is not disagreement.** `100000.00` and `100000`, or `41272.00` and
`41272.0`, are the same money. The comparison tool originally compared strings and
counted five false disagreements; it now compares `Decimal` values.

## The third reader: IBM Docling across all 16 images

Docling ran once over every linked image (`--backend docling`, 600 s timeout, temp
dirs on D: because its conversion spills through the system temp directory). Full
table: [`docling-vs-tesseract.md`](docling-vs-tesseract.md).

| Reader | Resolved an amount | Agrees with Tesseract (numeric) |
|---|---|---|
| Tesseract | 15/16 | — |
| Qwen3-VL 2B | 15/16 | 14/15 with Tesseract |
| IBM Docling | 13/16 | 7/13 with Tesseract |

The three-run cache made the cross-checks free: the final table was produced with
**32 hits, 0 misses, 0 writes** — no image was re-read to compare three readers.

Where the readers disagree, Tesseract deserves the benefit of doubt for this
dataset. Its win rate is a property of *these documents*, not a universal ranking:

- The clean, simply-labelled receipts (images 01, 02, 03, 05, 11, 12, 13) agree
  across all three readers.
- On dense, itemised receipts Docling's layout-aware markdown loses the total:
  image_04 `376` vs `2854.00`, image_06 `84` vs `1995.00`, image_08 `6455` vs `15339`,
  image_09 `6320` vs `723.00`, image_15 `48` vs `9968.00`. These look like Docling
  reconciling subtotals instead of the grand total.
- image_07 is a near-miss (`8528` vs `8528.10` — paise lost in Docling's conversion).
- On images 10 and 16 Docling finds no labelled amount where Tesseract reads one
  (`79679.26`, `393.22`); image_14 (handwriting) still defeats every reader.

Practical conclusion kept for the product: `auto` still prefers the VLM, and the
deterministic labelled-amount parser over Tesseract text remains the tie-breaker.
Docling earns its slot as a configured, optional third opinion — chosen with
`BUY_WAIT_OCR_BACKEND=docling` or `BUY_WAIT_OCR_BACKEND=auto` plus
`BUY_WAIT_DOCLING_PYTHON` — and the pipeline's agreement requirement means a bad
Docling read can never silently change an amount.
