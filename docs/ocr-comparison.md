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
