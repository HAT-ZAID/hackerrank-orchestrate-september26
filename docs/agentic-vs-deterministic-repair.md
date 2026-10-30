# Agentic vs deterministic decisions

Provider: `ollama:gemma4:e2b-it-qat`
Requests compared: **3** (first 3 in dataset order)

## Guardrail

- Machine fields identical: **3/3**
- Verifier rewrote the explanation from the model's rationale: **1/3**
- Agent fallbacks: **0/3**

## Cost

- Model calls: **18** (6.0 per request)
- Tool calls: **18** (6.0 per request)
- Wall clock per request: mean **128.21s**, max **148.80s**

## Per request

| request | machine fields | explanation | rationale | fallback | model calls | tool calls | seconds |
|---|---|---|---|---|---|---|---|
| request_26 | identical | rewritten | candidate | — | 6 | 6 | 112.625 |
| request_27 | identical | kernel text | kernel | — | 5 | 5 | 123.203 |
| request_28 | identical | kernel text | kernel | — | 7 | 7 | 148.797 |

## Reading this

The verifier is the authority: it recomputes the kernel row and keeps its own
machine fields, so a differing amount or status here would be a bug, not a
better answer. The agent's real contribution is the explanation, the evidence
it pulled in, and the trace a human can audit afterwards.
