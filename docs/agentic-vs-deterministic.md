# Agentic vs deterministic decisions

Provider: `ollama:qwen3-vl:2b-instruct-q4_K_M`
Requests compared: **3** (first 3 in dataset order)

## Guardrail

- Machine fields identical: **3/3**
- Verifier rewrote the explanation from the model's rationale: **0/3**
- Agent fallbacks: **3/3**

## Cost

- Model calls: **18** (6.0 per request)
- Tool calls: **17** (5.7 per request)
- Wall clock per request: mean **74.80s**, max **135.36s**

## Per request

| request | machine fields | explanation | rationale | fallback | model calls | tool calls | seconds |
|---|---|---|---|---|---|---|---|
| request_26 | identical | kernel text | kernel | provider call budget exhausted (6/6 turns) | 6 | 6 | 135.36 |
| request_27 | identical | kernel text | kernel | provider call budget exhausted (6/6 turns) | 6 | 5 | 18.531 |
| request_28 | identical | kernel text | kernel | provider call budget exhausted (6/6 turns) | 6 | 6 | 70.516 |

## Fallback reasons

- `request_26`: provider call budget exhausted (6/6 turns)
- `request_27`: provider call budget exhausted (6/6 turns)
- `request_28`: provider call budget exhausted (6/6 turns)

## Reading this

The verifier is the authority: it recomputes the kernel row and keeps its own
machine fields, so a differing amount or status here would be a bug, not a
better answer. The agent's real contribution is the explanation, the evidence
it pulled in, and the trace a human can audit afterwards.
