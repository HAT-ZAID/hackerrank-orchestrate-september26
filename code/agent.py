"""Deterministic agent substrate (Task 5A).

``run_agent_once`` is deliberately boring: one proposal attempt, one verification,
one kernel-owned row.  There is no loop, no tool-calling model and no LangChain
here; Task 5B builds the real provider on top of this contract.

Failure policy: a provider that raises, returns junk or returns nothing is
recorded in the trace and falls through to the kernel row.  The agent layer can
never fail or change a request it was not explicitly pointed at.
"""
from __future__ import annotations

import hashlib

from code.providers import DecisionProvider, NullProvider, request_id_from_context
from code.verifier import enforce_decision

PROMPT_VERSION = "agent-substrate-v1"


def prompt_hash(version: str = PROMPT_VERSION) -> str:
    """Short stable digest of the prompt/contract version used in traces."""
    return hashlib.sha256(str(version).encode("utf-8")).hexdigest()[:16]


def _candidate_payload(proposal):
    if proposal is None:
        return None
    as_dict = getattr(proposal, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    if isinstance(proposal, dict):
        return dict(proposal)
    return str(proposal)


def _usage_for(provider) -> dict:
    usage = {
        "provider": str(getattr(provider, "name", "none")),
        "model": str(getattr(provider, "model", "none")),
        "proposal_calls": 0 if isinstance(provider, NullProvider) else 1,
        "model_calls": 0 if isinstance(provider, NullProvider) else 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    reported = getattr(provider, "last_usage", None)
    if isinstance(reported, dict):
        for key in ("model_calls", "input_tokens", "output_tokens", "total_tokens"):
            if key in reported:
                try:
                    usage[key] = int(reported[key])
                except (TypeError, ValueError):
                    continue
    return usage


def run_agent_once(
    request_context,
    provider: DecisionProvider | None = None,
    tools=None,
    trace=None,
    *,
    kernel_result=None,
) -> dict:
    """One deterministic agent pass for one request.

    Returns ``{"row", "verification", "trace_path", ...}`` where ``row`` is the
    kernel row (possibly with an agreed candidate rationale as its explanation).
    """
    context = dict(request_context or {})
    request_id = request_id_from_context(context)
    if not request_id:
        raise ValueError("request_context must carry a request_id")

    if kernel_result is None:
        resolver = getattr(tools, "kernel_result_for", None)
        if not callable(resolver):
            raise ValueError(
                "run_agent_once needs a kernel_result or a Toolbox to resolve one"
            )
        kernel_result = resolver(request_id)

    active = provider if provider is not None else NullProvider()
    enabled = trace is not None and bool(getattr(trace, "enabled", False))
    if enabled:
        trace.begin(request_id)
        trace.record(
            "provider",
            {
                "name": str(getattr(active, "name", "none")),
                "model": str(getattr(active, "model", "none")),
                "prompt_version": PROMPT_VERSION,
            },
        )

    proposal = None
    provider_error = None
    try:
        proposal = active.propose(context, tools, trace)
    except Exception as exc:  # a provider can never break the run
        provider_error = f"{type(exc).__name__}: {exc}"
        if enabled:
            trace.record("provider_failure", {"error": provider_error})

    candidate_payload = _candidate_payload(proposal)
    if enabled:
        trace.record(
            "proposal",
            {"candidate": candidate_payload, "provider_error": provider_error},
        )

    outcome = enforce_decision(proposal, kernel_result, request_context=context)
    verification = outcome["verification"]
    final_row = outcome["final_row"]
    if enabled:
        trace.record("verification", verification)
        trace.record("final_row", final_row)

    usage = _usage_for(active)
    trace_path = None
    if enabled:
        trace_path = trace.finish(
            request_id=request_id,
            provider=str(getattr(active, "name", "none")),
            model=str(getattr(active, "model", "none")),
            prompt_version=PROMPT_VERSION,
            prompt_hash=prompt_hash(),
            candidate=candidate_payload,
            verification=verification,
            final_row=final_row,
            usage=usage,
            fallback_reason=verification.get("fallback_reason"),
        )

    return {
        "request_id": request_id,
        "row": final_row,
        "verification": verification,
        "trace_path": trace_path,
        "provider": str(getattr(active, "name", "none")),
        "model": str(getattr(active, "model", "none")),
        "candidate": candidate_payload,
        "provider_error": provider_error,
    }


__all__ = ["PROMPT_VERSION", "prompt_hash", "run_agent_once"]
