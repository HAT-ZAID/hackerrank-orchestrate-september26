"""Decision-provider protocol and offline providers (Task 5A).

This task defines the *shape* of a provider and two deterministic, offline
implementations.  No network adapters, no model SDKs, no prompts: Task 5B adds
real providers on top of this contract.

A provider may only propose.  :func:`code.verifier.enforce_decision` is what
turns a proposal into the final row.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from code.schema import DecisionCandidate, parse_candidate

DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "agent_replay"


def request_id_from_context(request_context) -> str:
    """Extract the request id from either context shape used in this codebase."""
    if not isinstance(request_context, dict):
        return ""
    request_id = request_context.get("request_id")
    if not request_id and isinstance(request_context.get("request"), dict):
        request_id = request_context["request"].get("request_id")
    return str(request_id or "").strip()


@runtime_checkable
class DecisionProvider(Protocol):
    """Minimal provider contract: a name, a model and one proposal attempt."""

    name: str
    model: str

    def propose(self, request_context, tools, trace) -> DecisionCandidate | None:
        ...


class NullProvider:
    """Always declines to propose, so the kernel row is used unchanged."""

    name = "null"
    model = "none"

    def propose(self, request_context, tools, trace) -> DecisionCandidate | None:
        return None


class ReplayProvider:
    """Replay recorded candidates from ``tests/fixtures/agent_replay``.

    Missing or malformed fixtures return ``None`` (kernel fallback) instead of
    raising, so a broken recording can never block the pipeline.
    """

    def __init__(self, fixture_dir=None, *, name: str = "replay", model: str = "replay-fixture"):
        self.name = name
        self.model = model
        self.fixture_dir = (
            Path(fixture_dir) if fixture_dir is not None else DEFAULT_FIXTURE_DIR
        )
        self._fixtures: dict | None = None

    def fixtures(self) -> dict:
        if self._fixtures is None:
            self._fixtures = load_fixtures(self.fixture_dir)
        return self._fixtures

    def propose(self, request_context, tools, trace) -> DecisionCandidate | None:
        request_id = request_id_from_context(request_context)
        if not request_id:
            return None
        entry = self.fixtures().get(request_id)
        if isinstance(entry, dict) and isinstance(entry.get("candidate"), dict):
            entry = entry["candidate"]
        candidate, _errors = parse_candidate(entry)
        return candidate


def load_fixtures(fixture_dir) -> dict:
    """Load ``candidates.json``; any problem yields an empty mapping."""
    path = Path(fixture_dir) / "candidates.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("candidates"), dict):
        payload = payload["candidates"]
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items()}


__all__ = [
    "DecisionProvider",
    "DEFAULT_FIXTURE_DIR",
    "NullProvider",
    "ReplayProvider",
    "load_fixtures",
    "request_id_from_context",
]
