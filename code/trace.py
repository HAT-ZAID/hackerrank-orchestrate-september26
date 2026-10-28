"""Per-request JSON traces for the agentic layer (Task 5A).

Tracing is **opt-in**: ``TraceRecorder(run_id)`` is disabled and every method is
a no-op until an ``output_dir`` is supplied.  The default pipeline therefore
writes nothing new and stays byte-identical.

Determinism split:

* everything except ``metadata`` and ``duration_ms`` values is content-derived
  and byte-stable for the same inputs (use :func:`deterministic_view`);
* wall-clock timestamps and durations live only in the ``metadata`` block (and
  per-call ``duration_ms``), because they can never be compared across runs.

Secrets are redacted before anything is written: any mapping key matching
``key|token|secret|password|authorization`` has its value replaced.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

from code.schema import TraceEvent, stable_json

TRACE_SCHEMA_VERSION = 1
REDACTED = "[REDACTED]"
_SECRET_KEY = re.compile(r"key|token|secret|password|authorization", re.IGNORECASE)
_UNSTABLE_KEYS = frozenset({"duration_ms"})
_UNSTABLE_BLOCKS = frozenset({"metadata"})
_UNSAFE_FILE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value) -> str:
    name = _UNSAFE_FILE_CHARS.sub("_", str(value)).strip("._") or "unknown"
    return name[:120]


def sanitize(value):
    """Recursively redact secret-like mapping keys (case-insensitive)."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_text = str(key)
            cleaned[key_text] = REDACTED if _SECRET_KEY.search(key_text) else sanitize(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return sanitize(value.as_dict())
    return value


def deterministic_view(payload):
    """Strip wall-clock metadata so two equivalent runs compare byte-for-byte."""
    if isinstance(payload, dict):
        return {
            key: deterministic_view(item)
            for key, item in payload.items()
            if str(key) not in _UNSTABLE_KEYS and str(key) not in _UNSTABLE_BLOCKS
        }
    if isinstance(payload, list):
        return [deterministic_view(item) for item in payload]
    return payload


class TraceRecorder:
    """Collect events for one request and, when enabled, write one JSON file."""

    def __init__(self, run_id, output_dir=None):
        self.run_id = str(run_id)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._request_id: str | None = None
        self._events: list[TraceEvent] = []
        self._tool_calls: list[dict] = []
        self._started: float | None = None

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    def begin(self, request_id) -> None:
        self._request_id = str(request_id)
        self._events = []
        self._tool_calls = []
        self._started = time.perf_counter()
        self.record("request_start", {"request_id": self._request_id})

    def record(self, kind, payload=None):
        if not self.enabled:
            return None
        event = TraceEvent(kind=str(kind), payload=sanitize(dict(payload or {})))
        self._events.append(event)
        return event

    def record_tool_call(self, name, args=None, *, ok, error=None, duration_ms=None):
        if not self.enabled:
            return None
        call = {
            "name": str(name),
            "args": sanitize(dict(args or {})),
            "ok": bool(ok),
            "error": None if error is None else str(error),
        }
        if duration_ms is not None:
            call["duration_ms"] = round(float(duration_ms), 3)
        self._tool_calls.append(call)
        return call

    def finish(
        self,
        *,
        request_id=None,
        provider="none",
        model="none",
        prompt_version="",
        prompt_hash="",
        candidate=None,
        verification=None,
        final_row=None,
        usage=None,
        fallback_reason=None,
    ) -> str | None:
        """Write the trace and return its path (``None`` when disabled)."""
        if not self.enabled:
            return None
        request_id = str(request_id or self._request_id or "unknown")
        duration_ms = None
        if self._started is not None:
            duration_ms = round((time.perf_counter() - self._started) * 1000.0, 3)
        payload = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "request_id": request_id,
            "provider": str(provider),
            "model": str(model),
            "prompt_version": str(prompt_version),
            "prompt_hash": str(prompt_hash),
            "events": [event.as_dict() for event in self._events],
            "tool_calls": list(self._tool_calls),
            "candidate": candidate,
            "verification": verification,
            "final_row": final_row,
            "usage": dict(usage or {}),
            "fallback_reason": fallback_reason,
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "duration_ms": duration_ms,
                "tool_calls": len(self._tool_calls),
            },
        }
        payload = sanitize(payload)
        path = self.output_dir / _safe_name(self.run_id) / f"{_safe_name(request_id)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stable_json(payload) + "\n", encoding="utf-8", newline="\n")
        return str(path)


__all__ = [
    "TraceRecorder",
    "deterministic_view",
    "sanitize",
    "REDACTED",
    "TRACE_SCHEMA_VERSION",
]
