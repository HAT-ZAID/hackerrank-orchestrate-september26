"""Forecast policy for the daily ledger (Task 3).

The oracle answers one question: *how much cash can this user safely commit on
the request date without dropping below ``minimum_balance_to_keep`` anywhere in
the 90-day horizon?*  This module supplies the only forward-looking cash rows the
ledger may contain, under an explicit, conservative policy:

* only committed/essential expense categories are projected (:data:`FORECASTABLE_EXPENSE_CATEGORIES`);
  discretionary spending (:data:`DISCRETIONARY_CATEGORIES`) is never invented
  forward;
* an expense amount is the **median** of its eligible history, so a one-off
  outlier cannot dominate the projection;
* at most **one** regular salary stream is projected, chosen by cadence and
  amount stability, and never more than two occurrences inside the horizon;
* gig/platform/freelance/commission/bonus/arrears/prize/seasonal/temporary/
  final/prorated/secondary streams are never projected, and an ended employment
  stops projection entirely;
* typed evidence expense facts (percent increase, explicit new amount, ended
  category) move only the forecast rows on or after their effective date.  An
  ambiguous fact leaves the forecast unchanged and is reported as a warning.

Everything is deterministic: no randomness, no live data, no LLM.  Entries are
sorted by date, category, kind and source.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from code.fx import resolve_rate

HORIZON_DAYS = 90
DEFAULT_INTERVAL_DAYS = 30
MAX_INCOME_OCCURRENCES = 2

# Committed / essential categories the ledger may project forward.
FORECASTABLE_EXPENSE_CATEGORIES = frozenset(
    {
        "rent",
        "utilities",
        "groceries",
        "transport",
        "healthcare",
        "education",
        "family_support",
        "housing",
        "work_expense",
        "debt_repayment",
        "cloud_storage",
        "delivery_membership",
        "gym",
        "music_subscription",
        "streaming",
    }
)

# Lifestyle spending: never extrapolated into the future, even when it recurs.
DISCRETIONARY_CATEGORIES = frozenset({"dining", "entertainment", "shopping"})

# Essential spending that is variable rather than a fixed recurring bill.  The
# recurring forecast can miss part of this spend; the uncovered remainder is
# reserved conservatively over the horizon.
ESSENTIAL_VARIABLE_CATEGORIES = frozenset(
    {"groceries", "transport", "utilities", "healthcare", "education", "family_support"}
)
VARIABLE_RESERVE_BUCKET_DAYS = 30
MIN_VARIABLE_RESERVE_MONTHS = 2

# Event types that can carry recurring committed expense history.  The oracle's
# historical filter is preserved (Task 3 brief: keep event filtering); a
# `debt_payment` row is a cash event but not a recurring *expense* history row.
EXPENSE_EVENT_TYPES = frozenset({"expense", "subscription"})

# Only a regular salary can be projected as income.
PRIMARY_INCOME_CATEGORIES = frozenset({"salary"})

# Description tokens that disqualify an income stream from projection.  A
# stream is one (category, description) history, so "Payroll credit" stays
# eligible while "Weekly app earnings" or "Final employer payroll" never does.
EXCLUDED_INCOME_DESCRIPTION_TOKENS = (
    "gig",
    "platform",
    "marketplace",
    "app earnings",
    "freelance",
    "contract",
    "consulting",
    "invoice",
    "retainer",
    "commission",
    "bonus",
    "arrears",
    "prize",
    "seasonal",
    "temporary",
    "final",
    "prorated",
    "previous employer",
    "new employer",
    # Secondary household income is explicitly not projected.
    "second",
    "secondary",
)

# User-level evidence that an income source has ended.
INCOME_END_FLAGS = frozenset(
    {"employment_ended", "contract_ended", "household_ended", "partial_income_ended"}
)

# A salary history whose most recent row is terminal means the stream ended:
# earlier rows with a generic description must not be projected past it.
TERMINAL_INCOME_DESCRIPTION_TOKENS = (
    "final",
    "previous employer",
    "employment ended",
    "contract ended",
    "household ended",
)

_EXPENSE_END_FACT_KINDS = frozenset({"ended", "removed", "stop", "stopped", "cancel", "cancelled"})
_EXPENSE_AMOUNT_FACT_KINDS = frozenset({"new_recurring", "amount_change", "amount", "replacement"})
_EXPENSE_INCREASE_FACT_KINDS = frozenset({"increase", "percent_increase"})

_EXCLUDED_HISTORY_STATUSES = frozenset({"cancelled", "canceled", "failed", "unrealized"})


@dataclass(frozen=True)
class ForecastEntry:
    """One projected cash movement, signed in the home currency."""

    date: date
    amount: Decimal
    category: str
    kind: str  # "expense" | "income"
    source: str  # "history" | "evidence"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "amount": format(self.amount, "f"),
            "category": self.category,
            "kind": self.kind,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass
class ForecastResult:
    """Projected entries plus the warnings and stream choice behind them."""

    entries: list[ForecastEntry] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    income_stream: str | None = None


def _text(value) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and value != value:  # NaN
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _norm(value) -> str:
    return _text(value).lower()


def _as_date(value) -> date | None:
    if isinstance(value, date) and not isinstance(value, (int, float)):
        return value
    text = _text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _decimal(value) -> Decimal | None:
    text = _text(value)
    if not text or text.lower() == "nan":
        return None
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if number.is_finite() else None


def _median_int(values: list[int]) -> int:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def _median_decimal(values: list[Decimal]) -> Decimal:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _cv(values: list[Decimal]) -> Decimal:
    """Coefficient of variation, deterministic and outlier-sensitive."""
    count = len(values)
    if count < 2:
        return Decimal(1)
    mean = sum(values, Decimal(0)) / Decimal(count)
    if mean == 0:
        return Decimal(0) if all(value == 0 for value in values) else Decimal(1)
    variance = sum(((value - mean) ** 2 for value in values), Decimal(0)) / Decimal(count - 1)
    return variance.sqrt() / abs(mean)


def _normalized_flags(recurrence_flags) -> set[tuple[str, str, str]]:
    flags: set[tuple[str, str, str]] = set()
    for flag in recurrence_flags or ():
        try:
            user_id, category, event_type = flag
        except (TypeError, ValueError):
            continue
        flags.add((_text(user_id), _norm(category), _norm(event_type)))
    return flags


def _event_day(event: dict) -> date | None:
    return _as_date(event.get("settlement_date") or event.get("event_date"))


def _counts_as_history(event: dict, *, income: bool) -> bool:
    status = _norm(event.get("status"))
    direction = _norm(event.get("direction"))
    if direction == "non_cash" or status in _EXCLUDED_HISTORY_STATUSES:
        return False
    if status not in {"settled", "scheduled", "pending"}:
        return False
    if direction != ("credit" if income else "debit"):
        return False
    if status == "pending" and not income:
        # A pending debit is reserved cash, so it is valid expense evidence.
        return True
    if status == "pending" and income:
        return False
    return True


def _warning(flag: str, source: str, detail: str, category: str = "") -> dict:
    payload = {"flag": flag, "detail": detail}
    if source:
        payload["source_message_id"] = source
    if category:
        payload["category"] = category
    return payload


def _fact_value(fact, key: str):
    if isinstance(fact, dict):
        return fact.get(key)
    return getattr(fact, key, None)


# ---------------------------------------------------------------------------
# Expense projection
# ---------------------------------------------------------------------------


def _expense_history(
    events,
    req_date: date,
    *,
    user_id: str,
    home_currency: str,
    fx_rows,
    flags: set[tuple[str, str, str]],
    warnings: list[dict],
) -> dict[str, dict]:
    """Per-category (dates, home-currency magnitudes) history before the request."""
    history: dict[str, dict] = {}
    for event in events:
        category = _norm(event.get("category"))
        if category not in FORECASTABLE_EXPENSE_CATEGORIES:
            continue
        if category in DISCRETIONARY_CATEGORIES:
            continue
        event_type = _norm(event.get("event_type"))
        if event_type not in EXPENSE_EVENT_TYPES:
            continue
        if (user_id, category, event_type) not in flags:
            continue
        if not _counts_as_history(event, income=False):
            continue
        event_day = _event_day(event)
        if event_day is None or event_day >= req_date:
            continue
        # Cadence evidence comes from every countable row of the category: a
        # row whose rate is unknown still happened on that date.
        bucket = history.setdefault(category, {"dates": [], "amounts": []})
        bucket["dates"].append(event_day)
        amount = _decimal(event.get("amount"))
        if amount is None or amount <= 0:
            continue
        rate = resolve_rate(
            fx_rows, _text(event.get("currency")), home_currency, event_day.isoformat()
        )["rate"]
        if rate is None:
            # Never invent a rate; skip the observation rather than pollute the
            # median with raw foreign units.
            warnings.append(
                _warning(
                    "FORECAST_HISTORY_FX_UNKNOWN",
                    "",
                    f"skipped {category} observation on {event_day.isoformat()}",
                    category,
                )
            )
            continue
        bucket["amounts"].append(abs(amount * rate))
    return history


def _expense_forecast(
    events,
    req_date: date,
    window_end: date,
    *,
    user_id: str,
    home_currency: str,
    fx_rows,
    recurrence_flags,
    suppressed: set[str],
    protected: set[str],
    warnings: list[dict],
) -> list[ForecastEntry]:
    flags = _normalized_flags(recurrence_flags)
    history = _expense_history(
        events,
        req_date,
        user_id=user_id,
        home_currency=home_currency,
        fx_rows=fx_rows,
        flags=flags,
        warnings=warnings,
    )
    # Dates already carrying a countable row for the category: never double count.
    existing: set[tuple[str, date]] = set()
    for event in events:
        category = _norm(event.get("category"))
        event_day = _event_day(event)
        if event_day is None or not (req_date <= event_day <= window_end):
            continue
        if _norm(event.get("direction")) == "non_cash":
            continue
        if _norm(event.get("status")) in _EXCLUDED_HISTORY_STATUSES:
            continue
        existing.add((category, event_day))

    entries: list[ForecastEntry] = []
    for category in sorted(history):
        bucket = history[category]
        dates: list[date] = bucket["dates"]
        # Recurrence needs at least two dated observations, and the amount
        # projection needs at least one convertible amount.
        if len(dates) < 2 or not bucket["amounts"]:
            continue
        if category in suppressed:
            if category in protected:
                warnings.append(
                    _warning(
                        "PROTECTED_CATEGORY_NOT_SUPPRESSED",
                        "",
                        f"{category} is protected; the suppression was ignored",
                        category,
                    )
                )
            else:
                # A spending change stops this category forward.  Historical
                # amounts stay in the ledger; only the projection disappears.
                continue
        interval = _median_int(
            [
                (sorted(dates)[index + 1] - sorted(dates)[index]).days
                for index in range(len(dates) - 1)
            ]
        )
        if interval <= 0:
            interval = DEFAULT_INTERVAL_DAYS
        amount = -abs(_median_decimal(bucket["amounts"]))
        next_date = max(dates) + timedelta(days=interval)
        while next_date < req_date:
            next_date += timedelta(days=interval)
        while next_date <= window_end:
            if (category, next_date) not in existing:
                entries.append(
                    ForecastEntry(
                        date=next_date,
                        amount=amount,
                        category=category,
                        kind="expense",
                        source="history",
                        reason="recurring_median",
                    )
                )
            next_date += timedelta(days=interval)
    return entries


def _essential_variable_history(
    events,
    req_date: date,
    *,
    home_currency: str,
    fx_rows,
    warnings: list[dict],
) -> dict[str, list[Decimal]]:
    """Median monthly home-currency spend per essential-variable category.

    Only complete months before the request month count: a partial request month
    would understate typical spend and weaken the reserve.  At least two months
    are required, so a single observation never becomes a recurring assumption.
    """
    monthly: dict[str, dict[tuple[int, int], Decimal]] = {}
    for event in events:
        category = _norm(event.get("category"))
        if category not in ESSENTIAL_VARIABLE_CATEGORIES:
            continue
        if _norm(event.get("event_type")) not in EXPENSE_EVENT_TYPES:
            continue
        if not _counts_as_history(event, income=False):
            continue
        event_day = _event_day(event)
        if event_day is None or event_day >= req_date:
            continue
        if (event_day.year, event_day.month) >= (req_date.year, req_date.month):
            continue
        amount = _decimal(event.get("amount"))
        if amount is None or amount <= 0:
            continue
        rate = resolve_rate(
            fx_rows, _text(event.get("currency")), home_currency, event_day.isoformat()
        )["rate"]
        if rate is None:
            warnings.append(
                _warning(
                    "FORECAST_HISTORY_FX_UNKNOWN",
                    "",
                    f"skipped {category} reserve observation on {event_day.isoformat()}",
                    category,
                )
            )
            continue
        month_key = (event_day.year, event_day.month)
        bucket = monthly.setdefault(category, {})
        bucket[month_key] = bucket.get(month_key, Decimal(0)) + abs(amount * rate)

    history: dict[str, list[Decimal]] = {}
    for category, months in monthly.items():
        if len(months) < MIN_VARIABLE_RESERVE_MONTHS:
            continue
        history[category] = sorted(months.values())
    return history


def _essential_variable_reserve(
    events,
    req_date: date,
    window_end: date,
    *,
    home_currency: str,
    fx_rows,
    covered_entries: list[ForecastEntry],
    expense_facts,
    warnings: list[dict],
    skipped_categories=None,
) -> list[ForecastEntry]:
    """Reserve essential variable spend not already covered by committed rows.

    Each 30-day bucket reserves ``max(0, median_monthly_spend - covered)`` where
    ``covered`` is the recurring forecast plus real countable in-window rows for
    the same category.  Reserve rows post at the bucket start, so the ledger sees
    the outflow conservatively early.  An ended-expense fact suppresses reserve
    buckets on/after its effective date.  ``skipped_categories`` mirrors the
    unprotected suppressions applied by the direct forecast.
    """
    skipped = {_norm(category) for category in (skipped_categories or ()) if _norm(category)}
    history = _essential_variable_history(
        events,
        req_date,
        home_currency=home_currency,
        fx_rows=fx_rows,
        warnings=warnings,
    )
    if not history:
        return []

    real_rows: dict[str, list[tuple[date, Decimal]]] = {}
    for event in events:
        category = _norm(event.get("category"))
        if category not in ESSENTIAL_VARIABLE_CATEGORIES:
            continue
        if _norm(event.get("event_type")) not in EXPENSE_EVENT_TYPES:
            continue
        if not _counts_as_history(event, income=False):
            continue
        event_day = _event_day(event)
        if event_day is None or not (req_date <= event_day <= window_end):
            continue
        amount = _decimal(event.get("amount"))
        if amount is None or amount <= 0:
            continue
        rate = resolve_rate(
            fx_rows, _text(event.get("currency")), home_currency, event_day.isoformat()
        )["rate"]
        if rate is None:
            continue
        real_rows.setdefault(category, []).append((event_day, abs(amount * rate)))

    ended_effective: dict[str, date] = {}
    for fact in expense_facts or ():
        category = _norm(_fact_value(fact, "category"))
        if not category or _norm(_fact_value(fact, "kind")) not in _EXPENSE_END_FACT_KINDS:
            continue
        effective = _as_date(_fact_value(fact, "effective_date"))
        if effective is None:
            continue
        current = ended_effective.get(category)
        ended_effective[category] = effective if current is None else min(current, effective)

    entries: list[ForecastEntry] = []
    for category in sorted(history):
        if category in skipped:
            continue
        monthly_median = _median_decimal(history[category])
        for offset in range(0, HORIZON_DAYS, VARIABLE_RESERVE_BUCKET_DAYS):
            bucket_start = req_date + timedelta(days=offset)
            if bucket_start > window_end:
                break
            ended_on = ended_effective.get(category)
            if ended_on is not None and bucket_start >= ended_on:
                continue
            bucket_end = bucket_start + timedelta(days=VARIABLE_RESERVE_BUCKET_DAYS)
            covered = Decimal(0)
            for entry in covered_entries:
                if (
                    entry.kind == "expense"
                    and entry.category == category
                    and bucket_start <= entry.date < bucket_end
                ):
                    covered += abs(entry.amount)
            for event_day, amount in real_rows.get(category, ()):
                if bucket_start <= event_day < bucket_end:
                    covered += amount
            uncovered = monthly_median - covered
            if uncovered <= 0:
                continue
            entries.append(
                ForecastEntry(
                    date=bucket_start,
                    amount=-uncovered,
                    category=category,
                    kind="expense",
                    source="reserve",
                    reason="uncovered_essential_variable_spend",
                )
            )
    return entries


# ---------------------------------------------------------------------------
# Income projection
# ---------------------------------------------------------------------------


def _income_stream_is_eligible(description: str) -> bool:
    return not any(token in description for token in EXCLUDED_INCOME_DESCRIPTION_TOKENS)


def _income_forecast(
    events,
    req_date: date,
    window_end: date,
    *,
    user_id: str,
    home_currency: str,
    fx_rows,
    evidence_flags,
    warnings: list[dict],
) -> tuple[list[ForecastEntry], str | None]:
    if set(evidence_flags or ()) & INCOME_END_FLAGS:
        warnings.append(
            _warning(
                "INCOME_STREAM_ENDED_BY_EVIDENCE",
                "",
                "user evidence marks an income source as ended; no salary is projected",
            )
        )
        return [], None

    history: list[dict] = []
    for event in events:
        if _norm(event.get("event_type")) != "income":
            continue
        if _norm(event.get("category")) not in PRIMARY_INCOME_CATEGORIES:
            continue
        if not _counts_as_history(event, income=True):
            continue
        event_day = _event_day(event)
        if event_day is None or event_day >= req_date:
            continue
        description = _norm(event.get("description"))
        amount = _decimal(event.get("amount"))
        rate = None
        if amount is not None and amount > 0:
            rate = resolve_rate(
                fx_rows, _text(event.get("currency")), home_currency, event_day.isoformat()
            )["rate"]
        if rate is None:
            # A cadence date is still evidence that the stream was paid, even
            # when the amount cannot be converted; keep the date, drop the amount.
            warnings.append(
                _warning(
                    "FORECAST_HISTORY_FX_UNKNOWN",
                    "",
                    f"skipped income observation on {event_day.isoformat()}",
                    "salary",
                )
            )
        history.append(
            {
                "date": event_day,
                "description": description,
                "event_id": _text(event.get("event_id")),
                "amount": abs(amount * rate) if rate is not None else None,
            }
        )
    if not history:
        return [], None

    latest_date = max(row["date"] for row in history)
    if any(
        row["date"] == latest_date
        and any(token in row["description"] for token in TERMINAL_INCOME_DESCRIPTION_TOKENS)
        for row in history
    ):
        warnings.append(
            _warning(
                "INCOME_STREAM_ENDED",
                "",
                f"the latest salary row ({latest_date.isoformat()}) is a terminal payment",
            )
        )
        return [], None

    streams: dict[str, list[dict]] = {}
    for row in history:
        streams.setdefault(row["description"], []).append(row)

    candidates = []
    for description, rows in streams.items():
        if not _income_stream_is_eligible(description):
            continue
        dates = sorted(row["date"] for row in rows)
        if len(dates) < 2:
            continue
        amounts = [row["amount"] for row in rows if row["amount"] is not None]
        if not amounts:
            continue
        intervals = [
            (dates[index + 1] - dates[index]).days for index in range(len(dates) - 1)
        ]
        cadence = _cv([Decimal(interval) for interval in intervals])
        stability = _cv(amounts)
        median_amount = _median_decimal(amounts)
        candidates.append(
            (
                round(cadence, 6),
                round(stability, 6),
                -median_amount,
                description,
                rows,
                dates,
                intervals,
            )
        )
    if not candidates:
        return [], None

    _, _, _, description, rows, dates, intervals = sorted(candidates)[0]
    interval = _median_int(intervals)
    if interval <= 0:
        interval = DEFAULT_INTERVAL_DAYS

    latest = sorted(
        (row for row in rows if row["amount"] is not None),
        key=lambda row: (row["date"], row["event_id"]),
    )[-1]
    amount = latest["amount"]

    # Real countable salary rows already committed inside the window: they fill
    # part of the occurrence cap and anchor the projection after their latest date.
    real_dates: list[date] = []
    for event in events:
        if _norm(event.get("category")) not in PRIMARY_INCOME_CATEGORIES:
            continue
        if _norm(event.get("event_type")) != "income":
            continue
        real_description = _norm(event.get("description"))
        if not _income_stream_is_eligible(real_description):
            continue
        event_day = _event_day(event)
        if event_day is None or not (req_date <= event_day <= window_end):
            continue
        if not _counts_as_history(event, income=True):
            continue
        real_dates.append(event_day)
    existing = set(real_dates)

    remaining = MAX_INCOME_OCCURRENCES - len(real_dates)
    if remaining <= 0:
        return [], (description or None)

    entries: list[ForecastEntry] = []
    anchor = max(max(dates), max(real_dates)) if real_dates else max(dates)
    next_date = anchor + timedelta(days=interval)
    while next_date < req_date:
        next_date += timedelta(days=interval)
    emitted = 0
    while next_date <= window_end and emitted < remaining:
        if next_date not in existing:
            entries.append(
                ForecastEntry(
                    date=next_date,
                    amount=amount,
                    category="salary",
                    kind="income",
                    source="history",
                    reason="primary_income_stream",
                )
            )
            emitted += 1
        next_date += timedelta(days=interval)
    return entries, (description or None)


# ---------------------------------------------------------------------------
# Evidence expense facts
# ---------------------------------------------------------------------------


def _apply_expense_facts(
    entries: list[ForecastEntry],
    facts,
    *,
    window_end: date,
    home_currency: str,
    fx_rows,
    warnings: list[dict],
) -> list[ForecastEntry]:
    """Move forecast rows on/after a fact's effective date; never guess."""
    if not facts:
        return entries
    updated = list(entries)
    for fact in facts:
        category = _norm(_fact_value(fact, "category"))
        kind = _norm(_fact_value(fact, "kind"))
        source_id = _text(_fact_value(fact, "source_message_id"))
        effective = _as_date(_fact_value(fact, "effective_date"))
        if not category:
            warnings.append(
                _warning("EXPENSE_FACT_AMBIGUOUS", source_id, "fact has no category")
            )
            continue
        targeted = [
            entry
            for entry in updated
            if entry.kind == "expense" and entry.category == category
        ]
        if not targeted:
            warnings.append(
                _warning(
                    "EXPENSE_FACT_NOT_APPLIED",
                    source_id,
                    f"no forecast row for {category}",
                    category,
                )
            )
            continue
        if effective is None:
            warnings.append(
                _warning(
                    "EXPENSE_FACT_AMBIGUOUS",
                    source_id,
                    "fact has no effective date; forecast left unchanged",
                    category,
                )
            )
            continue
        if effective > window_end:
            # Outside the forecast horizon: cannot affect this request.
            continue
        if kind in _EXPENSE_END_FACT_KINDS:
            updated = [
                entry
                for entry in updated
                if not (
                    entry.kind == "expense"
                    and entry.category == category
                    and entry.date >= effective
                )
            ]
            continue
        if kind in _EXPENSE_INCREASE_FACT_KINDS:
            percent = _decimal(_fact_value(fact, "percent_change"))
            if percent is None:
                warnings.append(
                    _warning(
                        "EXPENSE_FACT_AMBIGUOUS",
                        source_id,
                        "increase fact has no percent change; forecast left unchanged",
                        category,
                    )
                )
                continue
            factor = Decimal(1) + percent / Decimal(100)
            if factor <= 0:
                warnings.append(
                    _warning(
                        "EXPENSE_FACT_AMBIGUOUS",
                        source_id,
                        "percent change would remove the expense; ignored",
                        category,
                    )
                )
                continue
            updated = [
                replace(
                    entry,
                    amount=entry.amount * factor,
                    source="evidence",
                    reason=f"expense_fact:increase:{format(percent, 'f')}%",
                )
                if entry.kind == "expense"
                and entry.category == category
                and entry.date >= effective
                else entry
                for entry in updated
            ]
            continue
        if kind in _EXPENSE_AMOUNT_FACT_KINDS:
            amount = _decimal(_fact_value(fact, "amount"))
            if amount is None or amount <= 0:
                warnings.append(
                    _warning(
                        "EXPENSE_FACT_AMBIGUOUS",
                        source_id,
                        "new recurring amount is missing; forecast left unchanged",
                        category,
                    )
                )
                continue
            currency = _text(_fact_value(fact, "currency")).upper()
            if not currency:
                warnings.append(
                    _warning(
                        "EXPENSE_FACT_AMBIGUOUS",
                        source_id,
                        "new recurring amount has no currency; forecast left unchanged",
                        category,
                    )
                )
                continue
            rate = resolve_rate(
                fx_rows, currency, home_currency, effective.isoformat()
            )["rate"]
            if rate is None:
                warnings.append(
                    _warning(
                        "EXPENSE_FACT_AMBIGUOUS",
                        source_id,
                        f"no {currency}->{home_currency} rate on {effective.isoformat()}",
                        category,
                    )
                )
                continue
            new_amount = -abs(amount * rate)
            updated = [
                replace(
                    entry,
                    amount=new_amount,
                    source="evidence",
                    reason=f"expense_fact:{kind}",
                )
                if entry.kind == "expense"
                and entry.category == category
                and entry.date >= effective
                else entry
                for entry in updated
            ]
            continue
        warnings.append(
            _warning(
                "EXPENSE_FACT_AMBIGUOUS",
                source_id,
                f"unknown expense fact kind {kind!r}; forecast left unchanged",
                category,
            )
        )
    return updated


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_forecast(
    events,
    req_date,
    *,
    user_id: str | None = None,
    home_currency: str = "USD",
    fx_rows=None,
    recurrence_flags=None,
    suppressed_categories=None,
    protected_categories=None,
    evidence_flags=None,
    expense_facts=None,
) -> ForecastResult:
    """Project committed cash rows for ``[req_date, req_date + 90]``.

    ``events`` is expected to be the user's normalized event rows (the oracle
    filters and dedups before calling).  Every policy input is explicit so the
    result is reproducible and unit-testable without pandas.
    """
    request_day = _as_date(req_date)
    if request_day is None:
        raise ValueError(f"req_date must be an ISO date, got {req_date!r}")
    events = list(events or ())
    fx_rows = list(fx_rows or ())
    if user_id is None:
        user_id = next(
            (_text(event.get("user_id")) for event in events if _text(event.get("user_id"))),
            "",
        )
    else:
        user_id = _text(user_id)
    home_currency = _text(home_currency) or "USD"
    window_end = request_day + timedelta(days=HORIZON_DAYS)
    suppressed = {_norm(category) for category in (suppressed_categories or ()) if _norm(category)}
    protected = {_norm(category) for category in (protected_categories or ()) if _norm(category)}

    warnings: list[dict] = []
    expense_entries = _expense_forecast(
        events,
        request_day,
        window_end,
        user_id=user_id,
        home_currency=home_currency,
        fx_rows=fx_rows,
        recurrence_flags=recurrence_flags,
        suppressed=suppressed,
        protected=protected,
        warnings=warnings,
    )
    expense_entries = _apply_expense_facts(
        expense_entries,
        expense_facts,
        window_end=window_end,
        home_currency=home_currency,
        fx_rows=fx_rows,
        warnings=warnings,
    )
    expense_entries = expense_entries + _essential_variable_reserve(
        events,
        request_day,
        window_end,
        home_currency=home_currency,
        fx_rows=fx_rows,
        covered_entries=expense_entries,
        expense_facts=expense_facts,
        warnings=warnings,
        skipped_categories=suppressed - protected,
    )
    income_entries, income_stream = _income_forecast(
        events,
        request_day,
        window_end,
        user_id=user_id,
        home_currency=home_currency,
        fx_rows=fx_rows,
        evidence_flags=evidence_flags,
        warnings=warnings,
    )

    entries = expense_entries + income_entries
    entries.sort(key=lambda entry: (entry.date, entry.category, entry.kind, entry.source, entry.reason))
    return ForecastResult(entries=entries, warnings=warnings, income_stream=income_stream)


__all__ = [
    "DISCRETIONARY_CATEGORIES",
    "ESSENTIAL_VARIABLE_CATEGORIES",
    "EXPENSE_EVENT_TYPES",
    "EXCLUDED_INCOME_DESCRIPTION_TOKENS",
    "FORECASTABLE_EXPENSE_CATEGORIES",
    "ForecastEntry",
    "ForecastResult",
    "HORIZON_DAYS",
    "INCOME_END_FLAGS",
    "MAX_INCOME_OCCURRENCES",
    "PRIMARY_INCOME_CATEGORIES",
    "build_forecast",
]
