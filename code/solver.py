"""P3 Full Solver: Integrates Oracle, Ranking, and Payment Options to make the final decision.

Launder Logic:
1. Process Messages/Images to update profiles and events.
2. Identify all viable payment options (Full, Partial, Installments, Wait).
3. For each option, use the Oracle to check safety.
4. Rank the safe options using Ranking logic (code/ranking.py).
5. Pick the best.
"""
from __future__ import annotations

import pandas as pd
from datetime import date, timedelta
from decimal import Decimal
from code.oracle import solve_safe_amount
from code.options import get_valid_options
from code.changes import search_changes
from code.recurrence import detect_recurrence
from code.parser import parse_messages
from code.reconcile import reconcile_events

# Salary kinds that may create a confirmed future income row when no matching
# row exists (the historical raise behavior). A resumed/regular salary and any
# reduction or temporary adjustment only amend an existing row: injecting them
# would duplicate income that the payroll cadence already provides.
_SALARY_INJECTABLE_KINDS = frozenset({"raise", "first", "final"})


def _empty_evidence() -> dict:
    return {
        "flags": [],
        "salary_changes": [],
        "income_facts": [],
        "expense_facts": [],
        "event_updates": [],
        "applied_salary_changes": [],
        "event_amendment_notes": [],
        "warnings": [],
        "notes": [],
    }


def _salary_row_date(event: dict) -> str:
    return str(event.get("settlement_date") or event.get("event_date") or "").strip()


def _is_salary_row(event: dict, user_id) -> bool:
    if str(event.get("user_id") or "") not in ("", str(user_id)):
        return False
    if str(event.get("category") or "").strip().lower() != "salary":
        return False
    return str(event.get("event_type") or "").strip().lower() == "income"


def _amend_salary_row(events, user_id, target_date, amount, currency) -> bool:
    matches = [
        event
        for event in events
        if _is_salary_row(event, user_id) and _salary_row_date(event) == target_date
    ]
    if not matches:
        return False
    # Prefer a countable future row over an already-settled historical one,
    # then order deterministically so a repeated run selects the same row.
    matches.sort(
        key=lambda event: (
            0 if str(event.get("status") or "").strip().lower() in {"scheduled", "pending"} else 1,
            str(event.get("event_id") or ""),
        )
    )
    event = matches[0]
    event["amount"] = str(amount)
    if str(currency or "").strip():
        event["currency"] = str(currency)
    # Countable scheduled income; never resurrect a failed/cancelled row.
    if str(event.get("status") or "").strip().lower() in ("", "nan", "none"):
        event["status"] = "scheduled"
    return True


def _inject_salary_row(events, user_id, kind, target_date, amount, currency) -> None:
    events.append({
        "event_id": f"salary_{kind}_{target_date}",
        "user_id": user_id,
        "event_type": "income",
        "description": "Salary update via payroll message (injected)",
        "category": "salary",
        "direction": "credit",
        "amount": str(amount),
        "currency": str(currency or ""),
        "event_date": target_date,
        "settlement_date": target_date,
        "status": "scheduled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    })


def _next_scheduled_salary_row(events, user_id, not_before: str):
    candidates = [
        event
        for event in events
        if _is_salary_row(event, user_id)
        and str(event.get("status") or "").strip().lower() in {"scheduled", "pending"}
        and _salary_row_date(event) >= not_before
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_salary_row_date)[0]


def _salary_change_warning(change: dict, detail: str) -> dict:
    return {
        "message_id": str(change.get("source_message_id") or ""),
        "flag": "SALARY_CHANGE_NOT_APPLIED",
        "kind": str(change.get("kind") or ""),
        "date": str(change.get("effective_date") or change.get("settlement_date") or ""),
        "detail": detail,
    }


def _apply_salary_changes(events, salary_changes, user_id, profile, req_date) -> tuple[list[dict], list[dict]]:
    """Apply typed salary facts to the oracle copy; return (applied, warnings).

    A dated change amends the salary row on that date, or injects a confirmed
    income row for kinds that legitimately create income. An undated "next
    payroll" change only amends the next scheduled salary row: it never invents
    a payment date. A change that matches no row and cannot be injected is
    reported in ``warnings`` so it is never silently dropped. Changes dated
    beyond the 90-day horizon are skipped by design (they cannot affect this
    request) and are not reported as data loss.
    """
    applied: list[dict] = []
    warnings: list[dict] = []
    request_day = str(req_date or "").strip()
    try:
        window_end = date.fromisoformat(request_day) + timedelta(days=90)
    except Exception:
        window_end = None

    for change in salary_changes or []:
        kind = str(change.get("kind") or "")
        amount = change.get("amount")
        if amount is None or kind == "ended":
            continue
        currency = change.get("currency") or profile.get("home_currency", "USD")
        target = change.get("effective_date") or change.get("settlement_date")

        if target:
            try:
                target_date = date.fromisoformat(str(target).strip())
            except Exception:
                warnings.append(
                    _salary_change_warning(change, "unparsable salary change date")
                )
                continue
            if window_end is not None and target_date > window_end:
                continue
            target_s = target_date.isoformat()
            if _amend_salary_row(events, user_id, target_s, amount, currency):
                applied.append({"kind": kind, "date": target_s, "amount": str(amount),
                                "source_message_id": change.get("source_message_id", ""),
                                "mode": "amended"})
            elif kind in _SALARY_INJECTABLE_KINDS:
                _inject_salary_row(events, user_id, kind, target_s, amount, currency)
                applied.append({"kind": kind, "date": target_s, "amount": str(amount),
                                "source_message_id": change.get("source_message_id", ""),
                                "mode": "injected"})
            else:
                warnings.append(
                    _salary_change_warning(
                        change, f"no salary row on {target_s} to amend; not injected by kind"
                    )
                )
            continue

        # Undated reduction/temporary/confirmation: amend the next payroll row
        # only. No row means no application (never invent a salary payment).
        row = _next_scheduled_salary_row(events, user_id, request_day) if request_day else None
        if row is None:
            warnings.append(
                _salary_change_warning(change, "no upcoming salary row to amend; no date stated")
            )
            continue
        row_date = _salary_row_date(row)
        row["amount"] = str(amount)
        if str(currency or "").strip():
            row["currency"] = str(currency)
        applied.append({"kind": kind, "date": row_date, "amount": str(amount),
                        "source_message_id": change.get("source_message_id", ""),
                        "mode": "amended"})
    return applied, warnings


def _evidence_notes(facts: dict) -> list[str]:
    """Limitations recorded for the forecast layer (Task 3), never silent."""
    notes: list[str] = []
    if "employment_ended" in facts.get("flags", set()) or "contract_ended" in facts.get("flags", set()):
        notes.append("income_end_not_applied_to_forecast")
    if "partial_income_ended" in facts.get("flags", set()):
        notes.append("partial_income_end_not_applied_to_forecast")
    if any(fact.get("status") == "confirmed" for fact in facts.get("income_facts", [])):
        notes.append("confirmed_income_not_injected")
    if any(change.get("amount") is None for change in facts.get("salary_changes", [])):
        notes.append("unquantified_salary_change_not_applied")
    if facts.get("expense_facts"):
        notes.append("expense_changes_not_applied_to_forecast")
    return notes


def _parse_plan_legs(payment_plan: str) -> list[tuple[date, Decimal]] | None:
    legs: list[tuple[date, Decimal]] = []
    try:
        parts = str(payment_plan or "").split("|")
    except Exception:
        return None
    for part in parts:
        part = part.strip()
        if not part or ":" not in part:
            return None
        d_s, a_s = part.split(":", 1)
        try:
            d = date.fromisoformat(d_s.strip())
            a = Decimal(str(a_s).strip())
            if not a.is_finite() or a < 0:
                return None
            legs.append((d, a))
        except Exception:
            return None
    if not legs:
        return None
    return sorted(legs)


def _installment_safe(ledger, req_date_str: str, payment_plan: str, profile: dict) -> bool:
    """Strict leg simulation: rebuild post-leg balances (ledger - cumulative legs).

    Returns True when every daily post-leg balance stays >= minimum_balance_to_keep.
    Fail-closed: empty/missing/short/unparsable ledger, unparsable req_date/floor,
    NaN balances, or legs beyond the ledger horizon all return False.
    """
    try:
        req_d = date.fromisoformat(str(req_date_str).strip())
    except Exception:
        return False
    try:
        min_bal = Decimal(str(profile.get("minimum_balance_to_keep", "0")))
        if min_bal.is_nan():
            return False
    except Exception:
        return False
    if ledger is None:
        return False
    try:
        bal_list = [Decimal(str(b)) for b in ledger]
    except Exception:
        return False
    # Oracle ledger contract is 91 daily balances; anything else is fail-closed.
    if len(bal_list) != 91:
        return False
    for b in bal_list:
        try:
            if not b.is_finite():
                return False
        except Exception:
            return False
    legs = _parse_plan_legs(payment_plan)
    if not legs:
        return False
    # Conservative: legs beyond the ledger horizon cannot be verified -> unsafe.
    try:
        horizon = req_d + timedelta(days=len(bal_list) - 1)
        for d, _ in legs:
            if d < req_d:
                return False
            if d > horizon:
                return False
    except Exception:
        return False
    n = len(bal_list)
    for i in range(n):
        day = req_d + timedelta(days=i)
        cum = sum((a for d, a in legs if d <= day), Decimal("0"))
        if not cum:
            continue
        try:
            if bal_list[i] - cum < min_bal:
                return False
        except Exception:
            return False
    return True


def _parse_deadline_date(val) -> date | None:
    try:
        if val is None:
            return None
        if isinstance(val, float) and val != val:  # NaN
            return None
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "null", "nat", "n/a", "na"):
            return None
        return date.fromisoformat(s)
    except Exception:
        return None


def meets_deadline(comp_date, desired_completion_date) -> bool:
    """Deadline check with NaN/'' normalization (missing/NaN = no constraint)."""
    desired = _parse_deadline_date(desired_completion_date)
    if desired is None:
        return True
    comp = _parse_deadline_date(comp_date)
    if comp is None:
        return True
    return comp <= desired


def solve_request(rid, user_id, req_date, req_amt, profile, events, fx_rows, options_df, messages_df=None, desired_completion_date=None, allows_partial_payment=None, installment_safety="strict"):
    import copy as _copy
    req_amt_dec = Decimal(str(req_amt))
    # Never mutate caller data: work on a modified copy for all oracle calls.
    try:
        events_for_oracle = _copy.deepcopy(events) if events is not None else []
    except Exception:
        events_for_oracle = [dict(e) for e in (events or [])]

    # 1. Process Evidence (Messages) into typed facts, then apply them to the
    #    oracle copy (oracle signature frozen). Untrusted content never decides
    #    a field: only typed facts reach the ledger.
    evidence = _empty_evidence()
    if messages_df is not None:
        try:
            facts = parse_messages(user_id, messages_df)
        except Exception:
            facts = None
        if facts is not None:
            # Event amendments: exact event_id match only, AGENTS 6.3 conflict order.
            events_for_oracle, amendment_warnings = reconcile_events(
                events_for_oracle, facts.get("event_updates", [])
            )
            # Flag stability: snapshot BEFORE salary injection so a huge raise
            # cannot drop the salary flag via CV (injection kept for amounts).
            try:
                _pre_injection = _copy.deepcopy(events_for_oracle)
            except Exception:
                _pre_injection = [dict(e) for e in events_for_oracle]
            applied_changes, salary_warnings = _apply_salary_changes(
                events_for_oracle, facts.get("salary_changes", []), user_id, profile, req_date
            )
            evidence = {
                "flags": sorted(facts.get("flags", set())),
                "salary_changes": facts.get("salary_changes", []),
                "income_facts": facts.get("income_facts", []),
                "expense_facts": facts.get("expense_facts", []),
                "event_updates": facts.get("event_updates", []),
                "applied_salary_changes": applied_changes,
                "event_amendment_notes": amendment_warnings,
                "warnings": list(facts.get("warnings", []))
                + list(amendment_warnings)
                + list(salary_warnings),
                "notes": _evidence_notes(facts),
            }
        else:
            try:
                _pre_injection = _copy.deepcopy(events_for_oracle)
            except Exception:
                _pre_injection = [dict(e) for e in events_for_oracle]

        recurrence_flags = detect_recurrence(_pre_injection)
    else:
        # No messages: flags from the (uninjected) oracle copy.
        recurrence_flags = detect_recurrence(events_for_oracle)

    # Parse eligibility per spec Â§Choosing Between Safe Plans
    will_raw = profile.get("payment_methods_user_will_consider", "") or ""
    will = set(s.strip() for s in str(will_raw).split("|") if s.strip())
    if desired_completion_date is None:
        # caller may embed in options_df path; default to None (no deadline filter)
        pass
    allows_partial = str(allows_partial_payment).strip().lower() == "true" if allows_partial_payment is not None else False

    # 2. Get Oracle results for the base state (modified copy carries salary injection).
    res_now = solve_safe_amount(user_id, req_date, str(req_amt_dec), profile, events_for_oracle, fx_rows, recurrence_flags)
    safe_amt = res_now["safe_amount"]
    earliest = res_now["earliest_date"]

    def _meets(comp_date) -> bool:
        return meets_deadline(comp_date, desired_completion_date)

    def _valid_earliest(val) -> bool:
        try:
            date.fromisoformat(str(val))
            return True
        except Exception:
            return False

    # Build viable options
    viable_options = []

    # Option A: Full Payment (Now) â€” eligible only if user considers full_payment
    if safe_amt >= req_amt_dec and "full_payment" in will:
        viable_options.append({
            "option_id": "opt_full_now",
            "id": "opt_full_now",
            "payment_method": "full_payment",
            "amount_safe_to_pay": req_amt_dec,
            "affordability_status": "affordable_now",
            "payment_plan": f"{req_date}:{req_amt_dec}",
            "earliest_date": req_date,
            "spending_changes": "none",
            "cost": req_amt_dec,
            "total_cost": req_amt_dec,
            "completion_date": req_date,
            "changes": 0,
            "legs": 1,
            "start_date": req_date,
            "num_payments": 1
        })

    # Option B: Partial Payment â€” only when request allows, user accepts, 0<safe<req, second leg by deadline
    if Decimal("0") < safe_amt < req_amt_dec and earliest and "partial_payment" in will:
        partial_ok = True
        if allows_partial is False:
            partial_ok = False
        if not _meets(earliest):
            partial_ok = False
        if partial_ok:
            viable_options.append({
            "option_id": "opt_partial",
            "id": "opt_partial",
            "payment_method": "partial_payment",
            "amount_safe_to_pay": safe_amt,
            "affordability_status": "affordable_with_plan",
            "payment_plan": f"{req_date}:{safe_amt}|{earliest}:{req_amt_dec - safe_amt}",
            "earliest_date": earliest,
            "spending_changes": "none",
            "cost": req_amt_dec,
            "total_cost": req_amt_dec,
            "completion_date": earliest,
            "changes": 0,
            "legs": 2,
            "start_date": req_date,
            "num_payments": 2
        })

    # Option C: Installments â€” eligible only if user considers installments.
    # Strict (default): deadline + per-leg ledger safety (post-leg balances >= floor).
    # Lenient: deadline only (legacy behavior).
    strict_inst = (installment_safety != "lenient")
    ledger_now = res_now.get("ledger", [])
    valid_inst = get_valid_options(rid, profile, options_df)
    for opt in valid_inst:
        if opt["payment_method"] == "installments" and "installments" in will:
            if not _meets(opt.get("completion_date", opt["first_payment_date"])):
                continue
            if strict_inst and not _installment_safe(ledger_now, req_date, opt.get("payment_plan", ""), profile):
                continue
            viable_options.append({
                "option_id": opt["option_id"],
                "id": opt["option_id"],
                "payment_method": "installments",
                "amount_safe_to_pay": safe_amt if safe_amt < req_amt_dec else req_amt_dec,
                "affordability_status": "affordable_with_plan",
                "payment_plan": opt["payment_plan"],
                "earliest_date": earliest,
                "spending_changes": "none",
                "cost": opt.get("total_payable", opt["amount"]),
                "total_cost": opt.get("total_payable", opt["amount"]),
                "completion_date": opt.get("completion_date", opt["first_payment_date"]),
                "changes": 0,
                "legs": opt["number_of_payments"],
                "start_date": opt["first_payment_date"],
                "num_payments": opt["number_of_payments"]
            })

    # Option D: Wait â€” eligible when full becomes safe later and user accepts full_payment
    if earliest and _valid_earliest(earliest) and str(earliest) > str(req_date) and "full_payment" in will and _meets(earliest):
        viable_options.append({
            "option_id": "opt_wait",
            "id": "opt_wait",
            "payment_method": "wait",
            "amount_safe_to_pay": safe_amt,
            "affordability_status": "affordable_later",
            "payment_plan": f"{earliest}:{req_amt_dec}",
            "earliest_date": earliest,
            "spending_changes": "none",
            "cost": req_amt_dec,
            "total_cost": req_amt_dec,
            "completion_date": earliest,
            "changes": 0,
            "legs": 1,
            "start_date": earliest,
            "num_payments": 1
        })

    # Spec Â§Choosing: try permitted spending changes when no viable plan,
    # or when viable exist but ALL miss desired_completion_date.
    def _append_changed_options(changed_safe, changed_earliest, changed_ledger, changes_text):
        """Add safe variants to the same pool as no-change options."""
        change_count = len(changes_text.split("|"))
        if changed_safe >= req_amt_dec and "full_payment" in will:
            viable_options.append({
                "option_id": "opt_full_changed", "id": "opt_full_changed",
                "payment_method": "full_payment", "amount_safe_to_pay": req_amt_dec,
                "affordability_status": "affordable_with_plan",
                "payment_plan": f"{req_date}:{req_amt_dec}", "earliest_date": changed_earliest,
                "spending_changes": changes_text, "cost": req_amt_dec, "total_cost": req_amt_dec,
                "completion_date": req_date, "changes": change_count, "legs": 1,
                "start_date": req_date, "num_payments": 1,
            })
        if (Decimal("0") < safe_amt < req_amt_dec and allows_partial and
                "partial_payment" in will and changed_earliest and _meets(changed_earliest)):
            viable_options.append({
                "option_id": "opt_partial_changed", "id": "opt_partial_changed",
                "payment_method": "partial_payment", "amount_safe_to_pay": safe_amt,
                "affordability_status": "affordable_with_plan",
                "payment_plan": f"{req_date}:{safe_amt}|{changed_earliest}:{req_amt_dec - safe_amt}",
                "earliest_date": changed_earliest, "spending_changes": changes_text,
                "cost": req_amt_dec, "total_cost": req_amt_dec, "completion_date": changed_earliest,
                "changes": change_count, "legs": 2, "start_date": req_date, "num_payments": 2,
            })
        for opt in valid_inst:
            if opt["payment_method"] != "installments" or "installments" not in will:
                continue
            completion = opt.get("completion_date", opt["first_payment_date"])
            if not _meets(completion):
                continue
            if strict_inst and not _installment_safe(changed_ledger, req_date, opt.get("payment_plan", ""), profile):
                continue
            changed_id = f"{opt['option_id']}_changed"
            viable_options.append({
                "option_id": changed_id, "id": changed_id, "payment_method": "installments",
                "amount_safe_to_pay": safe_amt if safe_amt < req_amt_dec else req_amt_dec,
                "affordability_status": "affordable_with_plan", "payment_plan": opt["payment_plan"],
                "earliest_date": changed_earliest, "spending_changes": changes_text,
                "cost": opt.get("total_payable", opt["amount"]),
                "total_cost": opt.get("total_payable", opt["amount"]), "completion_date": completion,
                "changes": change_count, "legs": opt["number_of_payments"],
                "start_date": opt["first_payment_date"], "num_payments": opt["number_of_payments"],
            })
        if (changed_earliest and _valid_earliest(changed_earliest) and
                date.fromisoformat(str(changed_earliest)) > date.fromisoformat(str(req_date)) and
                "full_payment" in will and _meets(changed_earliest)):
            viable_options.append({
                "option_id": "opt_wait_changed", "id": "opt_wait_changed", "payment_method": "wait",
                "amount_safe_to_pay": safe_amt, "affordability_status": "affordable_later",
                "payment_plan": f"{changed_earliest}:{req_amt_dec}", "earliest_date": changed_earliest,
                "spending_changes": changes_text, "cost": req_amt_dec, "total_cost": req_amt_dec,
                "completion_date": changed_earliest, "changes": change_count, "legs": 1,
                "start_date": changed_earliest, "num_payments": 1,
            })

    spending_needed = "none"
    _all_miss = False
    if viable_options:
        try:
            _desired_present = _parse_deadline_date(desired_completion_date) is not None
        except Exception:
            _desired_present = False
        if _desired_present and all(not _meets(o.get("completion_date")) for o in viable_options):
            _all_miss = True
    if (not viable_options) or _all_miss:
        try:
            changes = search_changes(user_id, req_date, req_amt_dec, profile, events_for_oracle, fx_rows, recurrence_flags)
            if changes:
                spending_needed = "|".join(changes)
                # Re-evaluate under the SAME semantics search_changes validated:
                # in-window rows are real upcoming cash (copy edit), while
                # pre-request history never hits the ledger, so only category
                # suppression removes the category's forward forecast.
                mod_events = [dict(e) for e in events_for_oracle]
                suppressed_cats: set[str] = set()
                for ch in changes:
                    parts = ch.split(":")
                    target = None
                    if parts[0] == "stop" and len(parts) == 2:
                        target = parts[1]
                    elif parts[0] == "reduce_to" and len(parts) == 3:
                        target = parts[1]
                    if target is None:
                        continue
                    for e in mod_events:
                        if e.get("event_id") == target:
                            e["amount"] = "0" if parts[0] == "stop" else parts[2]
                            cat = str(e.get("category", "") or "").strip().lower()
                            if cat:
                                suppressed_cats.add(cat)
                res2 = solve_safe_amount(user_id, req_date, str(req_amt_dec), profile, mod_events, fx_rows,
                                         recurrence_flags, suppressed_categories=suppressed_cats)
                _append_changed_options(res2["safe_amount"], res2["earliest_date"],
                                        res2.get("ledger", []), spending_needed)
        except Exception:
            pass
        if not viable_options:
            return {
                "request_id": rid,
                "amount_safe_to_pay": str(safe_amt),
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": None,
                "spending_changes_needed": "none",
                "decision_explanation": "No viable payment options found.",
                "evidence": evidence,
            }

    # Spec rank order: deadline âœ“, no changes, minimize total, start earlier, fewer legs, lowest option_id
    def spec_key(o):
        return (
            0 if _meets(o.get("completion_date")) else 1,
            o.get("changes", 0),
            Decimal(str(o.get("total_cost", o.get("cost", "Infinity")))),
            o.get("start_date", "9999-12-31"),
            o.get("legs", o.get("num_payments", 999)),
            o.get("id", o.get("option_id", "zzz")),
        )
    ranked = sorted(viable_options, key=spec_key)
    best = ranked[0]

    return {
        "request_id": rid,
        "amount_safe_to_pay": str(safe_amt),
        "affordability_status": best["affordability_status"],
        "recommended_payment_method": best["payment_method"],
        "payment_plan": best["payment_plan"],
        "earliest_date_for_full_payment": earliest,
        "spending_changes_needed": best["spending_changes"],
        "decision_explanation": f"Recommended {best['payment_method']} based on ranking.",
        "evidence": evidence,
    }
