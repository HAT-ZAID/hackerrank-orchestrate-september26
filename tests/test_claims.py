"""Claim checking: a model may phrase things, never add facts."""
from __future__ import annotations

from code.claims import describe_rejection, verify_claims
from code.verifier import enforce_decision

ROW = {
    "request_id": "request_demo",
    "amount_safe_to_pay": "15656000",
    "affordability_status": "affordable_now",
    "recommended_payment_method": "full_payment",
    "payment_plan": "2025-08-03:15656000",
    "earliest_date_for_full_payment": "2025-08-03",
    "spending_changes_needed": "none",
    "decision_explanation": "Recommended full_payment (affordable_now).",
}


def test_grounded_prose_passes_with_every_claim_counted():
    report = verify_claims(
        "Paying 15,656,000.00 on 2025-08-03 keeps the balance above the floor; "
        "full payment is the recommended method.",
        ROW,
    )
    assert report["ok"] is True
    assert report["checked"] >= 3
    assert report["unsupported"] == []


def test_invented_amount_is_rejected_with_the_reason():
    report = verify_claims("You can safely pay 20,000,000 today.", ROW)
    assert report["ok"] is False
    (claim,) = report["unsupported"]
    assert claim["kind"] == "amount"
    assert claim["text"] == "20,000,000"
    assert "15656000" in claim["expected"][0]
    assert "20,000,000" in describe_rejection(report)


def test_invented_date_is_rejected():
    report = verify_claims("Wait until 2025-09-30 and then pay.", ROW)
    assert report["ok"] is False
    assert {item["kind"] for item in report["unsupported"]} == {"date", "method"}


def test_wrong_method_and_status_words_are_rejected():
    report = verify_claims("You should wait and pay in installments.", ROW)
    kinds = {item["kind"] for item in report["unsupported"]}
    assert kinds == {"method"}


def test_literal_contract_tokens_are_checked():
    report = verify_claims("The kernel says not_affordable here.", ROW)
    assert report["ok"] is False
    (claim,) = report["unsupported"]
    assert claim["kind"] == "token"


def test_actions_must_match_the_decision():
    report = verify_claims("Stop stop:event_2322 to make room.", ROW)
    assert report["ok"] is False
    assert report["unsupported"][0]["kind"] == "action"


def test_small_counts_are_not_treated_as_amounts():
    report = verify_claims("It takes 2 payments and 3 days.", ROW)
    assert report["ok"] is True
    assert report["checked"] == 0


def test_extra_facts_from_the_request_may_be_cited():
    report = verify_claims(
        "The 1,000.00 request dated 2026-01-05 fits.",
        {**ROW, "amount_safe_to_pay": "1000"},
        extra_numbers=("1000",),
        extra_dates=("2026-01-05",),
    )
    assert report["ok"] is True


def test_empty_prose_is_not_a_claim_and_does_not_ship():
    report = verify_claims("   ", ROW)
    assert report["ok"] is False
    assert report["rejected"] is False  # nothing was asserted, so nothing was refused


def test_verifier_keeps_kernel_text_when_the_rationale_fabricates():
    candidate = {
        "request_id": "request_demo",
        "amount_safe_to_pay": "15656000",
        "affordability_status": "affordable_now",
        "recommended_payment_method": "full_payment",
        "payment_plan": "2025-08-03:15656000",
        "earliest_date_for_full_payment": "2025-08-03",
        "spending_changes_needed": "none",
        "evidence_sufficient": True,
        "confidence": 0.9,
        "rationale": "You can pay 99,000,000 and still be fine.",
    }
    result = enforce_decision(candidate, dict(ROW))
    assert result["final_row"]["decision_explanation"] == ROW["decision_explanation"]
    assert result["verification"]["rationale_source"] == "kernel"
    assert result["verification"]["claim_check"]["ok"] is False


def test_verifier_ships_a_grounded_rationale_even_when_fields_disagree():
    candidate = {
        "request_id": "request_demo",
        "amount_safe_to_pay": "1",  # wrong
        "affordability_status": "affordable_later",  # wrong
        "recommended_payment_method": "wait",  # wrong
        "payment_plan": "none",  # wrong
        "earliest_date_for_full_payment": "",  # wrong
        "spending_changes_needed": "none",
        "evidence_sufficient": True,
        "confidence": 0.4,
        "rationale": "The transfer of 15,656,000 clears on 2025-08-03 in one payment.",
    }
    result = enforce_decision(candidate, dict(ROW))
    assert result["verification"]["rationale_source"] == "candidate"
    assert result["verification"]["agent_agreement"] is False
    assert result["final_row"]["decision_explanation"] == candidate["rationale"]
    # machine fields still come from the kernel
    assert result["final_row"]["affordability_status"] == "affordable_now"
    assert result["final_row"]["amount_safe_to_pay"] == "15656000"
