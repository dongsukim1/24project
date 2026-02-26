"""Tests for shared Index 2 / Index 4 appeal precedent behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ems_pipeline.rag import retrieve_payer_requirements
from ems_pipeline.rag.appeal_precedents import AppealPrecedentQuery, AppealPrecedentsIndex
from ems_pipeline.rag.payer_rules import (
    AppealAttempt,
    DenialRecord,
    PayerRuleEntry,
    PayerRulesIndex,
)


def _denial(
    denial_id: str,
    payer_id: str = "BCBS_TX",
    denial_reason: str = "Medical necessity not established",
    cpt_codes: list[str] | None = None,
    appeal_attempts: list[AppealAttempt] | None = None,
) -> DenialRecord:
    return DenialRecord(
        denial_id=denial_id,
        payer_id=payer_id,
        cpt_codes=cpt_codes or ["99283"],
        icd_codes=["R07.9"],
        denial_reason=denial_reason,
        denial_code="CO-50",
        policy_citation="LCD L34462",
        date="2024-06-01",
        resolved=False,
        resolution=None,
        source="denial_eob",
        appeal_attempts=appeal_attempts or [],
    )


def _attempt(
    attempt_id: str,
    strategy: str,
    outcome: str | None,
    timestamp: str,
    notes: str | None = None,
) -> AppealAttempt:
    return AppealAttempt(
        attempt_id=attempt_id,
        strategy=strategy,
        outcome=outcome,
        notes=notes,
        timestamp=timestamp,
    )


def test_retrieve_excludes_pending_outcomes(tmp_path: Path) -> None:
    index = PayerRulesIndex(tmp_path / "payer_rules_index.json")
    index.add_denial(
        _denial(
            "denial-1",
            appeal_attempts=[
                _attempt(
                    "attempt-1",
                    "clinical_documentation_appeal",
                    "success",
                    "2024-06-01T10:00:00+00:00",
                )
            ],
        )
    )
    index.add_denial(
        _denial(
            "denial-2",
            appeal_attempts=[
                _attempt(
                    "attempt-2",
                    "policy_exception_appeal",
                    "success",
                    "2024-06-02T10:00:00+00:00",
                )
            ],
        )
    )
    index.add_denial(
        _denial(
            "denial-3",
            appeal_attempts=[
                _attempt(
                    "attempt-3",
                    "retroactive_auth_appeal",
                    "pending",
                    "2024-06-03T10:00:00+00:00",
                )
            ],
        )
    )

    precedents = AppealPrecedentsIndex(index).retrieve(
        AppealPrecedentQuery(
            payer_id="BCBS_TX",
            denial_reason="Medical necessity not established",
            cpt_codes=["99283"],
            top_k=5,
        )
    )

    assert len(precedents) == 2
    assert all(result["outcome"] != "pending" for result in precedents)


def test_record_outcome_persists_via_shared_index(tmp_path: Path) -> None:
    index_path = tmp_path / "payer_rules_index.json"
    index = PayerRulesIndex(index_path)
    index.add_denial(_denial("denial-save-1"))
    index.save()

    appeal_index = AppealPrecedentsIndex(index)
    appeal_index.record_outcome(
        "denial-save-1",
        _attempt(
            "attempt-save-1",
            "clinical_documentation_appeal",
            "success",
            "2024-06-04T11:00:00+00:00",
            notes="Submitted additional PCR addendum",
        ),
    )

    reloaded = PayerRulesIndex(index_path)
    saved_denial = reloaded.get_denial_by_id("denial-save-1")
    assert saved_denial is not None
    assert len(saved_denial.appeal_attempts) == 1
    assert saved_denial.appeal_attempts[0].strategy == "clinical_documentation_appeal"
    assert saved_denial.appeal_attempts[0].outcome == "success"


def test_retrieve_sorted_by_timestamp_descending(tmp_path: Path) -> None:
    index = PayerRulesIndex(tmp_path / "payer_rules_index.json")
    index.add_denial(
        _denial(
            "denial-sort-1",
            appeal_attempts=[
                _attempt(
                    "attempt-sort-older",
                    "strategy-older",
                    "failure",
                    "2024-06-01T09:00:00+00:00",
                )
            ],
        )
    )
    index.add_denial(
        _denial(
            "denial-sort-2",
            appeal_attempts=[
                _attempt(
                    "attempt-sort-newer",
                    "strategy-newer",
                    "success",
                    "2024-06-03T09:00:00+00:00",
                )
            ],
        )
    )

    precedents = AppealPrecedentsIndex(index).retrieve(
        AppealPrecedentQuery(
            payer_id="BCBS_TX",
            denial_reason="Medical necessity not established",
            cpt_codes=["99283"],
            top_k=5,
        )
    )

    assert [entry["timestamp"] for entry in precedents] == [
        "2024-06-03T09:00:00+00:00",
        "2024-06-01T09:00:00+00:00",
    ]


def test_cross_index_consistency_after_record_outcome(tmp_path: Path) -> None:
    index_path = tmp_path / "payer_rules_index.json"
    index = PayerRulesIndex(index_path)
    index.add_denial(_denial("denial-consistency-1"))
    index.add_rule(
        PayerRuleEntry(
            rule_id="rule-consistency-1",
            payer_id="BCBS_TX",
            rule_type="medical_necessity",
            codes_affected=["99283", "R07.9"],
            description="Rule remains readable after appeal attempt writes",
            common_denial_reason="Medical necessity not established",
            required_documentation=["clinical_notes"],
            derived_from_denial_ids=["denial-consistency-1"],
            confidence=0.8,
            source="synthesize_payer_rules",
        )
    )
    index.save()

    AppealPrecedentsIndex(index).record_outcome(
        "denial-consistency-1",
        _attempt(
            "attempt-consistency-1",
            "clinical_documentation_appeal",
            "success",
            "2024-06-05T08:00:00+00:00",
        ),
    )

    with patch.dict("os.environ", {"EMS_PAYER_RULES_INDEX": str(index_path)}):
        payer_ctx = retrieve_payer_requirements("BCBS_TX")

    assert payer_ctx["payer_id"] == "BCBS_TX"
    assert len(payer_ctx["rules"]) == 1
    assert payer_ctx["rules"][0].rule_id == "rule-consistency-1"
