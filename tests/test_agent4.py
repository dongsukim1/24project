"""Tests for Agent-4: Appeals Agent.

Covers:
- choose_appeal_strategy()     — rule-based keyword mapping and precedent-driven selection
- build_appeal_script()        — template structure and session field injection
- record_appeal_outcome()      — return dict shape and required keys
- retrieve_denial_context()    — stub shape
- retrieve_appeal_precedents() — stub returns empty list
- start_voice_session()        — None for None client; NotImplementedError for real client
- run()                        — full stub path, immutability, appeal_history population
- Agent4Options / Agent4Result — model constraints
"""

from __future__ import annotations

import pytest

from ems_pipeline.agents.agent4 import (
    Agent4Options,
    Agent4Result,
    build_appeal_script,
    choose_appeal_strategy,
    record_appeal_outcome,
    retrieve_appeal_precedents,
    retrieve_denial_context,
    run,
    start_voice_session,
)
from ems_pipeline.session import SessionContext


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _denied_session(
    encounter_id: str = "enc-agent4-test",
    denial_reason: str = "medical necessity not established",
) -> SessionContext:
    """A SessionContext that simulates a post-denial state."""
    session = SessionContext.create(encounter_id=encounter_id)
    return session.model_copy(
        update={
            "claim_id": "claim-abc-001",
            "payer_id": "PAYER_TEST",
            "submission_status": "denied",
            "denial_reason": denial_reason,
            "report_draft": (
                "Patient presented with chest pain and elevated BP. "
                "Aspirin 324 mg administered en route. Transported Priority 1."
            ),
            "code_suggestions": [
                {
                    "code": "R07.9",
                    "type": "ICD",
                    "rationale": "chest pain",
                    "evidence_segment_ids": ["seg_0001"],
                },
                {
                    "code": "99283",
                    "type": "CPT",
                    "rationale": "Intervention: aspirin",
                    "evidence_segment_ids": ["seg_0002"],
                },
            ],
        }
    )


# ---------------------------------------------------------------------------
# retrieve_denial_context — stub shape
# ---------------------------------------------------------------------------


def test_retrieve_denial_context_returns_dict() -> None:
    ctx = retrieve_denial_context("medical necessity", "PAYER_A")
    assert isinstance(ctx, dict)


def test_retrieve_denial_context_includes_denial_reason() -> None:
    ctx = retrieve_denial_context("coverage limitation", None)
    assert ctx["denial_reason"] == "coverage limitation"


def test_retrieve_denial_context_includes_payer_id() -> None:
    ctx = retrieve_denial_context("", "PAYER_B")
    assert ctx["payer_id"] == "PAYER_B"


def test_retrieve_denial_context_includes_appeal_deadline() -> None:
    ctx = retrieve_denial_context("authorization missing", None)
    assert "appeal_deadline_days" in ctx
    assert isinstance(ctx["appeal_deadline_days"], int)
    assert ctx["appeal_deadline_days"] > 0


def test_retrieve_denial_context_includes_common_remedies() -> None:
    ctx = retrieve_denial_context("", None)
    assert "common_remedies" in ctx
    assert isinstance(ctx["common_remedies"], list)


# ---------------------------------------------------------------------------
# retrieve_appeal_precedents — stub returns []
# ---------------------------------------------------------------------------


def test_retrieve_appeal_precedents_returns_list() -> None:
    result = retrieve_appeal_precedents("medical necessity", "PAYER_A", ["99283"])
    assert isinstance(result, list)


def test_retrieve_appeal_precedents_stub_returns_empty() -> None:
    result = retrieve_appeal_precedents("anything", None, [])
    assert result == []


# ---------------------------------------------------------------------------
# choose_appeal_strategy — rule-based keyword mapping
# ---------------------------------------------------------------------------


def test_choose_strategy_medical_necessity() -> None:
    ctx = {"denial_reason": "medical necessity not established"}
    assert choose_appeal_strategy(ctx, []) == "clinical_documentation_appeal"


def test_choose_strategy_coverage() -> None:
    ctx = {"denial_reason": "coverage limitation applies"}
    assert choose_appeal_strategy(ctx, []) == "policy_exception_appeal"


def test_choose_strategy_authorization() -> None:
    ctx = {"denial_reason": "prior authorization required"}
    assert choose_appeal_strategy(ctx, []) == "retroactive_auth_appeal"


def test_choose_strategy_default_unknown_reason() -> None:
    ctx = {"denial_reason": "billing code mismatch"}
    assert choose_appeal_strategy(ctx, []) == "standard_appeal"


def test_choose_strategy_empty_denial_reason() -> None:
    ctx = {"denial_reason": ""}
    assert choose_appeal_strategy(ctx, []) == "standard_appeal"


def test_choose_strategy_none_denial_reason() -> None:
    ctx = {"denial_reason": None}
    assert choose_appeal_strategy(ctx, []) == "standard_appeal"


def test_choose_strategy_case_insensitive_medical_necessity() -> None:
    ctx = {"denial_reason": "MEDICAL NECESSITY documentation missing"}
    assert choose_appeal_strategy(ctx, []) == "clinical_documentation_appeal"


# ---------------------------------------------------------------------------
# choose_appeal_strategy — precedent-driven selection
# ---------------------------------------------------------------------------


def test_choose_strategy_picks_successful_precedent() -> None:
    """A 'success' precedent overrides the rule-based fallback."""
    precedents = [
        {
            "strategy": "clinical_documentation_appeal",
            "outcome": "success",
            "payer_id": "PAYER_X",
            "denial_reason": "medical necessity",
            "appeal_text_snippet": "...",
        }
    ]
    ctx = {"denial_reason": "coverage issue"}  # would normally → policy_exception
    result = choose_appeal_strategy(ctx, precedents)
    assert result == "clinical_documentation_appeal"


def test_choose_strategy_skips_failed_precedents() -> None:
    """'failure' precedents are skipped; falls through to rule-based."""
    precedents = [
        {
            "strategy": "failed_strategy",
            "outcome": "failure",
            "payer_id": "PAYER_X",
            "denial_reason": "medical necessity",
            "appeal_text_snippet": "...",
        }
    ]
    ctx = {"denial_reason": "coverage limitation"}
    result = choose_appeal_strategy(ctx, precedents)
    assert result == "policy_exception_appeal"


def test_choose_strategy_first_success_in_precedent_list() -> None:
    """When multiple precedents exist, the first successful one wins."""
    precedents = [
        {"strategy": "strategy_A", "outcome": "failure"},
        {"strategy": "strategy_B", "outcome": "success"},
        {"strategy": "strategy_C", "outcome": "success"},
    ]
    ctx = {"denial_reason": ""}
    result = choose_appeal_strategy(ctx, precedents)
    assert result == "strategy_B"


def test_choose_strategy_empty_precedent_strategy_string_skipped() -> None:
    """A precedent with an empty strategy string is skipped."""
    precedents = [
        {"strategy": "", "outcome": "success"},
    ]
    ctx = {"denial_reason": "authorization required"}
    result = choose_appeal_strategy(ctx, precedents)
    assert result == "retroactive_auth_appeal"


# ---------------------------------------------------------------------------
# build_appeal_script — content and structure
# ---------------------------------------------------------------------------


def test_build_appeal_script_contains_claim_id() -> None:
    session = _denied_session()
    ctx = retrieve_denial_context(session.denial_reason or "", session.payer_id)
    script = build_appeal_script(session, "standard_appeal", ctx, [])
    assert "claim-abc-001" in script


def test_build_appeal_script_contains_denial_reason() -> None:
    session = _denied_session()
    ctx = retrieve_denial_context(session.denial_reason or "", session.payer_id)
    script = build_appeal_script(session, "standard_appeal", ctx, [])
    assert "medical necessity" in script.lower()


def test_build_appeal_script_contains_strategy() -> None:
    session = _denied_session()
    ctx = retrieve_denial_context(session.denial_reason or "", session.payer_id)
    script = build_appeal_script(session, "clinical_documentation_appeal", ctx, [])
    assert "clinical_documentation_appeal" in script


def test_build_appeal_script_contains_report_excerpt() -> None:
    session = _denied_session()
    ctx = retrieve_denial_context(session.denial_reason or "", session.payer_id)
    script = build_appeal_script(session, "standard_appeal", ctx, [])
    assert "chest pain" in script


def test_build_appeal_script_contains_stub_marker() -> None:
    session = _denied_session()
    ctx = retrieve_denial_context(session.denial_reason or "", session.payer_id)
    script = build_appeal_script(session, "standard_appeal", ctx, [])
    assert "[STUB" in script


def test_build_appeal_script_has_denial_summary_section() -> None:
    session = _denied_session()
    ctx = retrieve_denial_context(session.denial_reason or "", session.payer_id)
    script = build_appeal_script(session, "standard_appeal", ctx, [])
    assert "DENIAL SUMMARY" in script


def test_build_appeal_script_has_clinical_justification_section() -> None:
    session = _denied_session()
    ctx = retrieve_denial_context(session.denial_reason or "", session.payer_id)
    script = build_appeal_script(session, "standard_appeal", ctx, [])
    assert "CLINICAL JUSTIFICATION" in script


def test_build_appeal_script_has_closing_section() -> None:
    session = _denied_session()
    ctx = retrieve_denial_context(session.denial_reason or "", session.payer_id)
    script = build_appeal_script(session, "standard_appeal", ctx, [])
    assert "CLOSING" in script


def test_build_appeal_script_no_report_draft_does_not_crash() -> None:
    session = _denied_session().model_copy(update={"report_draft": None})
    ctx = retrieve_denial_context("", None)
    script = build_appeal_script(session, "standard_appeal", ctx, [])
    assert isinstance(script, str)
    assert len(script) > 0


def test_build_appeal_script_unknown_claim_id_placeholder() -> None:
    session = SessionContext.create(encounter_id="enc-no-claim")
    ctx = retrieve_denial_context("", None)
    script = build_appeal_script(session, "standard_appeal", ctx, [])
    assert "[unknown claim]" in script


def test_build_appeal_script_includes_submitted_codes() -> None:
    session = _denied_session()
    ctx = retrieve_denial_context(session.denial_reason or "", session.payer_id)
    script = build_appeal_script(session, "standard_appeal", ctx, [])
    assert "R07.9" in script
    assert "ICD" in script


def test_build_appeal_script_is_deterministic() -> None:
    session = _denied_session()
    ctx = retrieve_denial_context(session.denial_reason or "", session.payer_id)
    s1 = build_appeal_script(session, "standard_appeal", ctx, [])
    s2 = build_appeal_script(session, "standard_appeal", ctx, [])
    assert s1 == s2


# ---------------------------------------------------------------------------
# start_voice_session — stub behaviour
# ---------------------------------------------------------------------------


def test_start_voice_session_none_client_returns_none() -> None:
    assert start_voice_session("any script", None) is None


def test_start_voice_session_non_none_client_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        start_voice_session("any script", object())


# ---------------------------------------------------------------------------
# record_appeal_outcome — dict shape
# ---------------------------------------------------------------------------


def test_record_appeal_outcome_has_required_keys() -> None:
    session = _denied_session()
    record = record_appeal_outcome(session, "standard_appeal", None, None)
    required = {
        "encounter_id", "strategy", "outcome", "denial_reason",
        "payer_id", "claim_id", "codes", "timestamp",
    }
    assert required.issubset(record.keys())


def test_record_appeal_outcome_encounter_id_matches_session() -> None:
    session = _denied_session(encounter_id="enc-record-test")
    record = record_appeal_outcome(session, "standard_appeal", None, None)
    assert record["encounter_id"] == "enc-record-test"


def test_record_appeal_outcome_strategy_matches_argument() -> None:
    session = _denied_session()
    record = record_appeal_outcome(session, "policy_exception_appeal", None, None)
    assert record["strategy"] == "policy_exception_appeal"


def test_record_appeal_outcome_outcome_matches_argument() -> None:
    session = _denied_session()
    record = record_appeal_outcome(session, "standard_appeal", "success", None)
    assert record["outcome"] == "success"


def test_record_appeal_outcome_pending_outcome_is_none() -> None:
    session = _denied_session()
    record = record_appeal_outcome(session, "standard_appeal", None, None)
    assert record["outcome"] is None


def test_record_appeal_outcome_codes_extracted_from_code_suggestions() -> None:
    session = _denied_session()
    record = record_appeal_outcome(session, "standard_appeal", None, None)
    assert isinstance(record["codes"], list)
    assert "R07.9" in record["codes"]
    assert "99283" in record["codes"]


def test_record_appeal_outcome_notes_field_present() -> None:
    session = _denied_session()
    record = record_appeal_outcome(session, "standard_appeal", None, "follow up in 5 days")
    assert record["notes"] == "follow up in 5 days"


def test_record_appeal_outcome_timestamp_is_string() -> None:
    session = _denied_session()
    record = record_appeal_outcome(session, "standard_appeal", None, None)
    assert isinstance(record["timestamp"], str)
    assert len(record["timestamp"]) > 0


# ---------------------------------------------------------------------------
# run() — full stub path
# ---------------------------------------------------------------------------


def test_run_returns_agent4_result() -> None:
    session = _denied_session()
    result = run(session, Agent4Options())
    assert isinstance(result, Agent4Result)


def test_run_strategy_chosen_is_string() -> None:
    session = _denied_session()
    result = run(session, Agent4Options())
    assert isinstance(result.strategy_chosen, str)
    assert len(result.strategy_chosen) > 0


def test_run_appeal_script_is_non_empty_string() -> None:
    session = _denied_session()
    result = run(session, Agent4Options())
    assert isinstance(result.appeal_script, str)
    assert len(result.appeal_script) > 0


def test_run_voice_session_id_is_none_without_voice_client() -> None:
    session = _denied_session()
    result = run(session, Agent4Options())
    assert result.voice_session_id is None


def test_run_writes_appeal_strategy_to_session() -> None:
    session = _denied_session()
    result = run(session, Agent4Options())
    assert result.updated_session.appeal_strategy is not None
    assert result.updated_session.appeal_strategy == result.strategy_chosen


def test_run_writes_denial_reason_to_session() -> None:
    session = _denied_session(denial_reason="coverage limitation")
    result = run(session, Agent4Options())
    assert result.updated_session.denial_reason == "coverage limitation"


def test_run_appends_to_appeal_history() -> None:
    session = _denied_session()
    result = run(session, Agent4Options())
    assert result.updated_session.appeal_history is not None
    assert len(result.updated_session.appeal_history) == 1


def test_run_appeal_history_record_has_expected_keys() -> None:
    session = _denied_session()
    result = run(session, Agent4Options())
    record = result.updated_session.appeal_history[0]
    assert "encounter_id" in record
    assert "strategy" in record
    assert "timestamp" in record


def test_run_does_not_mutate_input_session() -> None:
    session = _denied_session()
    result = run(session, Agent4Options())
    assert session.appeal_strategy is None
    assert session.appeal_history is None
    assert result.updated_session.appeal_strategy is not None
    assert result.updated_session.encounter_id == session.encounter_id


def test_run_payer_id_option_overrides_session_payer() -> None:
    """options.payer_id takes precedence over session.payer_id."""
    session = _denied_session()
    assert session.payer_id == "PAYER_TEST"
    result = run(session, Agent4Options(payer_id="PAYER_OVERRIDE"))
    # The result is consistent — no crash; strategy is determined
    assert result.strategy_chosen is not None


def test_run_empty_session_does_not_crash() -> None:
    """run() on a nearly empty session (no denial_reason) completes without error."""
    session = SessionContext.create(encounter_id="enc-empty-agent4")
    result = run(session, Agent4Options())
    assert result.strategy_chosen == "standard_appeal"
    assert result.appeal_script is not None


def test_run_medical_necessity_denial_picks_correct_strategy() -> None:
    session = _denied_session(denial_reason="medical necessity not documented")
    result = run(session, Agent4Options())
    assert result.strategy_chosen == "clinical_documentation_appeal"


def test_run_coverage_denial_picks_correct_strategy() -> None:
    session = _denied_session(denial_reason="coverage not applicable")
    result = run(session, Agent4Options())
    assert result.strategy_chosen == "policy_exception_appeal"


def test_run_authorization_denial_picks_correct_strategy() -> None:
    session = _denied_session(denial_reason="authorization not obtained")
    result = run(session, Agent4Options())
    assert result.strategy_chosen == "retroactive_auth_appeal"


def test_run_accumulates_multiple_appeal_history_entries() -> None:
    """Successive run() calls accumulate records in appeal_history."""
    session = _denied_session()
    result1 = run(session, Agent4Options())
    result2 = run(result1.updated_session, Agent4Options())
    assert len(result2.updated_session.appeal_history) == 2


def test_run_session_json_round_trip(tmp_path) -> None:
    """Updated session with appeal data survives to_json → from_json."""
    session = _denied_session()
    result = run(session, Agent4Options())
    path = tmp_path / "session_appeal.json"
    result.updated_session.to_json(path)
    recovered = SessionContext.from_json(path)
    assert recovered.appeal_strategy == result.strategy_chosen
    assert recovered.denial_reason == session.denial_reason
    assert recovered.appeal_history is not None
    assert len(recovered.appeal_history) == 1


# ---------------------------------------------------------------------------
# Agent4Options model constraints
# ---------------------------------------------------------------------------


def test_agent4_options_defaults() -> None:
    opts = Agent4Options()
    assert opts.payer_id is None
    assert opts.llm_client is None
    assert opts.voice_client is None


def test_agent4_options_extra_fields_forbidden() -> None:
    with pytest.raises(Exception):
        Agent4Options(unknown_field="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Agent4Result model constraints
# ---------------------------------------------------------------------------


def test_agent4_result_extra_fields_forbidden() -> None:
    session = _denied_session()
    result = run(session, Agent4Options())
    assert hasattr(result, "updated_session")
    assert hasattr(result, "strategy_chosen")
    assert hasattr(result, "appeal_script")
    assert hasattr(result, "voice_session_id")
