"""Tests for Agent-3: Claims Filing Agent.

Covers:
- validate_claim()          — flag generation for missing/empty fields and provenance warnings
- remediation loop          — Agent3Result.remediation_requested when error flags present
- file_claim()              — confirmation dict shape
- rehydrate_from_existing_claim() — SessionContext reconstruction from Claim JSON
- run()                     — full stub path, immutability, options
- Agent3Options / Agent3Result — model constraints
"""

from __future__ import annotations

import json

import pytest

from ems_pipeline.agents.agent3 import (
    Agent3Options,
    Agent3Result,
    file_claim,
    rehydrate_from_existing_claim,
    run,
    validate_claim,
)
from ems_pipeline.models import Claim, ProvenanceLink
from ems_pipeline.session import SessionContext


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _fully_populated_session(encounter_id: str = "enc-agent3-test") -> SessionContext:
    """A SessionContext with Agent-2 fields populated — should pass validation cleanly."""
    session = SessionContext.create(encounter_id=encounter_id)
    return session.model_copy(
        update={
            "report_draft": (
                "Patient presented with chest pain. "
                "Aspirin 324 mg administered. Transport to Regional Medical."
            ),
            "clinical_reasoning": "Chest pain with elevated BP consistent with ACS protocol.",
            "code_suggestions": [
                {
                    "code": "R07.9",
                    "type": "ICD",
                    "rationale": "Primary impression: chest pain",
                    "evidence_segment_ids": ["seg_0001"],
                },
                {
                    "code": "99283",
                    "type": "CPT",
                    "rationale": "Intervention: aspirin 324 mg",
                    "evidence_segment_ids": ["seg_0002"],
                },
            ],
            "citation_map": {"chest pain": ["seg_0001"]},
        }
    )


def _minimal_claim(
    claim_id: str = "abc123",
    with_interventions: bool = True,
) -> Claim:
    """Minimal Claim matching schema_version 0.1 for rehydration tests."""
    fields: dict = {
        "primary_impression": {
            "value": "chest pain",
            "evidence_segment_ids": ["seg_0001"],
        },
        "interventions": (
            [
                {
                    "kind": "medication",
                    "text": "aspirin",
                    "normalized": "aspirin 324 mg",
                    "evidence_segment_ids": ["seg_0002"],
                }
            ]
            if with_interventions
            else []
        ),
    }
    return Claim(claim_id=claim_id, fields=fields, provenance=[])


# ---------------------------------------------------------------------------
# validate_claim — no flags for a fully populated session
# ---------------------------------------------------------------------------


def test_validate_claim_fully_populated_no_error_flags() -> None:
    """A session with report_draft and code_suggestions yields no error flags."""
    session = _fully_populated_session()
    flags = validate_claim(session, {})
    error_flags = [f for f in flags if f["severity"] == "error"]
    assert error_flags == []


def test_validate_claim_fully_populated_returns_list() -> None:
    session = _fully_populated_session()
    flags = validate_claim(session, {})
    assert isinstance(flags, list)


def test_validate_claim_flag_has_required_keys() -> None:
    """Every flag dict has the four required keys."""
    session = SessionContext.create(encounter_id="enc-flagtest")
    flags = validate_claim(session, {})
    for flag in flags:
        assert "field" in flag
        assert "issue" in flag
        assert "severity" in flag
        assert "remediation_hint" in flag


# ---------------------------------------------------------------------------
# validate_claim — report_draft missing
# ---------------------------------------------------------------------------


def test_validate_claim_report_draft_none_yields_error_flag() -> None:
    """report_draft=None must produce exactly one error flag for that field."""
    session = _fully_populated_session().model_copy(update={"report_draft": None})
    flags = validate_claim(session, {})
    report_errors = [
        f for f in flags if f["field"] == "report_draft" and f["severity"] == "error"
    ]
    assert len(report_errors) == 1


def test_validate_claim_empty_report_draft_yields_error_flag() -> None:
    """An empty string for report_draft is also treated as missing."""
    session = _fully_populated_session().model_copy(update={"report_draft": ""})
    flags = validate_claim(session, {})
    report_errors = [
        f for f in flags if f["field"] == "report_draft" and f["severity"] == "error"
    ]
    assert len(report_errors) == 1


def test_validate_claim_report_error_contains_remediation_hint() -> None:
    session = _fully_populated_session().model_copy(update={"report_draft": None})
    flags = validate_claim(session, {})
    report_errors = [f for f in flags if f["field"] == "report_draft"]
    assert report_errors[0]["remediation_hint"] != ""


# ---------------------------------------------------------------------------
# validate_claim — code_suggestions missing or empty
# ---------------------------------------------------------------------------


def test_validate_claim_no_code_suggestions_yields_error() -> None:
    session = _fully_populated_session().model_copy(update={"code_suggestions": None})
    flags = validate_claim(session, {})
    code_errors = [
        f for f in flags if f["field"] == "code_suggestions" and f["severity"] == "error"
    ]
    assert len(code_errors) == 1


def test_validate_claim_empty_code_suggestions_yields_error() -> None:
    session = _fully_populated_session().model_copy(update={"code_suggestions": []})
    flags = validate_claim(session, {})
    code_errors = [
        f for f in flags if f["field"] == "code_suggestions" and f["severity"] == "error"
    ]
    assert len(code_errors) == 1


# ---------------------------------------------------------------------------
# validate_claim — provenance quality warning
# ---------------------------------------------------------------------------


def test_validate_claim_codes_without_evidence_yield_warning() -> None:
    """Code suggestions that lack evidence_segment_ids produce a warning (not error)."""
    session = _fully_populated_session().model_copy(
        update={
            "code_suggestions": [
                {"code": "R07.9", "type": "ICD", "rationale": "chest pain", "evidence_segment_ids": []},
            ]
        }
    )
    flags = validate_claim(session, {})
    warnings = [f for f in flags if f["severity"] == "warning"]
    assert len(warnings) >= 1


def test_validate_claim_codes_with_evidence_no_provenance_warning() -> None:
    """Codes with evidence_segment_ids do NOT trigger the provenance warning."""
    session = _fully_populated_session()
    flags = validate_claim(session, {})
    provenance_warnings = [
        f
        for f in flags
        if f["severity"] == "warning" and "evidence" in f["issue"].lower()
    ]
    assert provenance_warnings == []


# ---------------------------------------------------------------------------
# validate_claim — payer_ctx is ignored (stub)
# ---------------------------------------------------------------------------


def test_validate_claim_non_empty_payer_ctx_does_not_crash() -> None:
    """Passing a non-empty payer_ctx does not raise (stub absorbs it)."""
    session = _fully_populated_session()
    flags = validate_claim(session, {"requires_ALS_doc": True, "max_miles": 50})
    assert isinstance(flags, list)


# ---------------------------------------------------------------------------
# Remediation loop
# ---------------------------------------------------------------------------


def test_remediation_requested_when_errors_and_max_loops_positive() -> None:
    """Errors + max_remediation_loops=1 → remediation_requested=True, submitted=False."""
    session = SessionContext.create(encounter_id="enc-remediation")
    result = run(session, Agent3Options(max_remediation_loops=1))
    assert result.remediation_requested is True
    assert result.submitted is False


def test_remediation_notes_are_non_empty_strings() -> None:
    session = SessionContext.create(encounter_id="enc-rem-notes")
    result = run(session, Agent3Options(max_remediation_loops=1))
    assert len(result.remediation_notes) > 0
    for note in result.remediation_notes:
        assert isinstance(note, str)
        assert len(note) > 0


def test_remediation_sets_session_remediation_request_field() -> None:
    """When remediation is triggered, session.remediation_request is populated."""
    session = SessionContext.create(encounter_id="enc-rem-field")
    result = run(session, Agent3Options(max_remediation_loops=1))
    rr = result.updated_session.remediation_request
    assert rr is not None
    assert "notes" in rr
    assert "loop_count" in rr
    assert rr["loop_count"] == 1


def test_remediation_not_triggered_when_max_loops_zero() -> None:
    """max_remediation_loops=0 bypasses remediation even when errors exist."""
    session = SessionContext.create(encounter_id="enc-no-rem")
    result = run(session, Agent3Options(max_remediation_loops=0))
    # With errors but max_loops=0, it should proceed to file_claim (stub → submitted).
    assert result.remediation_requested is False
    assert result.submitted is True


def test_valid_session_does_not_trigger_remediation() -> None:
    session = _fully_populated_session()
    result = run(session, Agent3Options())
    assert result.remediation_requested is False


# ---------------------------------------------------------------------------
# run() — successful submission path
# ---------------------------------------------------------------------------


def test_run_submits_valid_session() -> None:
    session = _fully_populated_session()
    result = run(session, Agent3Options())
    assert result.submitted is True


def test_run_populates_claim_id_on_session() -> None:
    session = _fully_populated_session()
    result = run(session, Agent3Options())
    assert result.updated_session.claim_id is not None
    assert len(result.updated_session.claim_id) > 0


def test_run_populates_submission_status_submitted() -> None:
    session = _fully_populated_session()
    result = run(session, Agent3Options())
    assert result.updated_session.submission_status == "submitted"


def test_run_stores_pre_submission_flags() -> None:
    """pre_submission_flags on session must be a list (may be empty for clean sessions)."""
    session = _fully_populated_session()
    result = run(session, Agent3Options())
    assert result.updated_session.pre_submission_flags is not None
    assert isinstance(result.updated_session.pre_submission_flags, list)


def test_run_does_not_mutate_input_session() -> None:
    """Original session is not mutated — immutable write pattern."""
    session = _fully_populated_session()
    result = run(session, Agent3Options())
    assert session.claim_id is None
    assert session.submission_status is None
    assert result.updated_session.claim_id is not None
    assert result.updated_session.encounter_id == session.encounter_id


def test_run_with_payer_id_does_not_crash() -> None:
    session = _fully_populated_session()
    result = run(session, Agent3Options(payer_id="PAYER_XYZ"))
    assert result.submitted is True
    assert result.updated_session.payer_id == "PAYER_XYZ"


def test_run_empty_session_does_not_crash() -> None:
    """run() on a nearly empty session (no Agent-2 data) completes without exception."""
    session = SessionContext.create(encounter_id="enc-empty-agent3")
    result = run(session, Agent3Options(max_remediation_loops=0))
    # max_loops=0 → skip remediation, file stub → submitted
    assert result.submitted is True


def test_run_result_flags_list_is_present() -> None:
    session = _fully_populated_session()
    result = run(session, Agent3Options())
    assert isinstance(result.flags, list)


# ---------------------------------------------------------------------------
# file_claim — direct tests
# ---------------------------------------------------------------------------


def test_file_claim_returns_claim_id() -> None:
    session = _fully_populated_session()
    result = file_claim(session, payer_id=None)
    assert "claim_id" in result
    assert isinstance(result["claim_id"], str)
    assert len(result["claim_id"]) > 0


def test_file_claim_returns_submitted_status() -> None:
    session = _fully_populated_session()
    result = file_claim(session, payer_id=None)
    assert result["status"] == "submitted"


def test_file_claim_returns_timestamp() -> None:
    session = _fully_populated_session()
    result = file_claim(session, payer_id=None)
    assert "timestamp" in result
    assert isinstance(result["timestamp"], str)


def test_file_claim_preserves_existing_claim_id() -> None:
    """If session already has a claim_id, file_claim reuses it."""
    session = _fully_populated_session().model_copy(update={"claim_id": "preexisting-id"})
    result = file_claim(session, payer_id=None)
    assert result["claim_id"] == "preexisting-id"


def test_file_claim_generates_uuid_when_no_claim_id() -> None:
    session = _fully_populated_session()
    assert session.claim_id is None
    result = file_claim(session, payer_id=None)
    # Should be a valid UUID-like string (36 chars with dashes)
    assert len(result["claim_id"]) == 36


def test_file_claim_records_payer_id() -> None:
    session = _fully_populated_session()
    result = file_claim(session, payer_id="PAYER_ABC")
    assert result["payer_id"] == "PAYER_ABC"


# ---------------------------------------------------------------------------
# rehydrate_from_existing_claim
# ---------------------------------------------------------------------------


def test_rehydrate_sets_claim_id(tmp_path) -> None:
    claim = _minimal_claim(claim_id="rehydrate-id-001")
    path = tmp_path / "claim.json"
    path.write_text(claim.model_dump_json(indent=2), encoding="utf-8")

    session = rehydrate_from_existing_claim(path)
    assert session.claim_id == "rehydrate-id-001"


def test_rehydrate_derives_icd_code_from_primary_impression(tmp_path) -> None:
    claim = _minimal_claim()
    path = tmp_path / "claim.json"
    path.write_text(claim.model_dump_json(indent=2), encoding="utf-8")

    session = rehydrate_from_existing_claim(path)
    assert session.code_suggestions is not None
    icd = [c for c in session.code_suggestions if c["type"] == "ICD"]
    assert len(icd) == 1
    assert "chest pain" in icd[0]["rationale"]


def test_rehydrate_derives_cpt_code_from_interventions(tmp_path) -> None:
    claim = _minimal_claim(with_interventions=True)
    path = tmp_path / "claim.json"
    path.write_text(claim.model_dump_json(indent=2), encoding="utf-8")

    session = rehydrate_from_existing_claim(path)
    assert session.code_suggestions is not None
    cpt = [c for c in session.code_suggestions if c["type"] == "CPT"]
    assert len(cpt) == 1
    assert "aspirin" in cpt[0]["rationale"]


def test_rehydrate_no_interventions_only_icd(tmp_path) -> None:
    claim = _minimal_claim(with_interventions=False)
    path = tmp_path / "claim.json"
    path.write_text(claim.model_dump_json(indent=2), encoding="utf-8")

    session = rehydrate_from_existing_claim(path)
    assert session.code_suggestions is not None
    cpt = [c for c in session.code_suggestions if c["type"] == "CPT"]
    assert cpt == []


def test_rehydrate_sets_submission_status_pending(tmp_path) -> None:
    claim = _minimal_claim()
    path = tmp_path / "claim.json"
    path.write_text(claim.model_dump_json(indent=2), encoding="utf-8")

    session = rehydrate_from_existing_claim(path)
    assert session.submission_status == "pending"


def test_rehydrate_creates_fresh_encounter_id(tmp_path) -> None:
    """Rehydrated session has a new encounter_id (not from the claim)."""
    claim = _minimal_claim()
    path = tmp_path / "claim.json"
    path.write_text(claim.model_dump_json(indent=2), encoding="utf-8")

    session = rehydrate_from_existing_claim(path)
    assert session.encounter_id is not None
    assert len(session.encounter_id) > 0


def test_rehydrate_claim_roundtrips_through_agent3(tmp_path) -> None:
    """A rehydrated session can be submitted successfully by Agent-3."""
    claim = _minimal_claim()
    path = tmp_path / "claim.json"
    path.write_text(claim.model_dump_json(indent=2), encoding="utf-8")

    session = rehydrate_from_existing_claim(path)
    # Inject a minimal report_draft so validation passes
    session = session.model_copy(
        update={"report_draft": "Rehydrated report: chest pain encounter."}
    )
    result = run(session, Agent3Options(max_remediation_loops=0))
    assert result.submitted is True


# ---------------------------------------------------------------------------
# Agent3Options model constraints
# ---------------------------------------------------------------------------


def test_agent3_options_defaults() -> None:
    opts = Agent3Options()
    assert opts.payer_id is None
    assert opts.llm_client is None
    assert opts.max_remediation_loops == 1


def test_agent3_options_extra_fields_forbidden() -> None:
    with pytest.raises(Exception):
        Agent3Options(unknown_field="oops")  # type: ignore[call-arg]


def test_agent3_options_max_loops_zero_accepted() -> None:
    opts = Agent3Options(max_remediation_loops=0)
    assert opts.max_remediation_loops == 0


# ---------------------------------------------------------------------------
# Agent3Result model structure
# ---------------------------------------------------------------------------


def test_agent3_result_has_all_fields() -> None:
    session = _fully_populated_session()
    result = run(session, Agent3Options())
    assert hasattr(result, "updated_session")
    assert hasattr(result, "submitted")
    assert hasattr(result, "flags")
    assert hasattr(result, "remediation_requested")
    assert hasattr(result, "remediation_notes")


def test_agent3_result_submitted_is_bool() -> None:
    result = run(_fully_populated_session(), Agent3Options())
    assert isinstance(result.submitted, bool)


def test_agent3_result_remediation_notes_is_list() -> None:
    result = run(_fully_populated_session(), Agent3Options())
    assert isinstance(result.remediation_notes, list)


# ---------------------------------------------------------------------------
# SessionContext.remediation_request round-trip
# ---------------------------------------------------------------------------


def test_remediation_request_round_trips_through_json(tmp_path) -> None:
    """remediation_request dict survives to_json → from_json."""
    session = SessionContext.create(encounter_id="enc-rr-rt")
    notes = ["[ERROR] report_draft: missing — Hint: run Agent-2"]
    updated = session.model_copy(
        update={
            "remediation_request": {
                "notes": notes,
                "payer_id": "PAYER_Z",
                "loop_count": 1,
            }
        }
    )
    path = tmp_path / "session_rr.json"
    updated.to_json(path)
    recovered = SessionContext.from_json(path)
    assert recovered.remediation_request is not None
    assert recovered.remediation_request["loop_count"] == 1
    assert recovered.remediation_request["notes"] == notes


def test_remediation_request_defaults_to_none() -> None:
    session = SessionContext.create(encounter_id="enc-rr-none")
    assert session.remediation_request is None
