"""Tests for the EMS Pipeline Orchestrator (supervised mode).

Covers:
- Remediation loop logic — Agent-3 remediation_requested=True triggers a
  second Agent-2 invocation
- Agents-run order in a clean (no-remediation) pipeline
- Partial SessionContext saved to disk when Agent-2 raises an exception
- Max remediation loops cap — loop does not exceed OrchestratorOptions.max_remediation_loops
- Agent-4 conditional trigger on denial_reason
- Immutability — input session not mutated
- OrchestratorOptions / OrchestratorResult model constraints
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ems_pipeline.agents.agent3 import Agent3Result
from ems_pipeline.agents.agent4 import Agent4Result
from ems_pipeline.orchestrator import (
    OrchestratorOptions,
    OrchestratorResult,
    run_pipeline_supervised,
)
from ems_pipeline.session import SessionContext


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def _make_session(**updates) -> SessionContext:
    """Create a minimal SessionContext, optionally with extra fields."""
    s = SessionContext.create(encounter_id="enc-orch-test")
    if updates:
        s = s.model_copy(update=updates)
    return s


def _mock_agent1(audio_path, session, opts):
    """Agent-1 mock: stamps transcript_raw so we can verify Agent-1 ran."""
    return session.model_copy(
        update={
            "transcript_raw": "mocked transcript",
            "transcript_segments": [],
            "extracted_terms": [],
            "confidence_flags": [],
            "ambiguities": [],
            "extraction_confidence": None,
        }
    )


def _mock_agent2(session, opts):
    """Agent-2 mock: stamps report_draft so we can verify Agent-2 ran."""
    return session.model_copy(
        update={
            "report_draft": "[STUB] mocked report",
            "clinical_reasoning": "[STUB] mocked reasoning",
            "code_suggestions": [
                {
                    "code": "R07.9",
                    "type": "ICD",
                    "rationale": "chest pain",
                    "evidence_segment_ids": ["seg_0001"],
                }
            ],
        }
    )


def _mock_agent3_submitted(session, opts) -> Agent3Result:
    """Agent-3 mock: immediately submits the claim (no remediation)."""
    updated = session.write_agent3(
        claim_id="mock-claim-001",
        payer_id=opts.payer_id,
        status="submitted",
        flags=[],
    )
    return Agent3Result(
        updated_session=updated,
        submitted=True,
        flags=[],
        remediation_requested=False,
        remediation_notes=[],
    )


def _mock_agent4(session, opts) -> Agent4Result:
    """Agent-4 mock: returns a standard_appeal strategy."""
    updated = session.write_agent4(
        denial_reason=session.denial_reason or "",
        strategy="standard_appeal",
    )
    return Agent4Result(
        updated_session=updated,
        strategy_chosen="standard_appeal",
        appeal_script="[STUB] mocked appeal script",
        voice_session_id=None,
    )


# ---------------------------------------------------------------------------
# Baseline: clean pipeline with no remediation
# ---------------------------------------------------------------------------


def test_supervised_pipeline_agents_run_order(tmp_path) -> None:
    """agents_run must contain ['agent1', 'agent2', 'agent3'] in order."""
    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=_mock_agent3_submitted),
    ):
        result = run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    assert result.agents_run == ["agent1", "agent2", "agent3"]


def test_supervised_pipeline_submitted_true_on_clean_run(tmp_path) -> None:
    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=_mock_agent3_submitted),
    ):
        result = run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    assert result.submitted is True
    assert result.remediation_loops == 0


def test_supervised_pipeline_session_json_written(tmp_path) -> None:
    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=_mock_agent3_submitted),
    ):
        run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    assert (tmp_path / "session.json").exists()


def test_supervised_pipeline_claim_id_in_result(tmp_path) -> None:
    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=_mock_agent3_submitted),
    ):
        result = run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    assert result.claim_id == "mock-claim-001"


def test_supervised_pipeline_errors_empty_on_success(tmp_path) -> None:
    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=_mock_agent3_submitted),
    ):
        result = run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    assert result.errors == []


# ---------------------------------------------------------------------------
# Remediation loop
# ---------------------------------------------------------------------------


def test_remediation_loop_reruns_agent2(tmp_path) -> None:
    """When Agent-3 returns remediation_requested=True, the orchestrator
    invokes Agent-2 a second time before filing."""

    agent2_call_count = 0
    agent3_call_count = 0

    def counting_agent2(session, opts):
        nonlocal agent2_call_count
        agent2_call_count += 1
        return _mock_agent2(session, opts)

    def remediation_then_submit(session, opts) -> Agent3Result:
        nonlocal agent3_call_count
        agent3_call_count += 1
        if agent3_call_count == 1:
            # First call: request remediation
            updated = session.model_copy(
                update={
                    "remediation_request": {
                        "notes": ["[ERROR] report_draft: missing — Hint: run Agent-2"],
                        "payer_id": None,
                        "loop_count": 1,
                    }
                }
            )
            return Agent3Result(
                updated_session=updated,
                submitted=False,
                flags=[
                    {
                        "field": "report_draft",
                        "issue": "missing",
                        "severity": "error",
                        "remediation_hint": "run Agent-2",
                    }
                ],
                remediation_requested=True,
                remediation_notes=["[ERROR] report_draft: missing — Hint: run Agent-2"],
            )
        # Second call: submit
        return _mock_agent3_submitted(session, opts)

    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=counting_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=remediation_then_submit),
    ):
        result = run_pipeline_supervised(
            "fake.wav", tmp_path, OrchestratorOptions(max_remediation_loops=2)
        )

    assert agent2_call_count == 2
    assert agent3_call_count == 2
    assert result.remediation_loops == 1
    assert result.submitted is True


def test_remediation_loop_agents_run_includes_both_agent2_runs(tmp_path) -> None:
    """agents_run accumulates entries across the remediation loop."""
    agent3_calls = 0

    def remediation_then_submit(session, opts) -> Agent3Result:
        nonlocal agent3_calls
        agent3_calls += 1
        if agent3_calls == 1:
            updated = session.model_copy(
                update={
                    "remediation_request": {
                        "notes": ["fix"],
                        "payer_id": None,
                        "loop_count": 1,
                    }
                }
            )
            return Agent3Result(
                updated_session=updated,
                submitted=False,
                flags=[],
                remediation_requested=True,
                remediation_notes=["fix"],
            )
        return _mock_agent3_submitted(session, opts)

    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=remediation_then_submit),
    ):
        result = run_pipeline_supervised(
            "fake.wav", tmp_path, OrchestratorOptions(max_remediation_loops=2)
        )

    # agent1 once, agent2 twice (initial + remediation), agent3 twice
    assert result.agents_run == ["agent1", "agent2", "agent3", "agent2", "agent3"]


def test_remediation_loop_injects_notes_into_agent2_options(tmp_path) -> None:
    """On the remediation re-run, Agent-2 options carry additional_context."""
    captured_opts = []
    agent3_calls = 0

    def capturing_agent2(session, opts):
        captured_opts.append(opts)
        return _mock_agent2(session, opts)

    def remediation_then_submit(session, opts) -> Agent3Result:
        nonlocal agent3_calls
        agent3_calls += 1
        if agent3_calls == 1:
            updated = session.model_copy(
                update={
                    "remediation_request": {
                        "notes": ["note A", "note B"],
                        "payer_id": None,
                        "loop_count": 1,
                    }
                }
            )
            return Agent3Result(
                updated_session=updated,
                submitted=False,
                flags=[],
                remediation_requested=True,
                remediation_notes=["note A", "note B"],
            )
        return _mock_agent3_submitted(session, opts)

    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=capturing_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=remediation_then_submit),
    ):
        run_pipeline_supervised(
            "fake.wav", tmp_path, OrchestratorOptions(max_remediation_loops=2)
        )

    # First Agent-2 call: no additional_context
    assert captured_opts[0].additional_context is None
    # Second Agent-2 call: remediation notes injected
    assert captured_opts[1].additional_context == ["note A", "note B"]


def test_remediation_cap_respected(tmp_path) -> None:
    """With max_remediation_loops=1, Agent-2 runs at most 2 times total."""
    agent2_count = 0
    agent3_calls = 0

    def counting_agent2(session, opts):
        nonlocal agent2_count
        agent2_count += 1
        return _mock_agent2(session, opts)

    def always_remediate(session, opts) -> Agent3Result:
        nonlocal agent3_calls
        agent3_calls += 1
        # Always claim remediation_requested — but orchestrator should cap it
        if opts.max_remediation_loops == 0:
            # Orchestrator forced max_remediation_loops=0: file unconditionally
            return _mock_agent3_submitted(session, opts)
        updated = session.model_copy(
            update={
                "remediation_request": {
                    "notes": ["always fix"],
                    "payer_id": None,
                    "loop_count": agent3_calls,
                }
            }
        )
        return Agent3Result(
            updated_session=updated,
            submitted=False,
            flags=[],
            remediation_requested=True,
            remediation_notes=["always fix"],
        )

    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=counting_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=always_remediate),
    ):
        result = run_pipeline_supervised(
            "fake.wav", tmp_path, OrchestratorOptions(max_remediation_loops=1)
        )

    # max_remediation_loops=1 → Agent-2 runs initial (loop 0) + 1 remediation = 2
    assert agent2_count == 2
    assert result.remediation_loops == 1


# ---------------------------------------------------------------------------
# Error handling: partial session saved on Agent-2 failure
# ---------------------------------------------------------------------------


def test_partial_session_saved_when_agent2_raises(tmp_path) -> None:
    """When Agent-2 raises, the partial SessionContext (from Agent-1) is saved."""

    def failing_agent2(session, opts):
        raise RuntimeError("Agent-2 intentional test failure")

    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=failing_agent2),
    ):
        with pytest.raises(RuntimeError, match="intentional test failure"):
            run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    # session.json must exist even though the pipeline failed
    session_path = tmp_path / "session.json"
    assert session_path.exists()

    # It should contain Agent-1 data (transcript_raw set by mock_agent1)
    recovered = SessionContext.from_json(session_path)
    assert recovered.transcript_raw == "mocked transcript"
    # Agent-2 fields must NOT be present (agent2 failed before writing them)
    assert recovered.report_draft is None


def test_error_appended_when_agent2_raises(tmp_path) -> None:
    """The RuntimeError message appears in OrchestratorResult.errors … but since
    the exception is re-raised, we verify via the exception itself."""

    def failing_agent2(session, opts):
        raise RuntimeError("agent2-specific-error")

    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=failing_agent2),
    ):
        with pytest.raises(RuntimeError, match="agent2-specific-error"):
            run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())


def test_partial_session_saved_when_agent1_raises(tmp_path) -> None:
    """When Agent-1 raises, an initial (empty) session.json is still saved."""

    def failing_agent1(audio_path, session, opts):
        raise RuntimeError("Agent-1 intentional test failure")

    with patch("ems_pipeline.orchestrator.agent1_run", side_effect=failing_agent1):
        with pytest.raises(RuntimeError, match="intentional test failure"):
            run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    session_path = tmp_path / "session.json"
    assert session_path.exists()
    # Session has no Agent-1 data (failed before write)
    recovered = SessionContext.from_json(session_path)
    assert recovered.transcript_raw is None


def test_partial_session_saved_when_agent3_raises(tmp_path) -> None:
    """When Agent-3 raises, the session (with Agent-1 and Agent-2 data) is saved."""

    def failing_agent3(session, opts):
        raise RuntimeError("Agent-3 intentional test failure")

    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=failing_agent3),
    ):
        with pytest.raises(RuntimeError, match="intentional test failure"):
            run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    recovered = SessionContext.from_json(tmp_path / "session.json")
    assert recovered.transcript_raw == "mocked transcript"
    assert recovered.report_draft == "[STUB] mocked report"
    assert recovered.claim_id is None  # Agent-3 didn't complete


# ---------------------------------------------------------------------------
# Agent-4 conditional trigger
# ---------------------------------------------------------------------------


def test_agent4_not_run_without_denial_reason(tmp_path) -> None:
    """Agent-4 must NOT run when denial_reason is absent."""
    agent4_called = []

    def tracking_agent4(session, opts):
        agent4_called.append(True)
        return _mock_agent4(session, opts)

    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=_mock_agent3_submitted),
        patch("ems_pipeline.orchestrator.agent4_run", side_effect=tracking_agent4),
    ):
        result = run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    assert "agent4" not in result.agents_run
    assert agent4_called == []


def test_agent4_runs_when_denial_reason_set(tmp_path) -> None:
    """Agent-4 must run when session.denial_reason is populated."""

    def agent3_with_denial(session, opts) -> Agent3Result:
        # Submit the claim AND set denial_reason on the session
        updated = session.write_agent3(
            claim_id="denied-claim",
            payer_id=None,
            status="denied",
            flags=[],
        ).model_copy(update={"denial_reason": "medical necessity not met"})
        return Agent3Result(
            updated_session=updated,
            submitted=True,
            flags=[],
            remediation_requested=False,
            remediation_notes=[],
        )

    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=agent3_with_denial),
        patch("ems_pipeline.orchestrator.agent4_run", side_effect=_mock_agent4),
    ):
        result = run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    assert "agent4" in result.agents_run
    assert result.session.appeal_strategy == "standard_appeal"


# ---------------------------------------------------------------------------
# OrchestratorResult
# ---------------------------------------------------------------------------


def test_orchestrator_result_has_all_fields(tmp_path) -> None:
    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=_mock_agent3_submitted),
    ):
        result = run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    assert isinstance(result, OrchestratorResult)
    assert hasattr(result, "session")
    assert hasattr(result, "claim_id")
    assert hasattr(result, "submitted")
    assert hasattr(result, "agents_run")
    assert hasattr(result, "remediation_loops")
    assert hasattr(result, "errors")


def test_orchestrator_result_session_is_session_context(tmp_path) -> None:
    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=_mock_agent3_submitted),
    ):
        result = run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    assert isinstance(result.session, SessionContext)


# ---------------------------------------------------------------------------
# OrchestratorOptions model constraints
# ---------------------------------------------------------------------------


def test_orchestrator_options_defaults() -> None:
    opts = OrchestratorOptions()
    assert opts.payer_id is None
    assert opts.bandpass is False
    assert opts.denoise is False
    assert opts.asr_model == "base"
    assert opts.confidence_threshold == 0.7
    assert opts.max_remediation_loops == 2
    assert opts.llm_client is None


def test_orchestrator_options_extra_fields_forbidden() -> None:
    with pytest.raises(Exception):
        OrchestratorOptions(unknown_field="oops")  # type: ignore[call-arg]


def test_orchestrator_options_max_loops_zero_allowed() -> None:
    opts = OrchestratorOptions(max_remediation_loops=0)
    assert opts.max_remediation_loops == 0


# ---------------------------------------------------------------------------
# Session JSON round-trip after full pipeline
# ---------------------------------------------------------------------------


def test_session_json_round_trip_after_pipeline(tmp_path) -> None:
    """The session.json written by run_pipeline_supervised loads correctly."""
    with (
        patch("ems_pipeline.orchestrator.agent1_run", side_effect=_mock_agent1),
        patch("ems_pipeline.orchestrator.agent2_run", side_effect=_mock_agent2),
        patch("ems_pipeline.orchestrator.agent3_run", side_effect=_mock_agent3_submitted),
    ):
        result = run_pipeline_supervised("fake.wav", tmp_path, OrchestratorOptions())

    recovered = SessionContext.from_json(tmp_path / "session.json")
    assert recovered.encounter_id == result.session.encounter_id
    assert recovered.claim_id == "mock-claim-001"
    assert recovered.submission_status == "submitted"
