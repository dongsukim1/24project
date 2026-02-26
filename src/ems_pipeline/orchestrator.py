"""Orchestrator: thin routing layer for the 4-agent EMS pipeline.

The orchestrator does NOT perform clinical reasoning.  All domain intelligence
lives in the individual agents.  The orchestrator's job is sequencing, error
handling, and remediation-loop management.

Modes
-----
SUPERVISED MODE  — ``run_pipeline_supervised()``.  Runs the full agent chain
                   with a configurable remediation loop ceiling.
INTERACTIVE MODE — Served by the individual MCP tools (``ems.agent1_run``,
                   ``ems.agent2_run``, ``ems.agent3_run``, ``ems.agent4_run``).
                   The orchestrator does not implement interactive mode; each
                   MCP tool calls its agent directly.

Remediation loop
----------------
Agent-3 may return ``remediation_requested=True`` when pre-submission checks
surface error-severity flags.  The orchestrator catches this signal and loops
back to Agent-2, injecting the remediation notes into ``Agent2Options``.
The loop is capped at ``OrchestratorOptions.max_remediation_loops`` (default 2).
On the final allowed iteration, Agent-3 is told not to request further
remediation (``max_remediation_loops=0``) so it files the claim regardless.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ems_pipeline.agents.agent1 import Agent1Options
from ems_pipeline.agents.agent1 import run as agent1_run
from ems_pipeline.agents.agent2 import Agent2Options
from ems_pipeline.agents.agent2 import run as agent2_run
from ems_pipeline.agents.agent3 import Agent3Options
from ems_pipeline.agents.agent3 import run as agent3_run
from ems_pipeline.agents.agent4 import Agent4Options
from ems_pipeline.agents.agent4 import run as agent4_run
from ems_pipeline.context_utils import (
    compress_agent1_output,
    compress_agent2_output,
    compress_agent3_output,
)
from ems_pipeline.session import SessionContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------------


class OrchestratorOptions(BaseModel):
    """Tunable parameters for the supervised orchestration pipeline."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    payer_id: str | None = None
    bandpass: bool = False
    denoise: bool = False
    asr_model: str = "base"
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # Maximum number of Agent-2 re-runs triggered by Agent-3 remediation.
    max_remediation_loops: int = 2
    # STUB: inject real LLM client when live.
    llm_client: Any | None = None


class OrchestratorResult(BaseModel):
    """Return value of run_pipeline_supervised()."""

    model_config = ConfigDict(extra="forbid")

    session: SessionContext
    claim_id: str | None
    submitted: bool
    agents_run: list[str]
    remediation_loops: int
    errors: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_session_safe(session: SessionContext, out_dir: Path) -> None:
    """Write session.json, suppressing any save errors so originals propagate."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        session.to_json(out_dir / "session.json")
    except Exception as save_exc:  # noqa: BLE001
        logger.error("Failed to save partial session to %s: %s", out_dir, save_exc)


# ---------------------------------------------------------------------------
# Supervised pipeline
# ---------------------------------------------------------------------------


def run_pipeline_supervised(
    audio_path: str | Path,
    out_dir: str | Path,
    options: OrchestratorOptions,
) -> OrchestratorResult:
    """Run the full 4-agent pipeline under orchestrator supervision.

    Steps
    -----
    1. Create a fresh ``SessionContext``.
    2. Run Agent-1 (transcription + entity extraction).
    3. Check ``session.confidence_flags`` for error-severity entries.
       # STUB: all current flags are informational (no ``severity`` field).
       # Extend the flag schema and add blocking logic here when payer-specific
       # error conditions are defined.
    4. Remediation loop (≤ ``max_remediation_loops`` extra passes):
       a. Run Agent-2; inject ``additional_context`` from ``remediation_request``
          on re-runs.
       b. Run Agent-3.  Pass ``max_remediation_loops=0`` on the final allowed
          iteration to force filing.
       c. If Agent-3 sets ``remediation_requested`` and loops remain: loop back.
    5. If ``session.denial_reason`` is set, run Agent-4.
       # STUB: denial detection — ``denial_reason`` must currently be set
       # externally (e.g., by the operator or a payer webhook).
    6. Save the final ``SessionContext`` to ``out_dir/session.json``.
    7. Return ``OrchestratorResult``.

    Error handling
    --------------
    Any agent exception is caught, logged, appended to ``errors``, and the
    *partial* ``SessionContext`` is saved before the exception is re-raised.

    Args:
        audio_path: Path to the input audio file.
        out_dir:    Directory for all output artefacts (session.json, etc.).
        options:    Orchestration parameters.

    Returns:
        ``OrchestratorResult`` describing the completed (or partial) run.
    """
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agents_run: list[str] = []
    errors: list[str] = []
    submitted = False
    remediation_loops = 0

    # --- Step 1: Create session ---
    session = SessionContext.create()
    logger.info(
        "[orchestrator] Starting supervised pipeline — encounter=%s",
        session.encounter_id,
    )

    # --- Step 2: Agent-1 ---
    try:
        logger.info("[agent1] Transcription and extraction starting")
        agent1_opts = Agent1Options(
            bandpass=options.bandpass,
            denoise=options.denoise,
            asr_model=options.asr_model,
            confidence_threshold=options.confidence_threshold,
        )
        session = agent1_run(audio_path, session, agent1_opts)
        agents_run.append("agent1")
        logger.info(
            "[agent1] Done — segments=%d, entities=%d, flags=%d",
            len(session.transcript_segments or []),
            len(session.extracted_terms or []),
            len(session.confidence_flags or []),
        )
        logger.info(
            "[agent1] Stage boundary context for Agent-2: %s",
            compress_agent1_output(session),
        )
    except Exception as exc:
        msg = f"agent1 failed: {exc}"
        logger.exception(msg)
        errors.append(msg)
        _save_session_safe(session, out_dir)
        raise

    # --- Step 3: Check error-severity confidence flags ---
    # STUB: all current flags are informational (no severity field).
    # Insert blocking logic here when payer-specific error conditions are added.
    error_confidence_flags = [
        f for f in (session.confidence_flags or []) if f.get("severity") == "error"
    ]
    if error_confidence_flags:
        logger.warning(
            "[orchestrator] %d error-severity confidence flag(s) detected",
            len(error_confidence_flags),
        )

    # --- Step 4: Agent-2 + Agent-3 remediation loop ---
    agent3_result = None
    for loop in range(options.max_remediation_loops + 1):
        loops_remaining = options.max_remediation_loops - loop

        # -- Agent-2 --
        try:
            additional_context: list[str] | None = None
            if session.remediation_request and loop > 0:
                additional_context = session.remediation_request.get("notes") or None
                logger.info(
                    "[agent2] Remediation run %d — injecting %d note(s)",
                    loop,
                    len(additional_context or []),
                )
            else:
                logger.info("[agent2] Clinical documentation starting (run %d)", loop)

            agent2_opts = Agent2Options(
                llm_client=options.llm_client,
                payer_id=options.payer_id,
                additional_context=additional_context,
            )
            session = agent2_run(session, agent2_opts)
            agents_run.append("agent2")
            logger.info(
                "[agent2] Done — report_draft_len=%d, codes=%d",
                len(session.report_draft or ""),
                len(session.code_suggestions or []),
            )
            logger.info(
                "[agent2] Stage boundary context for Agent-3: %s",
                compress_agent2_output(session),
            )
        except Exception as exc:
            msg = f"agent2 failed (loop {loop}): {exc}"
            logger.exception(msg)
            errors.append(msg)
            _save_session_safe(session, out_dir)
            raise

        # -- Agent-3 --
        try:
            logger.info("[agent3] Claims filing starting (loop %d)", loop)
            # On the final iteration, tell Agent-3 not to request remediation.
            agent3_opts = Agent3Options(
                payer_id=options.payer_id,
                max_remediation_loops=1 if loops_remaining > 0 else 0,
            )
            agent3_result = agent3_run(session, agent3_opts)
            session = agent3_result.updated_session
            agents_run.append("agent3")
            logger.info(
                "[agent3] Done — submitted=%s, remediation_requested=%s",
                agent3_result.submitted,
                agent3_result.remediation_requested,
            )
            logger.info(
                "[agent3] Stage boundary context for Agent-4: %s",
                compress_agent3_output(session),
            )
        except Exception as exc:
            msg = f"agent3 failed (loop {loop}): {exc}"
            logger.exception(msg)
            errors.append(msg)
            _save_session_safe(session, out_dir)
            raise

        if agent3_result.submitted:
            submitted = True
            break

        if agent3_result.remediation_requested and loops_remaining > 0:
            remediation_loops += 1
            logger.info(
                "[orchestrator] Remediation loop %d/%d — routing back to Agent-2",
                remediation_loops,
                options.max_remediation_loops,
            )
            continue

        # No more loops allowed or remediation not requested.
        break

    # --- Step 5: Agent-4 (conditional on denial) ---
    # STUB: denial_reason must be set externally (e.g., by the operator or a
    # payer EOB webhook) before the orchestrator reaches this check.
    if session.denial_reason:
        try:
            logger.info(
                "[agent4] Denial detected ('%s') — appeal starting",
                session.denial_reason,
            )
            agent4_opts = Agent4Options(payer_id=options.payer_id)
            agent4_result = agent4_run(session, agent4_opts)
            session = agent4_result.updated_session
            agents_run.append("agent4")
            logger.info(
                "[agent4] Done — strategy=%s", agent4_result.strategy_chosen
            )
        except Exception as exc:
            msg = f"agent4 failed: {exc}"
            logger.exception(msg)
            errors.append(msg)
            _save_session_safe(session, out_dir)
            raise

    # --- Step 6: Save final session ---
    _save_session_safe(session, out_dir)
    logger.info(
        "[orchestrator] Pipeline complete — encounter=%s, submitted=%s, "
        "loops=%d, agents=%s, errors=%d",
        session.encounter_id,
        submitted,
        remediation_loops,
        agents_run,
        len(errors),
    )

    return OrchestratorResult(
        session=session,
        claim_id=session.claim_id,
        submitted=submitted,
        agents_run=agents_run,
        remediation_loops=remediation_loops,
        errors=errors,
    )
