"""Agent 4: Appeals Agent.

Triggered conditionally when a claim is denied.  Retrieves denial context
and precedent outcomes, selects an appeal strategy, generates a structured
appeal script, and optionally initiates a voice session for phone-based
appeals.

Voice and LLM integrations are fully stubbed; the stubs are annotated
with the replacement points so they can be wired in incrementally.

Improvement loop
----------------
``record_appeal_outcome()`` is the *write side* of a retrieval improvement
loop: outcome records feed back into the Index 4 (Appeal Precedent Store)
so that ``retrieve_appeal_precedents()`` can return evidence-backed strategy
choices for future denials.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from ems_pipeline.rag import (
    record_appeal_outcome as rag_record_appeal_outcome,
)
from ems_pipeline.rag import (
    retrieve_appeal_precedents as rag_retrieve_appeal_precedents,
)
from ems_pipeline.session import SessionContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------------


class Agent4Options(BaseModel):
    """Tunable parameters for Agent-4: Appeals."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    payer_id: str | None = None
    # STUB: inject real LLM client (e.g., anthropic.Anthropic()) for script generation.
    llm_client: Any | None = None
    # STUB: inject real voice client (e.g., ElevenLabs / Twilio / Bland.ai) when live.
    voice_client: Any | None = None


class Agent4Result(BaseModel):
    """Return value of Agent-4 run()."""

    model_config = ConfigDict(extra="forbid")

    updated_session: SessionContext
    strategy_chosen: str
    appeal_script: str
    voice_session_id: str | None


# ---------------------------------------------------------------------------
# Retrieval pre-steps (stubs — wired for Index 2 / Index 4)
# ---------------------------------------------------------------------------


def retrieve_denial_context(
    denial_reason: str,
    payer_id: str | None,
) -> dict[str, Any]:
    """Return payer-specific denial context and common appeal remedies.

    # STUB: returns a synthetic context dict — replace with a combined call to:
    #   Index 2 (Payer Rules RAG)     — payer-specific denial policies
    #   Index 4 (Appeal Precedents)   — deadline and common remedy metadata

    Args:
        denial_reason: Verbatim denial reason string from the payer EOB.
        payer_id:      Payer identifier, or None for self-pay / unknown.

    Returns:
        Context dict with denial_reason, payer_id, common_remedies, and
        appeal_deadline_days.
    """
    _ = denial_reason, payer_id  # consumed by real retrieval call
    return {
        "denial_reason": denial_reason,
        "payer_id": payer_id,
        "common_remedies": [],
        "appeal_deadline_days": 30,
    }


def retrieve_appeal_precedents(
    denial_reason: str,
    payer_id: str | None,
    cpt_codes: list[str],
) -> list[dict[str, Any]]:
    """Return historical appeal outcomes similar to the current denial.

    Precedents are read from shared denial records in Index 2 via
    ``ems_pipeline.rag.retrieve_appeal_precedents()``.

    Args:
        denial_reason: Verbatim denial reason to match against precedents.
        payer_id:      Payer filter for precedent retrieval.
        cpt_codes:     CPT codes from the claim (additional filter).

    Returns:
        List of precedent dicts, most-recent first.
    """
    return rag_retrieve_appeal_precedents(
        denial_reason=denial_reason,
        payer_id=payer_id,
        cpt_codes=cpt_codes,
    )


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


def choose_appeal_strategy(
    denial_ctx: dict[str, Any],
    precedents: list[dict[str, Any]],
) -> str:
    """Select the best appeal strategy for the current denial.

    Selection logic (in priority order):

    1. **Precedent-driven** — if ``precedents`` is non-empty, use the
       strategy from the most-recent *successful* precedent.
    2. **Rule-based fallback** — keyword match on ``denial_ctx["denial_reason"]``:
       - "medical necessity"  → ``"clinical_documentation_appeal"``
       - "coverage"           → ``"policy_exception_appeal"``
       - "authorization"      → ``"retroactive_auth_appeal"``
       - (default)            → ``"standard_appeal"``

    Args:
        denial_ctx:  Context dict from retrieve_denial_context().
        precedents:  Precedent list from retrieve_appeal_precedents(),
                     ordered most-recent first.

    Returns:
        Strategy identifier string.
    """
    # --- Precedent-driven selection ---
    for precedent in precedents:
        if precedent.get("outcome") == "success":
            strategy = precedent.get("strategy") or ""
            if strategy:
                return strategy

    # --- Rule-based keyword fallback ---
    denial_reason_lower = (denial_ctx.get("denial_reason") or "").lower()

    if "medical necessity" in denial_reason_lower:
        return "clinical_documentation_appeal"
    if "coverage" in denial_reason_lower:
        return "policy_exception_appeal"
    if "authorization" in denial_reason_lower:
        return "retroactive_auth_appeal"

    return "standard_appeal"


# ---------------------------------------------------------------------------
# Appeal script generation (stub)
# ---------------------------------------------------------------------------


def build_appeal_script(
    session: SessionContext,
    strategy: str,
    denial_ctx: dict[str, Any],
    precedents: list[dict[str, Any]],
) -> str:
    """Generate a structured appeal script from session data.

    # STUB: uses a deterministic template — replace with LLM generation when
    # ``llm_client`` is available.  The LLM call would be::
    #
    #   response = llm_client.messages.create(
    #       model="claude-sonnet-4-6",
    #       max_tokens=1024,
    #       messages=[{"role": "user", "content": appeal_prompt}],
    #   )
    #   return response.content[0].text

    Script structure
    ----------------
    1. Opening   — claim reference, payer, strategy
    2. Denial summary — denial reason, appeal deadline
    3. Clinical justification — report excerpt + submitted codes
    4. Closing   — call-to-action, signature, stub marker

    Args:
        session:    Current SessionContext (populated by Agents 1–3).
        strategy:   Strategy identifier from choose_appeal_strategy().
        denial_ctx: Context dict from retrieve_denial_context().
        precedents: Precedents list (unused by stub; available to LLM prompt).

    Returns:
        Multi-line appeal script string.
    """
    _ = precedents  # available to future LLM prompt assembly

    claim_id = session.claim_id or "[unknown claim]"
    payer_ref = session.payer_id or denial_ctx.get("payer_id") or "[unknown payer]"
    denial_reason = (
        session.denial_reason
        or denial_ctx.get("denial_reason")
        or "[unspecified denial reason]"
    )
    deadline_days = denial_ctx.get("appeal_deadline_days", 30)

    # --- Clinical justification block ---
    report_excerpt = ""
    if session.report_draft:
        report_excerpt = session.report_draft[:200]
        if len(session.report_draft) > 200:
            report_excerpt += "..."

    codes_str = ""
    if session.code_suggestions:
        codes_str = ", ".join(
            f"{c.get('code', 'PENDING')} ({c.get('type', '')})"
            for c in session.code_suggestions
            if isinstance(c, dict)
        )

    # --- Assemble sections ---
    opening = (
        f"RE: Appeal for Claim {claim_id}\n"
        f"Payer: {payer_ref}\n"
        f"Appeal Strategy: {strategy}\n\n"
        f"To Whom It May Concern,\n\n"
        f"We are writing to formally appeal the denial of Claim {claim_id}."
    )

    denial_summary = (
        f"\n\n## DENIAL SUMMARY\n"
        f"Denial Reason: {denial_reason}\n"
        f"Appeal Deadline: {deadline_days} days from denial date."
    )

    clinical_justification = "\n\n## CLINICAL JUSTIFICATION\n"
    if report_excerpt:
        clinical_justification += f"Clinical Documentation Excerpt:\n{report_excerpt}\n"
    if codes_str:
        clinical_justification += f"Submitted Codes: {codes_str}\n"
    clinical_justification += (
        "The services rendered were medically necessary and consistent with "
        "established EMS clinical protocols and applicable payer guidelines."
    )

    closing = (
        f"\n\n## CLOSING\n"
        f"We respectfully request a full review of this claim under appeal "
        f"strategy '{strategy}'. Please contact us with any additional "
        f"documentation requirements.\n\n"
        f"Sincerely,\nEMS Billing Compliance Team\n\n"
        f"[STUB: Replace this template with LLM-generated appeal letter "
        f"when llm_client is available.]"
    )

    return opening + denial_summary + clinical_justification + closing


# ---------------------------------------------------------------------------
# Voice session (stub)
# ---------------------------------------------------------------------------


def start_voice_session(
    appeal_script: str,
    voice_client: Any,
) -> str | None:
    """Initiate a voice-based appeal session using the appeal script.

    # STUB: returns None when voice_client is None.
    # Replace with real TTS/telephony call when the integration layer is live,
    # e.g. ElevenLabs, Twilio, or Bland.ai::
    #
    #   session_id = voice_client.start_call(
    #       script=appeal_script,
    #       recipient=payer_phone_number,
    #   )
    #   return session_id

    Args:
        appeal_script: The generated appeal script text.
        voice_client:  Voice/telephony client instance, or None.

    Returns:
        Voice session identifier string, or None if no voice client.
    """
    _ = appeal_script  # consumed by real voice call

    if voice_client is None:
        return None

    # STUB: real voice call goes here.
    raise NotImplementedError(
        "Voice integration not yet implemented. "
        "Replace start_voice_session() with a real TTS/telephony call "
        "(e.g., ElevenLabs, Twilio, or Bland.ai)."
    )


# ---------------------------------------------------------------------------
# Outcome recording (write side of the improvement loop)
# ---------------------------------------------------------------------------


def record_appeal_outcome(
    session: SessionContext,
    strategy: str,
    outcome: str | None,
    notes: str | None,
) -> dict[str, Any]:
    """Build and persist an appeal outcome record for future retrieval.

    This is the *write side* of the Index 4 improvement loop.  Outcome records
    produced here are written to shared denial records so
    that ``retrieve_appeal_precedents()`` can surface evidence-backed strategies
    for future, similar denials.

    Args:
        session:  Current SessionContext (must have Agent-3 / Agent-4 fields).
        strategy: Strategy identifier chosen for this appeal.
        outcome:  ``"success"``, ``"failure"``, or ``None`` (pending).
        notes:    Free-text notes about the appeal outcome.

    Returns:
        Appeal record dict suitable for insertion into Index 4.
    """
    codes = [
        c.get("code", "PENDING")
        for c in (session.code_suggestions or [])
        if isinstance(c, dict)
    ]
    denial_id = session.encounter_id
    record: dict[str, Any] = {
        "denial_id": denial_id,
        "encounter_id": session.encounter_id,
        "strategy": strategy,
        "outcome": outcome,
        "denial_reason": session.denial_reason,
        "payer_id": session.payer_id,
        "claim_id": session.claim_id,
        "codes": codes,
        "notes": notes,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        rag_record_appeal_outcome(
            denial_id=denial_id,
            strategy=strategy,
            outcome=outcome or "pending",
            notes=notes,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist appeal outcome for denial_id=%s", denial_id)

    logger.debug("appeal_outcome_record: %s", record)
    return record


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    session: SessionContext,
    options: Agent4Options,
) -> Agent4Result:
    """Run Agent-4: retrieve denial context, choose strategy, generate appeal.

    Steps
    -----
    1. Retrieve denial context from payer/precedent stubs.
    2. Retrieve historical appeal precedents (stub → []).
    3. Choose appeal strategy (precedent-first, then rule-based).
    4. Build appeal script (template stub; LLM-ready).
    5. Start voice session (stub → None).
    6. Record appeal outcome for future precedent retrieval.
    7. Write Agent-4 fields (denial_reason, appeal_strategy, appeal_history)
       into a new SessionContext.

    Args:
        session: Current SessionContext (populated by Agents 1–3, with
                 session.denial_reason set by the orchestrator after denial).
        options: Agent-4 tunable parameters.

    Returns:
        Agent4Result with updated_session, strategy_chosen, appeal_script,
        and voice_session_id.
    """
    effective_payer_id = options.payer_id or session.payer_id

    # --- Step 1: Retrieve denial context ---
    denial_reason = session.denial_reason or ""
    denial_ctx = retrieve_denial_context(denial_reason, effective_payer_id)

    # --- Step 2: Retrieve appeal precedents ---
    cpt_codes = [
        c.get("code", "")
        for c in (session.code_suggestions or [])
        if isinstance(c, dict) and c.get("type") == "CPT"
    ]
    precedents = retrieve_appeal_precedents(denial_reason, effective_payer_id, cpt_codes)

    # --- Step 3: Choose strategy ---
    strategy = choose_appeal_strategy(denial_ctx, precedents)

    # --- Step 4: Build appeal script ---
    appeal_script = build_appeal_script(session, strategy, denial_ctx, precedents)

    # --- Step 5: Start voice session (stub) ---
    voice_session_id = start_voice_session(appeal_script, options.voice_client)

    # --- Step 6: Record appeal outcome (stub) ---
    appeal_record = record_appeal_outcome(
        session, strategy, outcome=None, notes=None
    )

    # --- Step 7: Write Agent-4 fields into session ---
    updated_session = session.write_agent4(
        denial_reason=denial_reason,
        strategy=strategy,
    )
    existing_history = list(session.appeal_history or [])
    existing_history.append(appeal_record)
    updated_session = updated_session.model_copy(
        update={"appeal_history": existing_history}
    )

    return Agent4Result(
        updated_session=updated_session,
        strategy_chosen=strategy,
        appeal_script=appeal_script,
        voice_session_id=voice_session_id,
    )
