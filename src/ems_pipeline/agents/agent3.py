"""Agent 3: Claims Filing Agent.

Validates the assembled session, runs an optional remediation loop when
pre-submission checks fail, then submits the claim.

Submission is currently a stub.  Replace ``file_claim()`` with a real
payer-API call when the integration layer is available.

Remediation flow
----------------
If ``validate_claim()`` surfaces error-severity flags *and*
``max_remediation_loops > 0``, the agent populates
``session.remediation_request`` and returns without filing.  The
orchestrator is responsible for routing the session back to Agent-2 with
the remediation notes attached.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ems_pipeline.models import Claim
from ems_pipeline.rag import retrieve_payer_requirements
from ems_pipeline.session import SessionContext


# ---------------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------------


class Agent3Options(BaseModel):
    """Tunable parameters for Agent-3: Claims Filing."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    payer_id: str | None = None
    # STUB: inject real LLM client for AI-assisted remediation notes when live.
    llm_client: Any | None = None
    max_remediation_loops: int = 1


class Agent3Result(BaseModel):
    """Return value of Agent-3 run()."""

    model_config = ConfigDict(extra="forbid")

    updated_session: SessionContext
    submitted: bool
    flags: list[dict[str, Any]]
    remediation_requested: bool
    remediation_notes: list[str]


# ---------------------------------------------------------------------------
# Pre-submission validation
# ---------------------------------------------------------------------------


def validate_claim(
    session: SessionContext,
    payer_ctx: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run pre-submission checks and return a list of flags.

    Each flag has the shape::

        {
            "field":             str,
            "issue":             str,
            "severity":          "error" | "warning",
            "remediation_hint":  str,
        }

    Error-severity flags block submission; warnings are advisory only.

    # STUB: payer-specific validation rules — replace with real payer rule
    # evaluation when Index 2 (Payer Rules RAG) is live.
    """
    flags: list[dict[str, Any]] = []

    # --- Check 1: report_draft must be present and non-empty ---
    if not session.report_draft:
        flags.append(
            {
                "field": "report_draft",
                "issue": "Report draft is missing or empty",
                "severity": "error",
                "remediation_hint": (
                    "Run Agent-2 to generate a clinical report before filing"
                ),
            }
        )

    # --- Check 2: code_suggestions must be present and non-empty ---
    if not session.code_suggestions:
        flags.append(
            {
                "field": "code_suggestions",
                "issue": "No ICD/CPT code suggestions are present",
                "severity": "error",
                "remediation_hint": (
                    "Run Agent-2 to generate code suggestions before filing"
                ),
            }
        )

    # --- Check 3: code provenance quality (advisory) ---
    if session.code_suggestions:
        missing_evidence = [
            c
            for c in session.code_suggestions
            if isinstance(c, dict) and not c.get("evidence_segment_ids")
        ]
        if missing_evidence:
            flags.append(
                {
                    "field": "code_suggestions",
                    "issue": (
                        f"{len(missing_evidence)} code suggestion(s) lack "
                        "evidence segment links"
                    ),
                    "severity": "warning",
                    "remediation_hint": (
                        "Review code suggestions and ensure each is grounded "
                        "in a transcript segment"
                    ),
                }
            )

    # --- STUB: payer-specific rules (placeholder) ---
    # Replace the block below with real payer rule evaluation once Index 2 is live.
    _ = payer_ctx  # consumed by real rule evaluation

    return flags


# ---------------------------------------------------------------------------
# Claim filing (stub)
# ---------------------------------------------------------------------------


def file_claim(
    session: SessionContext,
    payer_id: str | None,
) -> dict[str, Any]:
    """Simulate claim submission to the payer system.

    # STUB: returns a synthetic confirmation dict — replace with a real payer
    # API call (e.g. HTTP POST to clearinghouse endpoint) when the integration
    # layer is available.

    Args:
        session:  Current SessionContext (populated by Agents 1 & 2).
        payer_id: Target payer identifier, or None for self-pay / unknown.

    Returns:
        Confirmation dict: {claim_id, status, timestamp}.
    """
    claim_id = session.claim_id or str(uuid.uuid4())
    return {
        "claim_id": claim_id,
        "status": "submitted",
        "payer_id": payer_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Rehydration helper (old-pipeline bridge)
# ---------------------------------------------------------------------------


def rehydrate_from_existing_claim(claim_json_path: str | Path) -> SessionContext:
    """Load a Claim JSON produced by claim/builder.py and build a SessionContext.

    This bridge enables Agent-3 to process encounters that went through the
    old ``ems_pipeline build-claim`` path rather than the full 4-agent pipeline.

    The claim fields are used to reconstruct placeholder code_suggestions so
    that validate_claim() and file_claim() can operate normally.

    Args:
        claim_json_path: Path to a Claim JSON file (schema_version 0.1).

    Returns:
        A new SessionContext pre-populated with claim data.
    """
    path = Path(claim_json_path)
    claim = Claim.model_validate_json(path.read_text(encoding="utf-8"))
    fields = claim.fields

    # Reconstruct placeholder code_suggestions from claim fields.
    code_suggestions: list[dict[str, Any]] = []

    primary = fields.get("primary_impression", {})
    if isinstance(primary, dict) and primary.get("value"):
        code_suggestions.append(
            {
                "code": "PENDING",
                "type": "ICD",
                "rationale": f"Primary impression: {primary['value']}",
                "evidence_segment_ids": primary.get("evidence_segment_ids", []),
            }
        )

    for intervention in fields.get("interventions", []):
        if not isinstance(intervention, dict):
            continue
        label = intervention.get("normalized") or intervention.get("text") or ""
        if label:
            code_suggestions.append(
                {
                    "code": "PENDING",
                    "type": "CPT",
                    "rationale": f"Intervention: {label}",
                    "evidence_segment_ids": intervention.get(
                        "evidence_segment_ids", []
                    ),
                }
            )

    return SessionContext.create().model_copy(
        update={
            "claim_id": claim.claim_id,
            "code_suggestions": code_suggestions or None,
            "submission_status": "pending",
        }
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    session: SessionContext,
    options: Agent3Options,
) -> Agent3Result:
    """Run Agent-3: validate, optionally remediate, then file the claim.

    Steps
    -----
    1. Retrieve payer requirements (stub).
    2. Run pre-submission validation → list of flags.
    3. If error-severity flags exist *and* max_remediation_loops > 0:
       a. Build human-readable remediation_notes.
       b. Stamp session.remediation_request.
       c. Return without filing (orchestrator routes back to Agent-2).
    4. Otherwise: call file_claim() → confirmation dict.
    5. Write Agent-3 fields into session via write_agent3().
    6. Return Agent3Result(submitted=True).

    Args:
        session: Current SessionContext (must have Agent-2 fields populated
                 for a clean validation pass).
        options: Agent-3 tunable parameters.

    Returns:
        Agent3Result with updated_session, submitted flag, flags, and
        remediation metadata.
    """
    # --- Step 1: Retrieve payer requirements (stub) ---
    payer_ctx = retrieve_payer_requirements(options.payer_id)

    # --- Step 2: Pre-submission validation ---
    flags = validate_claim(session, payer_ctx)
    error_flags = [f for f in flags if f["severity"] == "error"]

    # --- Step 3: Remediation loop ---
    if error_flags and options.max_remediation_loops > 0:
        remediation_notes = [
            (
                f"[{f['severity'].upper()}] {f['field']}: {f['issue']} "
                f"— Hint: {f['remediation_hint']}"
            )
            for f in error_flags
        ]
        updated_session = session.model_copy(
            update={
                "remediation_request": {
                    "notes": remediation_notes,
                    "payer_id": options.payer_id,
                    "loop_count": 1,
                }
            }
        )
        return Agent3Result(
            updated_session=updated_session,
            submitted=False,
            flags=flags,
            remediation_requested=True,
            remediation_notes=remediation_notes,
        )

    # --- Step 4: File the claim ---
    filing_result = file_claim(session, options.payer_id)

    # --- Step 5: Write Agent-3 fields into session ---
    updated_session = session.write_agent3(
        claim_id=filing_result["claim_id"],
        payer_id=options.payer_id,
        status=filing_result["status"],
        flags=flags,
    )

    # --- Step 6: Return result ---
    return Agent3Result(
        updated_session=updated_session,
        submitted=True,
        flags=flags,
        remediation_requested=False,
        remediation_notes=[],
    )
