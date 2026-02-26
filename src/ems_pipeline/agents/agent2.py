"""Agent 2: Clinical Documentation Agent.

Co-generates the narrative report and code suggestions using retrieval hooks
(stubbed for future population) and the existing deterministic claim builder.

Retrieval index assignments (all stubs until infrastructure is live):
  Index 1 — Patient History    → rag.retrieve_patient_history()
  Index 2 — Payer Rules        → rag.retrieve_payer_requirements()
  Index 3 — Coding Guidelines  → rag.retrieve_coding_guidelines()
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ems_pipeline.claim.builder import build_claim as _build_claim_from_transcript
from ems_pipeline.context_utils import tag_citations
from ems_pipeline.models import Transcript
from ems_pipeline.rag import (
    retrieve_coding_guidelines,
    retrieve_patient_history,
    retrieve_payer_requirements,
)
from ems_pipeline.session import SessionContext


class Agent2Options(BaseModel):
    """Tunable parameters for Agent-2: Clinical Documentation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # STUB: inject real LLM client (e.g., anthropic.Anthropic()) when live.
    llm_client: Any | None = None
    payer_id: str | None = None
    # Remediation notes injected by the orchestrator on re-runs after Agent-3
    # validation failures.  Passed through to the LLM prompt when live.
    additional_context: list[str] | None = None


# ---------------------------------------------------------------------------
# Retrieval pre-step (called before report generation)
# ---------------------------------------------------------------------------


def _retrieve_context(
    session: SessionContext,
    payer_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fetch payer rules, coding guidelines, and patient history.

    All three calls are stubs that return empty dicts until the RAG
    infrastructure is wired in.

    Returns:
        (payer_ctx, coding_ctx, patient_ctx)
    """
    payer_ctx = retrieve_payer_requirements(payer_id)

    entity_types = list({ent.type for ent in (session.extracted_terms or [])})
    coding_ctx = retrieve_coding_guidelines(entity_types)

    patient_ctx = retrieve_patient_history(session.encounter_id)

    return payer_ctx, coding_ctx, patient_ctx


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def build_report_prompt(
    session: SessionContext,
    payer_ctx: dict[str, Any],
    coding_ctx: dict[str, Any],
    patient_ctx: dict[str, Any],
) -> str:
    """Assemble the LLM prompt for report generation and code suggestion.

    The prompt is fully deterministic and testable without an LLM.

    Structure
    ---------
    1. System instruction
    2. Payer constraints       (only when payer_ctx is non-empty)
    3. Coding criteria         (only when coding_ctx is non-empty)
    4. Patient history         (only when patient_ctx is non-empty)
    5. Extracted clinical terms from session
    6. Ambiguities flagged by Agent-1 (if any)
    7. Pre-populated ICD / CPT stubs (if any)
    8. Transcript segments (or raw transcript)
    9. Generation instruction

    Args:
        session:     Current SessionContext (populated by Agent-1).
        payer_ctx:   Payer requirements dict from RAG (may be empty).
        coding_ctx:  Coding guidelines dict from RAG (may be empty).
        patient_ctx: Patient history dict from RAG (may be empty).

    Returns:
        Assembled prompt string.
    """
    sections: list[str] = []

    # 1. System instruction
    sections.append(
        "## SYSTEM\n"
        "You are a clinical documentation specialist supporting EMS billing compliance.\n"
        "Generate a concise pre-hospital narrative report and medical code suggestions "
        "grounded strictly in the evidence below. Do not invent clinical details not "
        "present in the transcript or extracted terms."
    )

    # 2. Payer constraints (only when populated by RAG)
    if payer_ctx:
        lines = "\n".join(f"  - {k}: {v}" for k, v in payer_ctx.items())
        sections.append(f"## PAYER REQUIREMENTS\n{lines}")

    # 3. Coding criteria (only when populated by RAG)
    if coding_ctx:
        lines = "\n".join(f"  - {k}: {v}" for k, v in coding_ctx.items())
        sections.append(f"## CODING GUIDELINES\n{lines}")

    # 4. Patient history (only when populated by RAG)
    if patient_ctx:
        lines = "\n".join(f"  - {k}: {v}" for k, v in patient_ctx.items())
        sections.append(f"## PATIENT HISTORY\n{lines}")

    # 5. Extracted clinical terms
    term_lines: list[str] = []
    for ent in session.extracted_terms or []:
        parts: list[str] = [f"[{ent.type}]", ent.text]
        if ent.normalized and ent.normalized != ent.text:
            parts.append(f"(normalized: {ent.normalized})")
        if ent.confidence is not None:
            parts.append(f"confidence={ent.confidence:.2f}")
        term_lines.append("  - " + " ".join(parts))

    sections.append(
        "## EXTRACTED CLINICAL TERMS\n"
        + ("\n".join(term_lines) if term_lines else "  (none)")
    )

    # 6. Ambiguities flagged by Agent-1
    if session.ambiguities:
        amb_lines = [
            "  - [{reason}] {text} (segment: {seg})".format(
                reason=a.get("reason", ""),
                text=a.get("text", ""),
                seg=a.get("segment_id", "unknown"),
            )
            for a in session.ambiguities
        ]
        sections.append("## AMBIGUITIES TO RESOLVE\n" + "\n".join(amb_lines))

    # 7. Pre-populated code stubs from session (if any)
    if session.icd_codes:
        icd_lines = [
            f"  - {c.get('code', '')} — {c.get('description', '')}"
            for c in session.icd_codes
        ]
        sections.append(
            "## SUGGESTED ICD CODES (pre-processing)\n" + "\n".join(icd_lines)
        )

    if session.cpt_codes:
        cpt_lines = [
            f"  - {c.get('code', '')} — {c.get('description', '')}"
            for c in session.cpt_codes
        ]
        sections.append(
            "## SUGGESTED CPT CODES (pre-processing)\n" + "\n".join(cpt_lines)
        )

    # 8. Transcript segments (prefer structured; fall back to raw text)
    if session.transcript_segments:
        seg_lines = [
            "  [{speaker} {start:.1f}s\u2013{end:.1f}s conf={conf:.2f}]: {text}".format(
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
                conf=seg.confidence,
                text=seg.text,
            )
            for seg in session.transcript_segments
        ]
        sections.append("## TRANSCRIPT SEGMENTS\n" + "\n".join(seg_lines))
    elif session.transcript_raw:
        sections.append(f"## TRANSCRIPT\n{session.transcript_raw}")

    # 9. Generation instruction
    sections.append(
        "## INSTRUCTION\n"
        "1. Write a concise pre-hospital narrative report (3\u20136 sentences).\n"
        "2. Provide medical code suggestions as a JSON array:\n"
        '   [{"code": "...", "type": "ICD"|"CPT", "rationale": "...", '
        '"evidence_segment_ids": [...]}]\n'
        "3. State your clinical reasoning in 1\u20133 sentences.\n"
        "Return ONLY valid JSON with this structure:\n"
        '{"report": "...", "reasoning": "...", "codes": [...]}'
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# LLM call (stub)
# ---------------------------------------------------------------------------


def call_llm(
    prompt: str,
    llm_client: Any,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Call the LLM to generate a clinical report and code suggestions.

    # STUB: when llm_client is None, returns placeholder strings and an empty
    # code_suggestions list.  Replace the stub body with a real Anthropic SDK
    # call (claude-sonnet-4-6) when live:
    #
    #   response = llm_client.messages.create(
    #       model="claude-sonnet-4-6",
    #       max_tokens=1024,
    #       messages=[{"role": "user", "content": prompt}],
    #   )
    #   raw = response.content[0].text
    #   parsed = json.loads(raw)
    #   return parsed["report"], parsed["reasoning"], parsed["codes"]

    Args:
        prompt:     Assembled prompt from build_report_prompt().
        llm_client: LLM client instance. None triggers the stub path.

    Returns:
        (report_draft, clinical_reasoning, code_suggestions)
        code_suggestions shape: {code, type: "ICD"|"CPT", rationale, evidence_segment_ids}
    """
    _ = prompt  # consumed by real LLM call; referenced here to satisfy linters

    if llm_client is None:
        # STUB: returns placeholder output — replace with real LLM call above.
        return (
            "[STUB] Clinical report pending LLM integration.",
            "[STUB] Clinical reasoning pending LLM integration.",
            [],
        )

    # STUB: LLM client provided but call not yet implemented.
    # Replace this entire block with the real Anthropic SDK call shown above.
    raise NotImplementedError(
        "Real LLM call not yet implemented. "
        "Replace call_llm() with an Anthropic SDK call to claude-sonnet-4-6."
    )


# ---------------------------------------------------------------------------
# Citation chain tagging
# ---------------------------------------------------------------------------


def tag_citation_map(
    report_draft: str,
    entities: list[Any],
) -> dict[str, list[str]]:
    """Build a citation map: entity normalized values → source segment IDs.

    Scans report_draft (case-insensitively) for each entity's normalized value.
    For matched entities, records the segment_id stored in entity.attributes.

    Args:
        report_draft: Generated report text.
        entities:     List of Entity objects from session.extracted_terms.

    Returns:
        Mapping {entity.normalized: [segment_id, ...]} for entities whose
        normalized value appears in the report.  Entities with no normalized
        value or not found in the report are omitted.
    """
    return tag_citations(report_draft, entities, segments=[])


# ---------------------------------------------------------------------------
# Deterministic claim enrichment helper
# ---------------------------------------------------------------------------


def _derive_base_codes_from_claim(
    claim_fields: dict[str, Any],
) -> list[dict[str, Any]]:
    """Derive placeholder code suggestions from deterministic claim fields.

    Provides a grounded base for Agent-3 and the LLM to build on.  LLM
    code_suggestions overlay and replace these entries when the LLM is live.
    """
    codes: list[dict[str, Any]] = []

    # Primary impression → ICD placeholder
    primary = claim_fields.get("primary_impression", {})
    if isinstance(primary, dict):
        value = primary.get("value") or ""
        evidence = primary.get("evidence_segment_ids") or []
        if value:
            codes.append(
                {
                    "code": "PENDING",
                    "type": "ICD",
                    "rationale": f"Primary impression: {value}",
                    "evidence_segment_ids": evidence,
                }
            )

    # Interventions → CPT placeholders
    for intervention in claim_fields.get("interventions", []):
        if not isinstance(intervention, dict):
            continue
        normalized = intervention.get("normalized") or intervention.get("text") or ""
        evidence = intervention.get("evidence_segment_ids") or []
        if normalized:
            codes.append(
                {
                    "code": "PENDING",
                    "type": "CPT",
                    "rationale": f"Intervention: {normalized}",
                    "evidence_segment_ids": evidence,
                }
            )

    return codes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    session: SessionContext,
    options: Agent2Options,
) -> SessionContext:
    """Run Agent-2: generate a clinical documentation report and code suggestions.

    Steps
    -----
    1. Retrieve payer requirements, coding guidelines, patient history (stubs).
    2. Assemble the LLM prompt.
    3. Call the LLM stub → report_draft, clinical_reasoning, code_suggestions.
    4. Run the deterministic claim builder as a grounded evidence base.
    5. Merge claim fields with LLM code_suggestions (LLM overlays claim base).
    6. Tag citation_map: entity normalized values → source segment IDs.
    7. Write all Agent-2 fields into session.

    Args:
        session: Current SessionContext (populated by Agent-1).
        options: Agent-2 options (llm_client, payer_id).

    Returns:
        Updated SessionContext with all Agent-2 fields populated.
    """
    # --- Step 1: Retrieval pre-step (all stubs) ---
    payer_ctx, coding_ctx, patient_ctx = _retrieve_context(session, options.payer_id)

    # --- Step 2: Build prompt ---
    prompt = build_report_prompt(session, payer_ctx, coding_ctx, patient_ctx)

    # --- Step 3: Call LLM (stub when llm_client is None) ---
    report_draft, clinical_reasoning, llm_codes = call_llm(prompt, options.llm_client)

    # --- Step 4: Deterministic claim builder as grounded evidence base ---
    claim_fields: dict[str, Any] = {}
    if session.transcript_segments and session.extracted_terms:
        transcript = Transcript(
            segments=session.transcript_segments,
            metadata={"audio_filename": session.encounter_id},
        )
        _, claim_json = _build_claim_from_transcript(
            transcript,
            list(session.extracted_terms),
            events=[],
        )
        claim_fields = claim_json.get("fields", {})

    # --- Step 5: Overlay LLM suggestions on deterministic base ---
    # LLM suggestions take precedence; base codes used only when LLM returns nothing.
    base_codes = _derive_base_codes_from_claim(claim_fields)
    code_suggestions = llm_codes if llm_codes else base_codes

    # --- Step 6: Citation chain tagging ---
    citation_map = tag_citations(
        report_draft,
        session.extracted_terms or [],
        session.transcript_segments or [],
    ) or None

    # --- Step 7: Write Agent-2 fields into session ---
    return session.write_agent2(
        report=report_draft,
        reasoning=clinical_reasoning,
        codes=code_suggestions,
        citation_map=citation_map,
    )
