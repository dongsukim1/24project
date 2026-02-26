"""SessionContext: accumulates state across the 4-agent EMS pipeline.

A single SessionContext instance is created at the start of a pipeline run and
progressively enriched by each agent stage via the .write_agent* methods.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ems_pipeline.models import EntitiesDocument, Entity, Segment, Transcript


class SessionContext(BaseModel):
    """Accumulated pipeline state for a single EMS encounter session."""

    model_config = ConfigDict(extra="forbid")

    # --- Core identity ---
    encounter_id: str

    # --- Agent-1: Transcription & Extraction ---
    patient_context: dict[str, Any] | None = None  # STUB: replace with patient history RAG
    transcript_raw: str | None = None
    transcript_segments: list[Segment] | None = None
    confidence_flags: list[dict[str, Any]] | None = None  # {segment_id, confidence, reason}
    extracted_terms: list[Entity] | None = None
    icd_codes: list[dict[str, Any]] | None = None  # STUB: {code, description, confidence, supporting_segment_ids}
    cpt_codes: list[dict[str, Any]] | None = None  # STUB: same shape as icd_codes
    extraction_confidence: dict[str, float] | None = None  # per entity_type average confidence
    ambiguities: list[dict[str, Any]] | None = None  # {segment_id, text, reason, resolution: None}

    # --- Agent-2: Reporting ---
    report_draft: str | None = None
    clinical_reasoning: str | None = None
    code_suggestions: list[dict[str, Any]] | None = None  # {code, type: "ICD"|"CPT", rationale, evidence_segment_ids}
    citation_map: dict[str, list[str]] | None = None  # {entity_normalized: [segment_id, ...]}

    # --- Agent-3: Claim Submission ---
    claim_id: str | None = None
    payer_id: str | None = None
    submission_status: str | None = None
    pre_submission_flags: list[dict[str, Any]] | None = None
    remediation_request: dict[str, Any] | None = None  # {notes, payer_id, loop_count}

    # --- Agent-4: Denial / Appeal ---
    denial_reason: str | None = None
    appeal_strategy: str | None = None
    appeal_history: list[dict[str, Any]] | None = None  # {denial_id, strategy, outcome, ...}

    # --- Factory ---

    @classmethod
    def create(cls, encounter_id: str | None = None) -> SessionContext:
        """Create a new SessionContext, generating a UUID encounter_id if not provided."""
        return cls(encounter_id=encounter_id or str(uuid.uuid4()))

    # --- Agent write methods ---

    def write_agent1(
        self,
        transcript: Transcript,
        entities: EntitiesDocument,
    ) -> SessionContext:
        """Populate Agent-1 fields from Transcript and EntitiesDocument.

        Computes:
        - transcript_raw: joined segment text
        - transcript_segments: segment list
        - confidence_flags: segments where Segment.confidence < 0.7
        - extracted_terms: entity list
        - extraction_confidence: per entity_type average confidence
        """
        transcript_raw = " ".join(seg.text for seg in transcript.segments)

        # Build confidence_flags for segments below threshold.
        confidence_flags: list[dict[str, Any]] = []
        for i, seg in enumerate(transcript.segments):
            if seg.confidence < 0.7:
                confidence_flags.append(
                    {
                        "segment_id": f"seg_{i:04d}",
                        "confidence": seg.confidence,
                        "reason": "low_asr_confidence",
                    }
                )

        # Compute per entity_type average confidence; skip entities with None confidence.
        type_confidences: dict[str, list[float]] = defaultdict(list)
        for ent in entities.entities:
            if ent.confidence is not None:
                type_confidences[ent.type].append(ent.confidence)

        extraction_confidence: dict[str, float] | None = (
            {
                entity_type: sum(confs) / len(confs)
                for entity_type, confs in type_confidences.items()
            }
            if type_confidences
            else None
        )

        return self.model_copy(
            update={
                "transcript_raw": transcript_raw,
                "transcript_segments": list(transcript.segments),
                "confidence_flags": confidence_flags,
                "extracted_terms": list(entities.entities),
                "extraction_confidence": extraction_confidence,
            }
        )

    def write_agent2(
        self,
        report: str,
        reasoning: str,
        codes: list[dict[str, Any]],
        citation_map: dict[str, list[str]] | None = None,
    ) -> SessionContext:
        """Populate Agent-2 fields: report draft, clinical reasoning, code suggestions."""
        return self.model_copy(
            update={
                "report_draft": report,
                "clinical_reasoning": reasoning,
                "code_suggestions": codes,
                "citation_map": citation_map,
            }
        )

    def write_agent3(
        self,
        claim_id: str,
        payer_id: str | None,
        status: str,
        flags: list[dict[str, Any]],
    ) -> SessionContext:
        """Populate Agent-3 fields: claim submission metadata."""
        return self.model_copy(
            update={
                "claim_id": claim_id,
                "payer_id": payer_id,
                "submission_status": status,
                "pre_submission_flags": flags,
            }
        )

    def write_agent4(
        self,
        denial_reason: str,
        strategy: str,
    ) -> SessionContext:
        """Populate Agent-4 fields: denial reason and appeal strategy."""
        return self.model_copy(
            update={
                "denial_reason": denial_reason,
                "appeal_strategy": strategy,
            }
        )

    # --- Memory footprint helpers ---

    def estimate_size_bytes(self) -> int:
        """Approximate serialized payload size for large-field transport checks."""
        payload = self.model_dump(mode="json")
        total_bytes = 0
        for value in payload.values():
            if isinstance(value, (str, list)):
                total_bytes += len(json.dumps(value))
        return total_bytes

    def trim_for_transport(self, max_bytes: int = 50_000) -> dict[str, Any]:
        """Return compact transport payload when the session grows too large.

        # STUB: real token counting should use tiktoken or Anthropic token APIs.
        """
        if self.estimate_size_bytes() > max_bytes:
            from ems_pipeline.context_utils import compress_agent3_output

            return compress_agent3_output(self)
        return self.model_dump(mode="json")

    # --- Serialization ---

    def to_json(self, path: str | Path) -> None:
        """Serialize this SessionContext to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> SessionContext:
        """Deserialize a SessionContext from a JSON file."""
        path = Path(path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
