"""Agent 1: Transcription & Extraction.

Wraps the existing transcribe → extract pipeline stages and writes structured
output into a SessionContext.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ems_pipeline.extract import extract_entities
from ems_pipeline.models import EntitiesDocument, Transcript
from ems_pipeline.session import SessionContext
from ems_pipeline.transcribe import transcribe_audio


class Agent1Options(BaseModel):
    """Tunable parameters for Agent-1: Transcription & Extraction."""

    model_config = ConfigDict(extra="forbid")

    bandpass: bool = False
    denoise: bool = False
    asr_model: str = "base"
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


def compute_extraction_confidence(entities_doc: EntitiesDocument) -> dict[str, float]:
    """Compute average extraction confidence per entity type.

    For Entity objects where confidence is None, a default of 0.5 is used.

    Note: None confidence is treated as 0.5 (mid-point assumption) — rule-based
    extractors that do not emit per-entity confidence scores are assumed to be no
    better than chance, rather than being penalised with 0.0.
    """
    type_confidences: dict[str, list[float]] = defaultdict(list)
    for ent in entities_doc.entities:
        # None confidence → 0.5 (mid-point assumption; see docstring).
        conf = ent.confidence if ent.confidence is not None else 0.5
        type_confidences[ent.type].append(conf)

    return {
        entity_type: sum(confs) / len(confs)
        for entity_type, confs in type_confidences.items()
    }


def _confidence_flags_for_transcript(
    transcript: Transcript,
    threshold: float,
) -> list[dict[str, Any]]:
    """Return confidence flag entries for segments below *threshold*."""
    flags: list[dict[str, Any]] = []
    for i, seg in enumerate(transcript.segments):
        if seg.confidence < threshold:
            flags.append(
                {
                    "segment_id": f"seg_{i:04d}",
                    "confidence": seg.confidence,
                    "reason": "low_asr_confidence",
                }
            )
    return flags


def _ambiguities_for_entities(entities_doc: EntitiesDocument) -> list[dict[str, Any]]:
    """Return ambiguity entries for low-confidence or un-normalised entities.

    Checks performed per entity (each may produce multiple entries):
    - confidence < 0.6  → reason="low_confidence_extraction"
    - normalized is None → reason="unresolved_normalization"
    """
    ambiguities: list[dict[str, Any]] = []
    for ent in entities_doc.entities:
        seg_id = ent.attributes.get("segment_id") if isinstance(ent.attributes, dict) else None

        if ent.confidence is not None and ent.confidence < 0.6:
            ambiguities.append(
                {
                    "segment_id": seg_id,
                    "text": ent.text,
                    "reason": "low_confidence_extraction",
                    "resolution": None,
                }
            )

        if ent.normalized is None:
            ambiguities.append(
                {
                    "segment_id": seg_id,
                    "text": ent.text,
                    "reason": "unresolved_normalization",
                    "resolution": None,
                }
            )

    return ambiguities


def run(
    audio_path: str | Path,
    session: SessionContext,
    options: Agent1Options,
) -> SessionContext:
    """Run Agent-1: transcribe audio and extract entities into SessionContext.

    Steps:
    1. Transcribe audio → Transcript (sets EMS_ASR_MODEL from options).
    2. Extract entities → EntitiesDocument.
    3. Populate session structural fields via session.write_agent1().
    4. Override confidence_flags using options.confidence_threshold.
    5. Populate ambiguities for low-confidence or un-normalised entities.
    6. Compute extraction_confidence with 0.5 fallback for None values.

    Args:
        audio_path: Path to the input audio file.
        session: Current SessionContext (not mutated; returns updated copy).
        options: Tunable parameters for transcription and extraction.

    Returns:
        Updated SessionContext with all Agent-1 fields populated.
    """
    # Set EMS_ASR_MODEL for the duration of the transcription call; restore on exit.
    _prev_model = os.environ.get("EMS_ASR_MODEL")
    try:
        os.environ["EMS_ASR_MODEL"] = options.asr_model
        transcript = transcribe_audio(
            audio_path,
            bandpass=options.bandpass,
            denoise=options.denoise,
        )
    finally:
        if _prev_model is None:
            os.environ.pop("EMS_ASR_MODEL", None)
        else:
            os.environ["EMS_ASR_MODEL"] = _prev_model

    entities_doc = extract_entities(transcript)

    # Populate structural transcript/entity fields.
    session = session.write_agent1(transcript, entities_doc)

    # Recompute flags with the agent's configurable threshold.
    confidence_flags = _confidence_flags_for_transcript(transcript, options.confidence_threshold)

    # Detect ambiguous entities.
    ambiguities = _ambiguities_for_entities(entities_doc)

    # Recompute extraction confidence using the 0.5 fallback for None values.
    extraction_confidence: dict[str, float] | None = compute_extraction_confidence(entities_doc) or None

    return session.model_copy(
        update={
            "confidence_flags": confidence_flags,
            "ambiguities": ambiguities,
            "extraction_confidence": extraction_confidence,
        }
    )
