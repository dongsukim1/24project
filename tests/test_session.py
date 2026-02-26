"""Tests for SessionContext: save/load round-trip and agent write methods.

Fixtures mirror the pattern used in test_claim_builder.py and
test_models_roundtrip.py so no synthetic patient data is generated.
"""

from __future__ import annotations

import json

import pytest

from ems_pipeline.models import Entity, EntitiesDocument, Segment, Transcript
from ems_pipeline.session import SessionContext


# ---------------------------------------------------------------------------
# Shared fixtures — same structure as test_claim_builder._toy_transcript /
# _toy_entities so the test suite stays internally consistent.
# ---------------------------------------------------------------------------


def _fixture_transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(
                start=0.0,
                end=3.0,
                speaker="spk0",
                text="00:05 911 call received at 123 Main St.",
                confidence=0.9,
            ),
            Segment(
                start=3.0,
                end=6.0,
                speaker="spk0",
                text="00:10 Unit 12 dispatched priority 1 for chest pain.",
                confidence=0.9,
            ),
            Segment(
                start=60.0,
                end=66.0,
                speaker="spk1",
                text="01:00 Patient is a 54 year old male complaining of chest pain.",
                # Deliberately low to exercise confidence_flags
                confidence=0.65,
            ),
            Segment(
                start=80.0,
                end=84.0,
                speaker="spk1",
                text="01:20 BP 150/90, SpO2 92%.",
                confidence=0.88,
            ),
        ],
        metadata={
            "audio_filename": "toy.wav",
            "segment_id_map": {
                "seg_0000": 0,
                "seg_0001": 1,
                "seg_0002": 2,
                "seg_0003": 3,
            },
        },
    )


def _fixture_entities() -> EntitiesDocument:
    return EntitiesDocument(
        entities=[
            Entity(
                type="symptom",
                text="chest pain",
                normalized="chest pain",
                start=62.5,
                end=63.0,
                confidence=0.9,
            ),
            Entity(
                type="vital",
                text="BP 150/90",
                normalized="150/90",
                start=80.5,
                end=81.5,
                confidence=0.85,
                attributes={"name": "BP", "value": "150/90", "unit": "mmHg"},
            ),
            Entity(
                type="vital",
                text="SpO2 92%",
                normalized="92%",
                start=82.0,
                end=83.0,
                confidence=0.92,
                attributes={"name": "SpO2", "value": "92%", "unit": "%"},
            ),
            # Entity with no confidence — should be excluded from extraction_confidence
            Entity(
                type="unit_id",
                text="Unit 12",
                normalized="UNIT12",
                start=3.1,
                end=3.4,
                confidence=None,
            ),
        ],
        metadata={"source": "unit-test"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_generates_encounter_id() -> None:
    ctx = SessionContext.create()
    assert isinstance(ctx.encounter_id, str)
    assert len(ctx.encounter_id) > 0


def test_create_accepts_explicit_encounter_id() -> None:
    ctx = SessionContext.create(encounter_id="enc-001")
    assert ctx.encounter_id == "enc-001"


def test_write_agent1_populates_fields() -> None:
    transcript = _fixture_transcript()
    entities = _fixture_entities()

    ctx = SessionContext.create(encounter_id="enc-test-001").write_agent1(transcript, entities)

    # transcript_raw is the concatenation of all segment texts
    assert ctx.transcript_raw == " ".join(seg.text for seg in transcript.segments)

    # transcript_segments mirrors the input
    assert ctx.transcript_segments is not None
    assert len(ctx.transcript_segments) == len(transcript.segments)
    assert ctx.transcript_segments[0] == transcript.segments[0]

    # confidence_flags: only segment index 2 has confidence 0.65 < 0.7
    assert ctx.confidence_flags is not None
    assert len(ctx.confidence_flags) == 1
    flag = ctx.confidence_flags[0]
    assert flag["segment_id"] == "seg_0002"
    assert flag["confidence"] == pytest.approx(0.65)
    assert flag["reason"] == "low_asr_confidence"

    # extracted_terms mirrors entities
    assert ctx.extracted_terms is not None
    assert len(ctx.extracted_terms) == len(entities.entities)

    # extraction_confidence: unit_id excluded (confidence=None), averages for symptom + vital
    assert ctx.extraction_confidence is not None
    assert "symptom" in ctx.extraction_confidence
    assert ctx.extraction_confidence["symptom"] == pytest.approx(0.9)
    assert "vital" in ctx.extraction_confidence
    assert ctx.extraction_confidence["vital"] == pytest.approx((0.85 + 0.92) / 2)
    assert "unit_id" not in ctx.extraction_confidence


def test_write_agent2_populates_fields() -> None:
    ctx = SessionContext.create(encounter_id="enc-test-002").write_agent2(
        report="Patient presented with chest pain.",
        reasoning="Ruled out STEMI based on 12-lead.",
        codes=[{"code": "R07.9", "type": "ICD", "rationale": "Chest pain NOS", "evidence_segment_ids": ["seg_0002"]}],
    )

    assert ctx.report_draft == "Patient presented with chest pain."
    assert ctx.clinical_reasoning == "Ruled out STEMI based on 12-lead."
    assert ctx.code_suggestions is not None
    assert ctx.code_suggestions[0]["code"] == "R07.9"


def test_write_agent3_populates_fields() -> None:
    ctx = SessionContext.create(encounter_id="enc-test-003").write_agent3(
        claim_id="claim-abc",
        payer_id="payer-xyz",
        status="submitted",
        flags=[],
    )

    assert ctx.claim_id == "claim-abc"
    assert ctx.payer_id == "payer-xyz"
    assert ctx.submission_status == "submitted"
    assert ctx.pre_submission_flags == []


def test_write_agent4_populates_fields() -> None:
    ctx = SessionContext.create(encounter_id="enc-test-004").write_agent4(
        denial_reason="Missing primary diagnosis code.",
        strategy="Resubmit with corrected ICD-10 codes.",
    )

    assert ctx.denial_reason == "Missing primary diagnosis code."
    assert ctx.appeal_strategy == "Resubmit with corrected ICD-10 codes."


def test_write_methods_are_non_destructive() -> None:
    """Each write call returns a new instance; the original is unchanged."""
    original = SessionContext.create(encounter_id="enc-immutable")
    updated = original.write_agent2(
        report="draft",
        reasoning="reasoning",
        codes=[],
    )

    assert original.report_draft is None
    assert updated.report_draft == "draft"
    # encounter_id is preserved across write calls
    assert updated.encounter_id == original.encounter_id


def test_round_trip_empty_session(tmp_path) -> None:
    """A freshly-created SessionContext survives to_json / from_json."""
    ctx = SessionContext.create(encounter_id="enc-roundtrip-empty")
    path = tmp_path / "session_empty.json"

    ctx.to_json(path)
    recovered = SessionContext.from_json(path)

    assert recovered == ctx
    assert recovered.encounter_id == "enc-roundtrip-empty"


def test_round_trip_after_agent1(tmp_path) -> None:
    """A SessionContext populated by write_agent1 survives to_json / from_json."""
    transcript = _fixture_transcript()
    entities = _fixture_entities()

    ctx = SessionContext.create(encounter_id="enc-roundtrip-agent1").write_agent1(transcript, entities)
    path = tmp_path / "session_agent1.json"

    ctx.to_json(path)
    recovered = SessionContext.from_json(path)

    assert recovered == ctx
    assert recovered.transcript_segments is not None
    assert len(recovered.transcript_segments) == len(transcript.segments)
    assert recovered.confidence_flags is not None
    assert len(recovered.confidence_flags) == 1


def test_round_trip_full_pipeline(tmp_path) -> None:
    """A fully-populated SessionContext (all agents) survives to_json / from_json."""
    transcript = _fixture_transcript()
    entities = _fixture_entities()

    ctx = (
        SessionContext.create(encounter_id="enc-roundtrip-full")
        .write_agent1(transcript, entities)
        .write_agent2(
            report="Chest pain, rule out STEMI.",
            reasoning="12-lead negative.",
            codes=[{"code": "R07.9", "type": "ICD", "rationale": "Chest pain", "evidence_segment_ids": ["seg_0002"]}],
        )
        .write_agent3(
            claim_id="claim-full",
            payer_id="payer-full",
            status="submitted",
            flags=[{"field": "dx_code", "issue": "unspecified"}],
        )
        .write_agent4(
            denial_reason="Unspecified diagnosis.",
            strategy="Resubmit with specific ICD-10.",
        )
    )

    path = tmp_path / "session_full.json"
    ctx.to_json(path)
    recovered = SessionContext.from_json(path)

    assert recovered == ctx
    assert recovered.denial_reason == "Unspecified diagnosis."
    assert recovered.claim_id == "claim-full"


def test_json_file_is_valid_json(tmp_path) -> None:
    """The written file is well-formed JSON with the expected top-level keys."""
    ctx = SessionContext.create(encounter_id="enc-json-check")
    path = tmp_path / "session_check.json"
    ctx.to_json(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["encounter_id"] == "enc-json-check"
    # All optional fields absent at creation should be serialised as null
    assert data["transcript_raw"] is None


def test_extra_fields_forbidden() -> None:
    """SessionContext rejects unknown fields (extra='forbid')."""
    with pytest.raises(Exception):
        SessionContext(encounter_id="enc-bad", unknown_field="oops")  # type: ignore[call-arg]
