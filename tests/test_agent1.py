"""Tests for Agent-1: Transcription & Extraction.

Covers:
- compute_extraction_confidence() with hand-constructed EntitiesDocument
- confidence_flags population from a Transcript with low-confidence segments
- Full agent1.run() path with transcribe_audio and extract_entities mocked
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ems_pipeline.agents.agent1 import (
    Agent1Options,
    _ambiguities_for_entities,
    _confidence_flags_for_transcript,
    compute_extraction_confidence,
    run,
)
from ems_pipeline.models import Entity, EntitiesDocument, Segment, Transcript
from ems_pipeline.session import SessionContext


# ---------------------------------------------------------------------------
# Shared fixtures (inline, consistent with test_claim_builder.py pattern)
# ---------------------------------------------------------------------------


def _fixture_transcript() -> Transcript:
    """Three segments: one below 0.7, one below 0.6, one high-confidence."""
    return Transcript(
        segments=[
            Segment(
                start=0.0,
                end=3.0,
                speaker="spk0",
                text="00:05 911 call received at 123 Main St.",
                confidence=0.9,  # high — no flags
            ),
            Segment(
                start=3.0,
                end=6.0,
                speaker="spk0",
                text="00:10 Unit 12 dispatched priority 1 for chest pain.",
                confidence=0.65,  # < 0.7 → flagged at default threshold
            ),
            Segment(
                start=60.0,
                end=66.0,
                speaker="spk1",
                text="01:00 Patient is a 54 year old male complaining of chest pain.",
                confidence=0.5,  # < 0.6 → flagged at both thresholds
            ),
        ],
        metadata={
            "audio_filename": "toy.wav",
            "segment_id_map": {
                "seg_0000": 0,
                "seg_0001": 1,
                "seg_0002": 2,
            },
        },
    )


def _fixture_entities_doc() -> EntitiesDocument:
    """Hand-constructed entities covering the ambiguity and confidence cases."""
    return EntitiesDocument(
        entities=[
            # High-confidence, normalised — no flags or ambiguities
            Entity(
                type="symptom",
                text="chest pain",
                normalized="chest pain",
                start=62.5,
                end=63.0,
                confidence=0.9,
                attributes={"segment_id": "seg_0002"},
            ),
            # Confidence is None → should contribute 0.5 to extraction_confidence
            Entity(
                type="unit_id",
                text="Unit 12",
                normalized="UNIT12",
                start=3.1,
                end=3.4,
                confidence=None,
                attributes={"segment_id": "seg_0001"},
            ),
            # Low confidence < 0.6 → ambiguity: low_confidence_extraction
            Entity(
                type="vital",
                text="BP 150/90",
                normalized="150/90",
                start=80.5,
                end=81.5,
                confidence=0.55,
                attributes={"segment_id": "seg_0003"},
            ),
            # normalized is None → ambiguity: unresolved_normalization
            Entity(
                type="procedure",
                text="unknown_proc",
                normalized=None,
                start=90.0,
                end=91.0,
                confidence=0.8,
                attributes={"segment_id": "seg_0003"},
            ),
            # Both: confidence < 0.6 AND normalized is None → two ambiguity entries
            Entity(
                type="medication",
                text="mystery_drug",
                normalized=None,
                start=100.0,
                end=101.0,
                confidence=0.45,
                attributes={"segment_id": "seg_0004"},
            ),
        ],
        metadata={"source": "unit-test"},
    )


# ---------------------------------------------------------------------------
# compute_extraction_confidence
# ---------------------------------------------------------------------------


def test_compute_extraction_confidence_with_explicit_values() -> None:
    """Entities with explicit confidence are averaged per type."""
    doc = EntitiesDocument(
        entities=[
            Entity(type="symptom", text="a", confidence=0.8),
            Entity(type="symptom", text="b", confidence=0.6),
            Entity(type="vital", text="c", confidence=1.0),
        ]
    )
    result = compute_extraction_confidence(doc)

    assert "symptom" in result
    assert result["symptom"] == pytest.approx((0.8 + 0.6) / 2)
    assert "vital" in result
    assert result["vital"] == pytest.approx(1.0)


def test_compute_extraction_confidence_none_treated_as_half() -> None:
    """None confidence contributes 0.5 to the per-type average."""
    doc = EntitiesDocument(
        entities=[
            Entity(type="unit_id", text="Unit 1", confidence=None),
            Entity(type="unit_id", text="Unit 2", confidence=None),
            Entity(type="symptom", text="SOB", confidence=1.0),
            Entity(type="symptom", text="pain", confidence=None),
        ]
    )
    result = compute_extraction_confidence(doc)

    # unit_id: both None → average of [0.5, 0.5]
    assert result["unit_id"] == pytest.approx(0.5)
    # symptom: [1.0, 0.5] → average 0.75
    assert result["symptom"] == pytest.approx(0.75)


def test_compute_extraction_confidence_empty_document() -> None:
    """Empty document returns an empty dict (no entities → no types)."""
    doc = EntitiesDocument(entities=[])
    assert compute_extraction_confidence(doc) == {}


def test_compute_extraction_confidence_mixed() -> None:
    """Mix of None and real confidence values in the fixture document."""
    doc = _fixture_entities_doc()
    result = compute_extraction_confidence(doc)

    # symptom: [0.9] → 0.9
    assert result["symptom"] == pytest.approx(0.9)
    # unit_id: [None→0.5] → 0.5
    assert result["unit_id"] == pytest.approx(0.5)
    # vital: [0.55] → 0.55
    assert result["vital"] == pytest.approx(0.55)
    # procedure: [0.8] → 0.8
    assert result["procedure"] == pytest.approx(0.8)
    # medication: [0.45] → 0.45
    assert result["medication"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# _confidence_flags_for_transcript
# ---------------------------------------------------------------------------


def test_confidence_flags_default_threshold() -> None:
    """Segments with confidence < 0.7 are flagged at the default threshold."""
    transcript = _fixture_transcript()
    flags = _confidence_flags_for_transcript(transcript, threshold=0.7)

    # seg_0001 (0.65) and seg_0002 (0.5) are below 0.7; seg_0000 (0.9) is not.
    assert len(flags) == 2
    segment_ids = {f["segment_id"] for f in flags}
    assert "seg_0001" in segment_ids
    assert "seg_0002" in segment_ids
    assert "seg_0000" not in segment_ids


def test_confidence_flags_reason_field() -> None:
    """Every flag entry carries reason='low_asr_confidence'."""
    transcript = _fixture_transcript()
    flags = _confidence_flags_for_transcript(transcript, threshold=0.7)
    assert all(f["reason"] == "low_asr_confidence" for f in flags)


def test_confidence_flags_custom_threshold() -> None:
    """Only the very-low segment is flagged when threshold is tightened to 0.6."""
    transcript = _fixture_transcript()
    flags = _confidence_flags_for_transcript(transcript, threshold=0.6)

    # Only seg_0002 (0.5) is below 0.6; seg_0001 (0.65) is not.
    assert len(flags) == 1
    assert flags[0]["segment_id"] == "seg_0002"
    assert flags[0]["confidence"] == pytest.approx(0.5)


def test_confidence_flags_none_when_all_high() -> None:
    """No flags produced when all segments exceed the threshold."""
    transcript = Transcript(
        segments=[
            Segment(start=0.0, end=1.0, speaker="spk0", text="ok", confidence=0.95),
            Segment(start=1.0, end=2.0, speaker="spk0", text="ok", confidence=0.80),
        ]
    )
    flags = _confidence_flags_for_transcript(transcript, threshold=0.7)
    assert flags == []


# ---------------------------------------------------------------------------
# _ambiguities_for_entities
# ---------------------------------------------------------------------------


def test_ambiguities_low_confidence() -> None:
    """Entities with confidence < 0.6 produce a low_confidence_extraction entry."""
    doc = EntitiesDocument(
        entities=[
            Entity(type="vital", text="BP 150/90", normalized="150/90", confidence=0.55),
        ]
    )
    ambs = _ambiguities_for_entities(doc)
    assert len(ambs) == 1
    assert ambs[0]["reason"] == "low_confidence_extraction"
    assert ambs[0]["text"] == "BP 150/90"
    assert ambs[0]["resolution"] is None


def test_ambiguities_unresolved_normalization() -> None:
    """Entities with normalized=None produce an unresolved_normalization entry."""
    doc = EntitiesDocument(
        entities=[
            Entity(type="procedure", text="unknown_proc", normalized=None, confidence=0.9),
        ]
    )
    ambs = _ambiguities_for_entities(doc)
    assert len(ambs) == 1
    assert ambs[0]["reason"] == "unresolved_normalization"
    assert ambs[0]["resolution"] is None


def test_ambiguities_both_conditions_emit_two_entries() -> None:
    """An entity matching both conditions produces two separate ambiguity entries."""
    doc = EntitiesDocument(
        entities=[
            Entity(type="medication", text="mystery_drug", normalized=None, confidence=0.45),
        ]
    )
    ambs = _ambiguities_for_entities(doc)
    assert len(ambs) == 2
    reasons = {a["reason"] for a in ambs}
    assert "low_confidence_extraction" in reasons
    assert "unresolved_normalization" in reasons


def test_ambiguities_segment_id_from_attributes() -> None:
    """The segment_id in an ambiguity entry comes from entity.attributes."""
    doc = EntitiesDocument(
        entities=[
            Entity(
                type="vital",
                text="x",
                normalized=None,
                confidence=0.4,
                attributes={"segment_id": "seg_0007"},
            ),
        ]
    )
    ambs = _ambiguities_for_entities(doc)
    for a in ambs:
        assert a["segment_id"] == "seg_0007"


def test_ambiguities_clean_entity_produces_no_entry() -> None:
    """A high-confidence, normalised entity does not produce any ambiguity."""
    doc = EntitiesDocument(
        entities=[
            Entity(type="symptom", text="chest pain", normalized="chest pain", confidence=0.9),
        ]
    )
    assert _ambiguities_for_entities(doc) == []


def test_ambiguities_none_confidence_not_flagged() -> None:
    """Entities with confidence=None are NOT flagged as low-confidence (checked explicitly)."""
    doc = EntitiesDocument(
        entities=[
            Entity(type="unit_id", text="Unit 1", normalized="UNIT1", confidence=None),
        ]
    )
    ambs = _ambiguities_for_entities(doc)
    # confidence=None is not < 0.6, and normalized is set → no ambiguity
    assert ambs == []


def test_ambiguities_full_fixture() -> None:
    """The full fixture document produces the expected number of ambiguity entries."""
    doc = _fixture_entities_doc()
    ambs = _ambiguities_for_entities(doc)
    # vital (0.55) → 1, procedure (None norm) → 1, medication (0.45 + None norm) → 2
    assert len(ambs) == 4


# ---------------------------------------------------------------------------
# Full agent1.run() with mocked I/O
# ---------------------------------------------------------------------------


def test_agent1_run_mocked_end_to_end() -> None:
    """agent1.run() populates session correctly when ASR and extraction are mocked."""
    transcript = _fixture_transcript()
    entities_doc = _fixture_entities_doc()

    session = SessionContext.create(encounter_id="enc-agent1-mock")
    options = Agent1Options()  # defaults: threshold=0.7

    with (
        patch("ems_pipeline.agents.agent1.transcribe_audio", return_value=transcript),
        patch("ems_pipeline.agents.agent1.extract_entities", return_value=entities_doc),
    ):
        result = run("fake_audio.wav", session, options)

    # Identity is preserved
    assert result.encounter_id == "enc-agent1-mock"

    # Structural transcript fields
    assert result.transcript_raw is not None
    assert result.transcript_segments is not None
    assert len(result.transcript_segments) == 3

    # Extracted terms
    assert result.extracted_terms is not None
    assert len(result.extracted_terms) == len(entities_doc.entities)

    # Confidence flags: seg_0001 (0.65) and seg_0002 (0.5) < default 0.7
    assert result.confidence_flags is not None
    flagged_ids = {f["segment_id"] for f in result.confidence_flags}
    assert "seg_0001" in flagged_ids
    assert "seg_0002" in flagged_ids

    # Ambiguities: populated (not None)
    assert result.ambiguities is not None

    # Extraction confidence: populated (not None)
    assert result.extraction_confidence is not None


def test_agent1_run_custom_threshold_mocked() -> None:
    """A custom confidence_threshold is respected in the run() output."""
    transcript = _fixture_transcript()
    entities_doc = _fixture_entities_doc()

    session = SessionContext.create(encounter_id="enc-threshold")
    options = Agent1Options(confidence_threshold=0.6)

    with (
        patch("ems_pipeline.agents.agent1.transcribe_audio", return_value=transcript),
        patch("ems_pipeline.agents.agent1.extract_entities", return_value=entities_doc),
    ):
        result = run("fake_audio.wav", session, options)

    # Only seg_0002 (0.5) is below 0.6; seg_0001 (0.65) is above.
    assert result.confidence_flags is not None
    flagged_ids = {f["segment_id"] for f in result.confidence_flags}
    assert "seg_0002" in flagged_ids
    assert "seg_0001" not in flagged_ids


def test_agent1_run_original_session_not_mutated() -> None:
    """The SessionContext passed to run() is not mutated (immutable write pattern)."""
    transcript = _fixture_transcript()
    entities_doc = _fixture_entities_doc()

    original = SessionContext.create(encounter_id="enc-immutable")

    with (
        patch("ems_pipeline.agents.agent1.transcribe_audio", return_value=transcript),
        patch("ems_pipeline.agents.agent1.extract_entities", return_value=entities_doc),
    ):
        updated = run("fake_audio.wav", original, Agent1Options())

    assert original.transcript_raw is None
    assert updated.transcript_raw is not None
    assert original.encounter_id == updated.encounter_id


def test_agent1_options_defaults() -> None:
    """Agent1Options exposes the correct defaults."""
    opts = Agent1Options()
    assert opts.bandpass is False
    assert opts.denoise is False
    assert opts.asr_model == "base"
    assert opts.confidence_threshold == pytest.approx(0.7)


def test_agent1_options_extra_fields_forbidden() -> None:
    """Agent1Options rejects unknown fields (extra='forbid')."""
    with pytest.raises(Exception):
        Agent1Options(unknown_field="oops")  # type: ignore[call-arg]
