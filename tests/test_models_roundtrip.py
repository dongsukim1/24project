from __future__ import annotations

from ems_pipeline.models import (
    Claim,
    EntitiesDocument,
    Entity,
    ProvenanceLink,
    Segment,
    Transcript,
)


def test_transcript_roundtrip() -> None:
    transcript = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=1.23,
                speaker="spk0",
                text="Unit 12 responding.",
                confidence=0.91,
            ),
            Segment(
                start=1.23,
                end=2.50,
                speaker="spk1",
                text="Copy, en route.",
                confidence=0.88,
            ),
        ],
        metadata={"segment_id_map": {"seg_0000": 0, "seg_0001": 1}},
    )

    payload = transcript.model_dump_json()
    recovered = Transcript.model_validate_json(payload)
    assert recovered == transcript


def test_entities_document_roundtrip() -> None:
    doc = EntitiesDocument(
        entities=[
            Entity(
                type="symptom",
                text="chest pain",
                normalized="chest pain",
                start=10.0,
                end=11.0,
                speaker="spk1",
                confidence=0.8,
                attributes={"negated": False, "uncertain": False},
            ),
            Entity(
                type="allergy",
                text="no known drug allergies",
                normalized="NKDA",
                speaker="spk0",
                attributes={"negated": False, "uncertain": False},
            ),
        ],
        metadata={"source": "unit-test"},
    )

    payload = doc.model_dump_json()
    recovered = EntitiesDocument.model_validate_json(payload)
    assert recovered == doc


def test_claim_roundtrip() -> None:
    claim = Claim(
        claim_id="claim_001",
        fields={"chief_complaint": "chest pain"},
        provenance=[
            ProvenanceLink(segment_id="seg_0001", entity_index=0, note="Mentioned by caller"),
        ],
    )

    payload = claim.model_dump_json()
    recovered = Claim.model_validate_json(payload)
    assert recovered == claim

