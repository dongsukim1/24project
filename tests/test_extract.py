from __future__ import annotations

from ems_pipeline.extract import extract_entities
from ems_pipeline.models import Segment, Transcript


def _keys(entities_doc) -> set[tuple[str, str]]:
    return {(e.type, (e.normalized or e.text)) for e in entities_doc.entities}


def test_extract_matches_tiny_gold_case_001() -> None:
    transcript = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=10.0,
                speaker="spk0",
                text=(
                    "Medic 12 on scene with a 55-year-old male complaining of SOB. "
                    "SpO2 88%. BP 150/90. Administered naloxone and started CPR."
                ),
                confidence=0.9,
            )
        ],
        metadata={"segment_id_map": {"seg_0000": 0}},
    )

    doc = extract_entities(transcript)
    assert doc.metadata["transcript_original"]
    assert doc.metadata["transcript_normalized"]

    expected = {
        ("UNIT_ID", "MEDIC12"),
        ("SYMPTOM", "SOB"),
        ("VITAL_SPO2", "SpO2 88%"),
        ("VITAL_BP", "150/90"),
        ("MEDICATION", "naloxone"),
        ("PROCEDURE", "CPR"),
    }
    assert expected <= _keys(doc)


def test_extract_matches_tiny_gold_case_002() -> None:
    transcript = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=5.0,
                speaker="spk1",
                text="STEMI alert. GCS 14 on arrival. ETA 5 minutes to the ER.",
                confidence=0.85,
            )
        ]
    )

    doc = extract_entities(transcript)
    expected = {
        ("CONDITION", "STEMI"),
        ("ASSESSMENT", "GCS 14"),
        ("ETA", "5 minutes"),
    }
    assert expected <= _keys(doc)


def test_extract_matches_tiny_gold_case_003() -> None:
    transcript = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=5.0,
                speaker="spk0",
                text="Engine 3 requests ALS/BLS intercept. Pulse ox 92 and blood pressure 110 over 70.",
                confidence=0.88,
            )
        ]
    )

    doc = extract_entities(transcript)
    expected = {
        ("UNIT_ID", "ENGINE3"),
        ("RESOURCE", "ALS/BLS"),
        ("VITAL_SPO2", "SpO2 92%"),
        ("VITAL_BP", "110/70"),
    }
    assert expected <= _keys(doc)

