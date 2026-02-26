from __future__ import annotations

from ems_pipeline.claim.timeline import build_events
from ems_pipeline.models import Entity, Segment, Transcript


def test_build_events_orders_by_time() -> None:
    transcript = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=3.0,
                speaker="spk0",
                text="00:05 911 call received.",
                confidence=0.9,
            ),
            Segment(
                start=3.0,
                end=6.0,
                speaker="spk0",
                text="00:10 Unit 12 dispatched and responding.",
                confidence=0.9,
            ),
            Segment(
                start=60.0,
                end=64.0,
                speaker="spk1",
                text="01:00 Patient states chest pain started 30 minutes ago.",
                confidence=0.9,
            ),
            Segment(
                start=120.0,
                end=124.0,
                speaker="spk1",
                text="02:00 Administered aspirin 324 milligrams.",
                confidence=0.9,
            ),
            Segment(
                start=180.0,
                end=184.0,
                speaker="spk0",
                text="03:00 Transporting to ED.",
                confidence=0.9,
            ),
        ],
        metadata={
            "segment_id_map": {
                "seg_0000": 0,
                "seg_0001": 1,
                "seg_0002": 2,
                "seg_0003": 3,
                "seg_0004": 4,
            }
        },
    )

    entities = [
        Entity(
            type="symptom",
            text="chest pain",
            normalized="chest pain",
            start=61.5,
            end=62.5,
            speaker="spk1",
            confidence=0.8,
            attributes={},
        ),
        Entity(
            type="intervention",
            text="aspirin 324 mg",
            normalized="aspirin 324 mg",
            start=121.0,
            end=121.5,
            speaker="spk1",
            confidence=0.8,
            attributes={},
        ),
    ]

    events = build_events(transcript, entities)

    starts = [e["start"] for e in events]
    assert starts == sorted(starts)


def test_intervention_after_symptom_mention() -> None:
    transcript = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=2.0,
                speaker="spk0",
                text="00:05 911 call received.",
                confidence=0.9,
            ),
            Segment(
                start=60.0,
                end=63.0,
                speaker="spk1",
                text="01:00 Patient complains of chest pain.",
                confidence=0.9,
            ),
            Segment(
                start=120.0,
                end=123.0,
                speaker="spk1",
                text="02:00 Started oxygen.",
                confidence=0.9,
            ),
        ],
        metadata={"segment_id_map": {"seg_0000": 0, "seg_0001": 1, "seg_0002": 2}},
    )

    entities = [
        Entity(
            type="symptom",
            text="chest pain",
            normalized="chest pain",
            start=60.5,
            end=61.5,
            speaker="spk1",
            confidence=0.8,
            attributes={},
        ),
        Entity(
            type="intervention",
            text="oxygen",
            normalized="oxygen",
            start=120.5,
            end=121.0,
            speaker="spk1",
            confidence=0.8,
            attributes={},
        ),
    ]

    events = build_events(transcript, entities)
    symptom_time = next(e["start"] for e in events if e["type"] == "PATIENT_CONTACT")
    intervention_time = next(e["start"] for e in events if e["type"] == "INTERVENTION")
    assert intervention_time > symptom_time

