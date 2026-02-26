from __future__ import annotations

import hashlib

from ems_pipeline.claim.builder import build_claim
from ems_pipeline.claim.timeline import build_events
from ems_pipeline.models import Entity, Segment, Transcript


def _toy_transcript() -> Transcript:
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
                text="01:00 Patient is a 54 year old male complaining of chest pain and shortness of breath.",
                confidence=0.9,
            ),
            Segment(
                start=80.0,
                end=84.0,
                speaker="spk1",
                text="01:20 BP 150/90, SpO2 92%.",
                confidence=0.9,
            ),
            Segment(
                start=120.0,
                end=125.0,
                speaker="spk1",
                text="02:00 Administered aspirin 324 mg and started oxygen.",
                confidence=0.9,
            ),
            Segment(
                start=180.0,
                end=184.0,
                speaker="spk0",
                text="03:00 Transporting to Mercy Hospital ED.",
                confidence=0.9,
            ),
        ],
        metadata={
            "audio_filename": "toy.wav",
            "segment_id_map": {
                "seg_0000": 0,
                "seg_0001": 1,
                "seg_0002": 2,
                "seg_0003": 3,
                "seg_0004": 4,
                "seg_0005": 5,
            },
        },
    )


def _toy_entities() -> list[Entity]:
    return [
        Entity(type="location", text="123 Main St", normalized="123 Main St", start=0.5, end=1.0),
        Entity(type="unit_id", text="Unit 12", normalized="UNIT12", start=3.1, end=3.4),
        Entity(type="priority", text="priority 1", normalized="1", start=3.8, end=4.2),
        Entity(type="incident_type", text="chest pain", normalized="chest pain", start=4.6, end=5.1),
        Entity(type="age", text="54", normalized="54", start=61.0, end=61.2),
        Entity(type="sex", text="male", normalized="male", start=61.8, end=62.1),
        Entity(type="chief_complaint", text="chest pain", normalized="chest pain", start=62.5, end=63.0),
        Entity(type="symptom", text="shortness of breath", normalized="SOB", start=63.2, end=64.5),
        Entity(
            type="vital",
            text="BP 150/90",
            normalized="150/90",
            start=80.5,
            end=81.5,
            attributes={"name": "BP", "value": "150/90", "unit": "mmHg"},
        ),
        Entity(
            type="vital",
            text="SpO2 92%",
            normalized="92%",
            start=82.0,
            end=83.0,
            attributes={"name": "SpO2", "value": "92%", "unit": "%"},
        ),
        Entity(type="medication", text="aspirin 324 mg", normalized="aspirin 324 mg", start=121.0, end=121.8),
        Entity(type="procedure", text="oxygen", normalized="oxygen", start=123.0, end=123.5),
        Entity(type="destination", text="Mercy Hospital", normalized="Mercy Hospital", start=181.0, end=182.0),
        Entity(type="disposition", text="transported", normalized="transported", start=180.0, end=184.0),
    ]


def test_build_claim_output_json_structure() -> None:
    transcript = _toy_transcript()
    entities = _toy_entities()
    events = build_events(transcript, entities)

    claim, claim_json = build_claim(transcript, entities, events)

    expected_incident_id = hashlib.sha256("toy.wav:0.000".encode("utf-8")).hexdigest()
    assert claim.claim_id == expected_incident_id
    assert claim_json["claim_id"] == expected_incident_id
    assert claim_json["fields"]["incident_id"] == expected_incident_id

    assert set(claim_json.keys()) == {"schema_version", "claim_id", "fields", "provenance"}
    assert set(claim_json["fields"].keys()) >= {
        "incident_id",
        "location_hint",
        "dispatch",
        "patient",
        "primary_impression",
        "findings",
        "interventions",
        "disposition",
        "provenance",
    }

    assert claim_json["fields"]["dispatch"]["units"]
    assert claim_json["fields"]["findings"]["symptoms"]
    assert claim_json["fields"]["findings"]["vitals"]
    assert claim_json["fields"]["interventions"]


def test_every_applicable_field_has_evidence_segment_ids() -> None:
    transcript = _toy_transcript()
    entities = _toy_entities()
    events = build_events(transcript, entities)

    _, claim_json = build_claim(transcript, entities, events)
    f = claim_json["fields"]

    assert f["location_hint"]["evidence_segment_ids"] == ["seg_0000"]

    dispatch = f["dispatch"]
    assert "seg_0001" in dispatch["incident_type"]["evidence_segment_ids"]
    assert "seg_0001" in dispatch["priority"]["evidence_segment_ids"]
    assert dispatch["units"][0]["evidence_segment_ids"] == ["seg_0001"]

    patient = f["patient"]
    assert patient["age_hint"]["evidence_segment_ids"] == ["seg_0002"]
    assert patient["sex_hint"]["evidence_segment_ids"] == ["seg_0002"]

    assert f["primary_impression"]["evidence_segment_ids"] == ["seg_0002"]

    symptoms = f["findings"]["symptoms"]
    assert symptoms[0]["evidence_segment_ids"] == ["seg_0002"]

    vitals = f["findings"]["vitals"]
    assert any(v["evidence_segment_ids"] == ["seg_0003"] for v in vitals)

    interventions = f["interventions"]
    assert any(item["evidence_segment_ids"] == ["seg_0004"] for item in interventions)

    disposition = f["disposition"]
    assert disposition["status"]["evidence_segment_ids"] == ["seg_0005"]
    assert disposition["destination_hint"]["evidence_segment_ids"] == ["seg_0005"]
    assert "seg_0005" in f["provenance"]["evidence_segment_ids"]

