from __future__ import annotations

import ems_pipeline.extract as extract_module
from ems_pipeline.extract import extract_entities, merge_entities
from ems_pipeline.models import Entity, Segment, Transcript


def _keys(entities_doc) -> set[tuple[str, str]]:
    return {(e.type, (e.normalized or e.text)) for e in entities_doc.entities}


def _entity(
    *,
    entity_type: str,
    text: str,
    segment_id: str,
    start: float,
    end: float,
    confidence: float | None = 0.9,
) -> Entity:
    return Entity(
        type=entity_type,
        text=text,
        normalized=text,
        start=start,
        end=end,
        speaker="spk0",
        confidence=confidence,
        attributes={"segment_id": segment_id},
    )


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
                text=(
                    "Engine 3 requests ALS/BLS intercept. "
                    "Pulse ox 92 and blood pressure 110 over 70."
                ),
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


def test_merge_entities_dedups_related_overlaps_and_keeps_rule_entities() -> None:
    rule_entities = [
        _entity(
            entity_type="SYMPTOM",
            text="SOB",
            segment_id="seg_0000",
            start=0.0,
            end=10.0,
        )
    ]
    nlp_entities = [
        _entity(
            entity_type="CONDITION",
            text="dyspnea",
            segment_id="seg_0000",
            start=1.0,
            end=2.0,
        ),
        _entity(
            entity_type="ANATOMY",
            text="chest",
            segment_id="seg_0000",
            start=1.0,
            end=2.0,
        ),
        _entity(
            entity_type="CONDITION",
            text="stemi",
            segment_id="seg_0000",
            start=11.0,
            end=12.0,
        ),
    ]

    merged = merge_entities(rule_entities, nlp_entities)

    assert merged[0] is rule_entities[0]
    assert merged[1] is nlp_entities[1]
    assert merged[2] is nlp_entities[2]


def test_merge_entities_applies_confidence_and_nlp_nlp_dedup() -> None:
    nlp_entities = [
        _entity(
            entity_type="DRUG",
            text="aspirin",
            segment_id="seg_0000",
            start=0.0,
            end=5.0,
        ),
        _entity(
            entity_type="MEDICATION",
            text="asa",
            segment_id="seg_0000",
            start=1.0,
            end=4.0,
        ),
        _entity(
            entity_type="ANATOMY",
            text="arm",
            segment_id="seg_0000",
            start=1.0,
            end=4.0,
        ),
        _entity(
            entity_type="SYMPTOM",
            text="pain",
            segment_id="seg_0000",
            start=6.0,
            end=7.0,
            confidence=0.2,
        ),
    ]

    merged = merge_entities([], nlp_entities, confidence_threshold=0.5)

    assert merged[0] is nlp_entities[0]
    assert merged[1] is nlp_entities[2]


def test_extract_entities_merges_nlp_without_dropping_rules(monkeypatch: object) -> None:
    class _StubExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def extract(
            self,
            transcript: Transcript,
            *,
            index_to_segment_id: dict[int, str],
        ) -> list[Entity]:
            self.calls += 1
            seg = transcript.segments[0]
            return [
                Entity(
                    type="CONDITION",
                    text="dyspnea",
                    normalized="dyspnea",
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    confidence=0.9,
                    attributes={"segment_id": index_to_segment_id[0]},
                )
            ]

    calls = {"negation": 0}
    stub_extractor = _StubExtractor()

    def fake_get_nlp_extractor() -> _StubExtractor:
        return stub_extractor

    def fake_apply_negation(entities: list[Entity], transcript: Transcript) -> list[Entity]:
        calls["negation"] += 1
        _ = transcript
        return entities

    monkeypatch.setattr(extract_module, "_get_nlp_extractor", fake_get_nlp_extractor)
    monkeypatch.setattr(extract_module, "apply_negation", fake_apply_negation)

    transcript = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=10.0,
                speaker="spk0",
                text="Patient has SOB.",
                confidence=0.9,
            )
        ],
        metadata={"segment_id_map": {"seg_0000": 0}},
    )

    doc = extract_module.extract_entities(transcript)

    assert stub_extractor.calls == 1
    assert calls["negation"] == 1
    assert any(e.type == "SYMPTOM" and (e.normalized or e.text) == "SOB" for e in doc.entities)
    assert not any(
        e.type == "CONDITION" and (e.normalized or e.text) == "dyspnea" for e in doc.entities
    )


def test_extract_entities_debug_emits_requested_views(monkeypatch: object, capsys) -> None:
    class _StubExtractor:
        confidence_threshold = 0.5

        def extract(self, text: str, segment: Segment) -> list[Entity]:
            _ = text
            return [
                Entity(
                    type="ANATOMY",
                    text="chest",
                    normalized="chest",
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.speaker,
                    confidence=0.83,
                    attributes={"segment_id": "seg_0000", "source": "nlp"},
                )
            ]

        def pop_negation_drops(self) -> list[dict[str, object]]:
            return [
                {
                    "type": "SYMPTOM",
                    "text": "chest pain",
                    "source": "nlp",
                    "segment_id": "seg_0000",
                    "confidence": 0.71,
                    "trigger_term": "denies",
                    "trigger_type": "pre_negation",
                }
            ]

    def fake_get_nlp_extractor() -> _StubExtractor:
        return _StubExtractor()

    def fake_apply_negation(entities: list[Entity], transcript: Transcript) -> list[Entity]:
        _ = transcript
        return entities

    monkeypatch.setattr(extract_module, "_get_nlp_extractor", fake_get_nlp_extractor)
    monkeypatch.setattr(extract_module, "apply_negation", fake_apply_negation)

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

    _ = extract_module.extract_entities(transcript, debug=True)
    captured = capsys.readouterr()

    assert "[extract.debug] rule_entities:" in captured.err
    assert "[extract.debug] nlp_entities_pre_merge:" in captured.err
    assert "[extract.debug] entities_removed_by_negation:" in captured.err
    assert "[extract.debug] final_merged_entities:" in captured.err
    assert '"source": "rule"' in captured.err
    assert '"source": "nlp"' in captured.err
    assert '"confidence": 0.83' in captured.err
    assert '"trigger_term": "denies"' in captured.err
    assert '"trigger_type": "pre_negation"' in captured.err
