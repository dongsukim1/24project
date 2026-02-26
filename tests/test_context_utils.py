"""Tests for stage-boundary context compression and provenance helpers."""

from __future__ import annotations

import pytest

from ems_pipeline.context_utils import (
    build_provenance_chain,
    compress_agent1_output,
    compress_agent3_output,
    tag_citations,
)
from ems_pipeline.models import Entity, Segment
from ems_pipeline.session import SessionContext


def _session_with_segments() -> SessionContext:
    segments = [
        Segment(
            start=float(i),
            end=float(i + 1),
            speaker="medic" if i < 5 else "patient",
            text=f"segment {i}",
            confidence=0.5 + (i * 0.04),
        )
        for i in range(10)
    ]
    entities = [
        Entity(type="symptom", text="chest pain", normalized="chest pain", attributes={"segment_id": "seg_0001"}),
        Entity(type="symptom", text="dyspnea", normalized="dyspnea", attributes={"segment_id": "seg_0002"}),
        Entity(type="vital", text="BP 150/90", normalized="150/90", attributes={"segment_id": "seg_0006"}),
    ]
    return SessionContext.create(encounter_id="enc-context-001").model_copy(
        update={
            "transcript_raw": " ".join(segment.text for segment in segments),
            "transcript_segments": segments,
            "extracted_terms": entities,
            "confidence_flags": [
                {"segment_id": "seg_0004", "confidence": 0.65, "reason": "low_asr_confidence"},
                {"segment_id": "seg_0000", "confidence": 0.52, "reason": "low_asr_confidence"},
                {"segment_id": "seg_0003", "confidence": 0.61, "reason": "low_asr_confidence"},
            ],
            "ambiguities": [
                {"segment_id": "seg_0002", "text": "dyspnea", "reason": "unresolved_normalization", "resolution": None}
            ],
        }
    )


def test_compress_agent1_output_groups_speakers_and_excludes_raw_text() -> None:
    session = _session_with_segments()

    compressed = compress_agent1_output(session)

    assert "transcript_raw" not in compressed
    assert "transcript_segments" not in compressed
    assert compressed["total_segment_count"] == 10
    assert compressed["entity_counts_by_type"] == {"symptom": 2, "vital": 1}
    assert compressed["top_confidence_flags"][0]["segment_id"] == "seg_0000"
    assert compressed["top_confidence_flags"][1]["segment_id"] == "seg_0003"
    assert compressed["top_confidence_flags"][2]["segment_id"] == "seg_0004"

    summary = {item["speaker"]: item for item in compressed["transcript_segments_summary"]}
    assert summary["medic"]["segment_count"] == 5
    assert summary["patient"]["segment_count"] == 5
    assert summary["medic"]["avg_confidence"] == pytest.approx((0.5 + 0.54 + 0.58 + 0.62 + 0.66) / 5)
    assert summary["patient"]["avg_confidence"] == pytest.approx((0.7 + 0.74 + 0.78 + 0.82 + 0.86) / 5)


def test_tag_citations_maps_known_normalized_entity() -> None:
    report = "Assessment notes chest pain with elevated blood pressure."
    entities = [
        Entity(type="symptom", text="Chest Pain", normalized="chest pain", attributes={"segment_id": "seg_0002"}),
        Entity(type="vital", text="BP", normalized="150/90", attributes={"segment_id": "seg_0003"}),
    ]
    segments = [
        Segment(start=0.0, end=1.0, speaker="medic", text="Patient has chest pain", confidence=0.9),
        Segment(start=1.0, end=2.0, speaker="medic", text="Blood pressure 150/90", confidence=0.9),
    ]

    citation_map = tag_citations(report, entities, segments)

    assert citation_map["chest pain"] == ["seg_0002"]
    assert "150/90" not in citation_map


def test_build_provenance_chain_contains_expected_shape() -> None:
    session = SessionContext.create(encounter_id="enc-context-002").model_copy(
        update={
            "report_draft": "Patient reports chest pain. Aspirin was administered in route.",
            "code_suggestions": [
                {
                    "code": "R07.9",
                    "type": "ICD",
                    "rationale": "Primary impression chest pain",
                    "evidence_segment_ids": ["seg_0001"],
                },
                {
                    "code": "J3490",
                    "type": "CPT",
                    "rationale": "Intervention aspirin administration",
                    "evidence_segment_ids": ["seg_0002"],
                },
            ],
            "extracted_terms": [
                Entity(
                    type="symptom",
                    text="chest pain",
                    normalized="chest pain",
                    attributes={"segment_id": "seg_0001"},
                ),
                Entity(
                    type="medication",
                    text="aspirin",
                    normalized="aspirin",
                    attributes={"segment_id": "seg_0002"},
                ),
            ],
        }
    )

    chain = build_provenance_chain(session)

    assert len(chain) == 2
    for entry in chain:
        assert set(entry.keys()) == {
            "code",
            "code_type",
            "report_excerpt",
            "segment_ids",
            "entities",
        }
        assert isinstance(entry["segment_ids"], list)
        assert isinstance(entry["entities"], list)


def test_session_size_and_trim_for_transport_returns_compressed_context() -> None:
    session = SessionContext.create(encounter_id="enc-context-003").model_copy(
        update={
            "report_draft": "A" * 1000,
            "clinical_reasoning": "B" * 200,
            "code_suggestions": [{"code": "R07.9", "type": "ICD", "evidence_segment_ids": ["seg_0001"]}],
            "pre_submission_flags": [{"field": "code", "severity": "warning", "issue": "check"}],
            "submission_status": "submitted",
        }
    )

    assert session.estimate_size_bytes() > 0

    trimmed = session.trim_for_transport(max_bytes=1)
    assert trimmed == compress_agent3_output(session)
    assert "transcript_raw" not in trimmed
