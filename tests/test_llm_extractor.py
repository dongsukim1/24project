from __future__ import annotations

from ems_pipeline.llm.extractor import adaptive_chunk
from ems_pipeline.models import Segment


def _seg(speaker: str = "spk0", text: str = "hello") -> Segment:
    return Segment(start=0.0, end=1.0, speaker=speaker, text=text, confidence=0.9)


def test_adaptive_chunk_short_transcript_single_chunk() -> None:
    segments = [_seg() for _ in range(10)]
    chunks = adaptive_chunk(segments, max_segments=15)
    assert len(chunks) == 1
    assert len(chunks[0]) == 10


def test_adaptive_chunk_exact_threshold_single_chunk() -> None:
    segments = [_seg() for _ in range(15)]
    chunks = adaptive_chunk(segments, max_segments=15)
    assert len(chunks) == 1
    assert len(chunks[0]) == 15


def test_adaptive_chunk_long_transcript_multiple_chunks() -> None:
    segments = [_seg(speaker=f"spk{i % 2}") for i in range(30)]
    chunks = adaptive_chunk(segments, max_segments=15)
    assert len(chunks) >= 2
    # All segments accounted for (minus overlaps)
    all_ids = set()
    for chunk in chunks:
        for seg in chunk:
            all_ids.add(id(seg))
    assert len(all_ids) >= 30


def test_adaptive_chunk_overlap_segment_included() -> None:
    segments = [_seg(speaker=f"spk{i % 2}") for i in range(25)]
    chunks = adaptive_chunk(segments, max_segments=10)
    assert len(chunks) >= 2
    # Last segment of chunk N should appear as first of chunk N+1
    for i in range(len(chunks) - 1):
        assert chunks[i][-1] is chunks[i + 1][0]


def test_adaptive_chunk_prefers_speaker_turn_boundary() -> None:
    # 16 segments: spk0 x8, then spk1 x8. With max_segments=15,
    # split should prefer the speaker turn at index 8 over mid-turn.
    segments = [_seg(speaker="spk0") for _ in range(8)] + [
        _seg(speaker="spk1") for _ in range(8)
    ]
    chunks = adaptive_chunk(segments, max_segments=15)
    assert len(chunks) == 2
    # First chunk should end at or near the speaker boundary
    assert all(s.speaker == "spk0" for s in chunks[0][:-1])  # overlap excluded


# --- merge_chunks tests ---

from ems_pipeline.llm.extractor import merge_chunks
from ems_pipeline.models import Entity


def _entity(
    entity_type: str = "SYMPTOM",
    text: str = "chest pain",
    segment_id: str = "seg_0001",
    confidence: float = 0.9,
) -> Entity:
    return Entity(
        type=entity_type,
        text=text,
        normalized=text,
        start=0.0,
        end=1.0,
        speaker="spk0",
        confidence=confidence,
        attributes={"segment_id": segment_id, "source": "llm"},
    )


def test_merge_chunks_single_chunk_passthrough() -> None:
    entities = [_entity(), _entity(text="SOB", entity_type="SYMPTOM")]
    result = merge_chunks([entities])
    assert len(result) == 2


def test_merge_chunks_dedup_same_segment_type_text() -> None:
    e1 = _entity(confidence=0.9)
    e2 = _entity(confidence=0.7)  # same segment, type, text — lower confidence
    result = merge_chunks([[e1], [e2]])
    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_merge_chunks_keeps_different_types() -> None:
    e1 = _entity(entity_type="SYMPTOM")
    e2 = _entity(entity_type="CONDITION", segment_id="seg_0001")
    result = merge_chunks([[e1], [e2]])
    assert len(result) == 2


def test_merge_chunks_keeps_different_segments() -> None:
    e1 = _entity(segment_id="seg_0001")
    e2 = _entity(segment_id="seg_0002")
    result = merge_chunks([[e1], [e2]])
    assert len(result) == 2


# --- build_extraction_prompt & ENTITY_TOOL_DEFINITION tests ---

from ems_pipeline.llm.extractor import build_extraction_prompt, ENTITY_TOOL_DEFINITION


def test_build_extraction_prompt_includes_segments() -> None:
    segments = [
        Segment(start=0.0, end=5.0, speaker="spk0", text="chest pain onset", confidence=0.9),
        Segment(start=5.0, end=10.0, speaker="spk1", text="SpO2 92%", confidence=0.95),
    ]
    segment_ids = ["seg_0000", "seg_0001"]
    prompt = build_extraction_prompt(segments, segment_ids)
    assert "[seg_0000]" in prompt
    assert "[seg_0001]" in prompt
    assert "(spk0)" in prompt
    assert "chest pain onset" in prompt


def test_build_extraction_prompt_no_timestamps_in_output() -> None:
    segments = [
        Segment(start=123.456, end=130.0, speaker="spk0", text="hello", confidence=0.9),
    ]
    prompt = build_extraction_prompt(segments, ["seg_0000"])
    assert "123.456" not in prompt


def test_entity_tool_definition_has_required_fields() -> None:
    schema = ENTITY_TOOL_DEFINITION
    assert schema["name"] == "record_entities"
    props = schema["input_schema"]["properties"]["entities"]["items"]["properties"]
    assert "type" in props
    assert "text" in props
    assert "segment_id" in props
    assert "confidence" in props
    assert "negated" in props
    assert "experiencer" in props
    assert "temporality" in props
