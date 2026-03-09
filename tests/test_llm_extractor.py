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
