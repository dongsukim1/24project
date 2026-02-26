from __future__ import annotations

import pytest

from ems_pipeline.speech.asr_whisper import ASRSegment, Word, asr_items_to_segments


def test_asr_segment_adapter_cleans_text_and_applies_speaker():
    items = [
        ASRSegment(start=0.0, end=1.0, text=" hello ,  world ! ", confidence=0.7),
    ]
    segs = asr_items_to_segments(items, speaker="spk1")
    assert len(segs) == 1
    assert segs[0].speaker == "spk1"
    assert segs[0].text == "hello, world!"
    assert segs[0].start == pytest.approx(0.0)
    assert segs[0].end == pytest.approx(1.0)
    assert segs[0].confidence == pytest.approx(0.7)


def test_word_merging_splits_on_gaps():
    words = [
        Word(start=0.00, end=0.10, text="hello", confidence=0.9),
        Word(start=0.12, end=0.20, text="world", confidence=0.8),
        # Big silence gap triggers a new segment
        Word(start=2.00, end=2.10, text="next", confidence=0.7),
    ]
    segs = asr_items_to_segments(words, speaker="spk0", max_gap_s=0.6)
    assert [s.text for s in segs] == ["hello world", "next"]
    assert segs[0].start == pytest.approx(0.0)
    assert segs[0].end == pytest.approx(0.20)
    assert segs[1].start == pytest.approx(2.0)
    assert segs[1].end == pytest.approx(2.10)


def test_word_merging_respects_max_segment_duration():
    words = [
        Word(start=0.0, end=5.0, text="a", confidence=1.0),
        Word(start=5.05, end=10.0, text="b", confidence=0.5),
        Word(start=10.05, end=21.0, text="c", confidence=0.0),
    ]
    segs = asr_items_to_segments(words, speaker="spk0", max_gap_s=1.0, max_segment_s=12.0)
    assert [s.text for s in segs] == ["a b", "c"]


def test_word_merging_confidence_is_mean_of_available():
    words = [
        Word(start=0.0, end=0.1, text="hi", confidence=0.2),
        Word(start=0.1, end=0.2, text="there", confidence=None),
        Word(start=0.2, end=0.3, text="!", confidence=0.8),
    ]
    segs = asr_items_to_segments(words, speaker="spk0", max_gap_s=1.0)
    assert len(segs) == 1
    assert segs[0].confidence == pytest.approx((0.2 + 0.8) / 2.0)

