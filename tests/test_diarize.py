from __future__ import annotations

import pytest

from ems_pipeline.speech.diarize import SpeakerTurn, diarize, validate_speaker_turns


def test_diarize_fallback_returns_single_turn_with_correct_duration() -> None:
    sr = 16_000
    wave = [0.0] * 40_000  # 2.5s

    turns = diarize(wave, sr)
    assert len(turns) == 1
    assert turns[0].start == 0.0
    assert turns[0].end == 2.5
    assert turns[0].speaker_id == "spk0"
    assert turns[0].confidence == 1.0


def test_validate_speaker_turns_rejects_overlaps() -> None:
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker_id="spk0", confidence=0.9),
        SpeakerTurn(start=1.5, end=3.0, speaker_id="spk1", confidence=0.9),
    ]

    with pytest.raises(ValueError, match="overlaps"):
        validate_speaker_turns(turns, duration=3.0, mode="reject")


def test_validate_speaker_turns_normalizes_overlaps() -> None:
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker_id="spk0", confidence=0.9),
        SpeakerTurn(start=1.5, end=3.0, speaker_id="spk1", confidence=0.9),
    ]

    normalized = validate_speaker_turns(turns, duration=3.0, mode="normalize")
    assert [(t.start, t.end, t.speaker_id) for t in normalized] == [
        (0.0, 2.0, "spk0"),
        (2.0, 3.0, "spk1"),
    ]

