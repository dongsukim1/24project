from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ems_pipeline.models import Transcript
from ems_pipeline.speech.asr_whisper import Word
from ems_pipeline.speech.diarize import DiarizeResult, SpeakerTurn
from ems_pipeline.transcribe import _speaker_for_interval, _speaker_for_time, transcribe_audio


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_transcribe_audio_assigns_speakers_and_merges_words(
    monkeypatch: object, tmp_path: Path
) -> None:
    # Avoid any real ASR/model dependency.
    wave = np.zeros((2 * 16_000,), dtype=np.float32)

    def fake_load_audio(_: str | Path) -> np.ndarray:
        return wave

    def fake_diarize(_: np.ndarray, sr: int) -> list[SpeakerTurn]:
        assert sr == 16_000
        return [
            SpeakerTurn(start=0.0, end=1.0, speaker_id="spkA"),
            SpeakerTurn(start=1.0, end=2.0, speaker_id="spkB"),
        ]

    def fake_transcribe_chunks(
        chunks: list[np.ndarray], sr: int, language: str = "en"
    ) -> list[Word]:
        assert sr == 16_000
        assert language == "en"
        assert sum(c.shape[0] for c in chunks) == wave.shape[0]
        return [
            Word(start=0.00, end=0.10, text="hello", confidence=0.9),
            Word(start=0.10, end=0.20, text="world", confidence=0.8),
            Word(start=1.10, end=1.20, text="next", confidence=0.7),
        ]

    import ems_pipeline.transcribe as t

    monkeypatch.setattr(t, "load_audio", fake_load_audio)
    monkeypatch.setattr(t, "diarize", fake_diarize)
    monkeypatch.setattr(t, "transcribe_chunks", fake_transcribe_chunks)

    audio_path = tmp_path / "dummy.wav"
    audio_path.write_bytes(b"not a real wav")  # should not be read due to monkeypatch

    transcript = transcribe_audio(audio_path)
    assert isinstance(transcript, Transcript)

    assert [(s.speaker, s.text) for s in transcript.segments] == [
        ("spkA", "hello world"),
        ("spkB", "next"),
    ]


def test_transcribe_audio_applies_filters(monkeypatch: object, tmp_path: Path) -> None:
    wave = np.zeros((16_000,), dtype=np.float32)

    calls: dict[str, int] = {"bandpass": 0, "denoise": 0}

    def fake_load_audio(_: str | Path) -> np.ndarray:
        return wave

    def fake_bandpass(w: np.ndarray, sr: int) -> list[float]:
        assert sr == 16_000
        calls["bandpass"] += 1
        return list(w)

    def fake_denoise(w: np.ndarray, sr: int) -> list[float]:
        assert sr == 16_000
        calls["denoise"] += 1
        return list(w)

    def fake_transcribe_chunks(
        chunks: list[np.ndarray], sr: int, language: str = "en"
    ) -> list[Word]:
        return []

    import ems_pipeline.transcribe as t

    monkeypatch.setattr(t, "load_audio", fake_load_audio)
    monkeypatch.setattr(t, "bandpass_filter", fake_bandpass)
    monkeypatch.setattr(t, "denoise_filter", fake_denoise)
    monkeypatch.setattr(t, "transcribe_chunks", fake_transcribe_chunks)

    audio_path = tmp_path / "dummy.wav"
    audio_path.write_bytes(b"x")

    _ = transcribe_audio(audio_path, bandpass=True, denoise=True)
    assert calls == {"bandpass": 1, "denoise": 1}


# ---------------------------------------------------------------------------
# _speaker_for_interval unit tests
# ---------------------------------------------------------------------------


def _turns(*specs: tuple[float, float, str]) -> list[SpeakerTurn]:
    return [SpeakerTurn(start=s, end=e, speaker_id=sp) for s, e, sp in specs]


def test_speaker_for_interval_clear_overlap() -> None:
    """Word entirely inside one turn → that speaker."""
    turns = _turns((0.0, 1.0, "A"), (1.0, 2.0, "B"))
    assert _speaker_for_interval(turns, 0.2, 0.4) == "A"
    assert _speaker_for_interval(turns, 1.1, 1.3) == "B"


def test_speaker_for_interval_max_overlap_wins() -> None:
    """Word straddling two turns → speaker with the larger overlap wins."""
    # Word [0.8, 1.2]: overlap with A (0-1) = 0.2, overlap with B (1-2) = 0.2 — tie → A (first)
    # Use asymmetric case to make the winner clear.
    turns = _turns((0.0, 1.0, "A"), (1.0, 2.0, "B"))
    # Word [0.5, 1.2]: overlap A = 0.5, overlap B = 0.2 → A wins
    assert _speaker_for_interval(turns, 0.5, 1.2) == "A"
    # Word [0.9, 1.6]: overlap A = 0.1, overlap B = 0.6 → B wins
    assert _speaker_for_interval(turns, 0.9, 1.6) == "B"


def test_speaker_for_interval_no_overlap_midpoint_fallback() -> None:
    """If word interval has no overlap with any turn, fall back to midpoint lookup."""
    # Turn only covers 0-1; word at 1.5-2.0 has no overlap
    turns = _turns((0.0, 1.0, "A"))
    # Midpoint = 1.75, no turn covers it → falls back to last turn (A)
    assert _speaker_for_interval(turns, 1.5, 2.0) == "A"


def test_speaker_for_interval_empty_turns() -> None:
    """Empty turn list returns 'spk0'."""
    assert _speaker_for_interval([], 0.0, 1.0) == "spk0"


def test_speaker_for_interval_swapped_timestamps() -> None:
    """Reversed start/end (malformed ASR) is handled gracefully."""
    turns = _turns((0.0, 1.0, "A"), (1.0, 2.0, "B"))
    # start=1.3, end=1.1 (reversed) → swapped to [1.1, 1.3] → B
    assert _speaker_for_interval(turns, 1.3, 1.1) == "B"


def test_speaker_for_interval_exact_boundary() -> None:
    """Word starting exactly at a turn boundary is assigned to the later turn."""
    turns = _turns((0.0, 1.0, "A"), (1.0, 2.0, "B"))
    # Word [1.0, 1.5]: overlap A = 0, overlap B = 0.5 → B
    assert _speaker_for_interval(turns, 1.0, 1.5) == "B"


def test_speaker_for_interval_word_before_all_turns() -> None:
    """Word entirely before all turns falls back gracefully."""
    turns = _turns((1.0, 2.0, "A"))
    # Word [0.0, 0.5]: no overlap → midpoint 0.25, no turn covers it → last turn's speaker
    result = _speaker_for_interval(turns, 0.0, 0.5)
    assert result == "A"  # _speaker_for_time returns turns[-1].speaker_id


# ---------------------------------------------------------------------------
# _speaker_for_time backward-compat unit tests
# ---------------------------------------------------------------------------


def test_speaker_for_time_finds_correct_turn() -> None:
    turns = _turns((0.0, 1.0, "A"), (1.0, 2.0, "B"))
    assert _speaker_for_time(turns, 0.5) == "A"
    assert _speaker_for_time(turns, 1.5) == "B"


def test_speaker_for_time_empty_returns_spk0() -> None:
    assert _speaker_for_time([], 0.5) == "spk0"


def test_speaker_for_time_out_of_range_returns_last() -> None:
    turns = _turns((0.0, 1.0, "A"), (1.0, 2.0, "B"))
    assert _speaker_for_time(turns, 99.0) == "B"


# ---------------------------------------------------------------------------
# Metadata correctness
# ---------------------------------------------------------------------------


def test_transcribe_audio_metadata_keys(monkeypatch: object, tmp_path: Path) -> None:
    """Transcript metadata must contain diarization.backend, .policy, .num_speakers."""
    wave = np.zeros((16_000,), dtype=np.float32)

    def fake_load_audio(_: str | Path) -> np.ndarray:
        return wave

    def fake_diarize(_: np.ndarray, sr: int) -> DiarizeResult:
        turns = [
            SpeakerTurn(start=0.0, end=0.5, speaker_id="X"),
            SpeakerTurn(start=0.5, end=1.0, speaker_id="Y"),
        ]
        return DiarizeResult(turns=turns, backend="pyannote", policy="normalize")

    def fake_transcribe_chunks(chunks, sr, language="en") -> list[Word]:
        return []

    import ems_pipeline.transcribe as t

    monkeypatch.setattr(t, "load_audio", fake_load_audio)
    monkeypatch.setattr(t, "diarize", fake_diarize)
    monkeypatch.setattr(t, "transcribe_chunks", fake_transcribe_chunks)

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"x")

    transcript = transcribe_audio(audio_path)
    diarize_meta = transcript.metadata["diarization"]

    assert diarize_meta["backend"] == "pyannote"
    assert diarize_meta["policy"] == "normalize"
    assert diarize_meta["num_speakers"] == 2


def test_transcribe_audio_metadata_fallback_backend(monkeypatch: object, tmp_path: Path) -> None:
    """When a plain list is returned (monkeypatched), metadata defaults to baseline/fallback."""
    wave = np.zeros((16_000,), dtype=np.float32)

    def fake_load_audio(_: str | Path) -> np.ndarray:
        return wave

    def fake_diarize(_: np.ndarray, sr: int) -> list[SpeakerTurn]:
        # plain list, no .backend or .policy attributes
        return [SpeakerTurn(start=0.0, end=1.0, speaker_id="spk0")]

    def fake_transcribe_chunks(chunks, sr, language="en") -> list[Word]:
        return []

    import ems_pipeline.transcribe as t

    monkeypatch.setattr(t, "load_audio", fake_load_audio)
    monkeypatch.setattr(t, "diarize", fake_diarize)
    monkeypatch.setattr(t, "transcribe_chunks", fake_transcribe_chunks)

    audio_path = tmp_path / "b.wav"
    audio_path.write_bytes(b"x")

    transcript = transcribe_audio(audio_path)
    diarize_meta = transcript.metadata["diarization"]

    assert diarize_meta["backend"] == "baseline/fallback"
    assert diarize_meta["policy"] == "reject"
    assert diarize_meta["num_speakers"] == 1


def test_transcribe_audio_metadata_num_speakers_single(
    monkeypatch: object, tmp_path: Path
) -> None:
    """Single-speaker fallback sets num_speakers=1."""
    wave = np.zeros((8_000,), dtype=np.float32)

    def fake_load_audio(_: str | Path) -> np.ndarray:
        return wave

    def fake_diarize(_: np.ndarray, sr: int) -> DiarizeResult:
        turns = [SpeakerTurn(start=0.0, end=0.5, speaker_id="spk0")]
        return DiarizeResult(turns=turns, backend="baseline/fallback", policy="reject")

    def fake_transcribe_chunks(chunks, sr, language="en") -> list[Word]:
        return []

    import ems_pipeline.transcribe as t

    monkeypatch.setattr(t, "load_audio", fake_load_audio)
    monkeypatch.setattr(t, "diarize", fake_diarize)
    monkeypatch.setattr(t, "transcribe_chunks", fake_transcribe_chunks)

    audio_path = tmp_path / "c.wav"
    audio_path.write_bytes(b"x")

    transcript = transcribe_audio(audio_path)
    assert transcript.metadata["diarization"]["num_speakers"] == 1


# ---------------------------------------------------------------------------
# Out-of-range ASR word timestamps
# ---------------------------------------------------------------------------


def test_transcribe_audio_out_of_range_word_timestamps(
    monkeypatch: object, tmp_path: Path
) -> None:
    """Words whose timestamps fall outside all diarization turns are assigned gracefully."""
    wave = np.zeros((2 * 16_000,), dtype=np.float32)

    def fake_load_audio(_: str | Path) -> np.ndarray:
        return wave

    def fake_diarize(_: np.ndarray, sr: int) -> list[SpeakerTurn]:
        # Only covers 0-1 s; the word at 1.5-1.8 s is out of range
        return [SpeakerTurn(start=0.0, end=1.0, speaker_id="spkA")]

    def fake_transcribe_chunks(chunks, sr, language="en") -> list[Word]:
        return [
            Word(start=0.2, end=0.4, text="in", confidence=0.9),
            Word(start=1.5, end=1.8, text="out", confidence=0.8),  # beyond turn end
        ]

    import ems_pipeline.transcribe as t

    monkeypatch.setattr(t, "load_audio", fake_load_audio)
    monkeypatch.setattr(t, "diarize", fake_diarize)
    monkeypatch.setattr(t, "transcribe_chunks", fake_transcribe_chunks)

    audio_path = tmp_path / "d.wav"
    audio_path.write_bytes(b"x")

    # Should not raise; out-of-range words are assigned to the nearest available speaker
    transcript = transcribe_audio(audio_path)
    assert isinstance(transcript, Transcript)
    speakers = {s.speaker for s in transcript.segments}
    # All segments should be assigned to a non-empty speaker string
    assert all(sp != "" for sp in speakers)
