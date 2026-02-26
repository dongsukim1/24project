from __future__ import annotations

from pathlib import Path

import numpy as np

from ems_pipeline.models import Transcript
from ems_pipeline.speech.asr_whisper import Word
from ems_pipeline.speech.diarize import SpeakerTurn
from ems_pipeline.transcribe import transcribe_audio


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
