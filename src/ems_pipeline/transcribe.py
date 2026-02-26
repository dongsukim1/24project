"""Audio -> diarized transcript stage.

Current implementation:
  - Offline/local ASR via `ems_pipeline.speech.asr_whisper` (prefers faster-whisper).
  - Baseline diarization fallback (`ems_pipeline.speech.diarize`) which assigns a single speaker.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ems_pipeline.audio.filters import bandpass_filter
from ems_pipeline.audio.filters import denoise as denoise_filter
from ems_pipeline.audio.preprocess import TARGET_SAMPLE_RATE, chunk_audio, load_audio
from ems_pipeline.models import Segment, Transcript
from ems_pipeline.speech.asr_whisper import (
    ASRSegment,
    Word,
    asr_items_to_segments,
    transcribe_chunks,
)
from ems_pipeline.speech.diarize import SpeakerTurn, diarize


def _speaker_for_time(turns: list[SpeakerTurn], t: float) -> str:
    if not turns:
        return "spk0"
    for turn in turns:
        if turn.start <= t < turn.end:
            return turn.speaker_id
    return turns[-1].speaker_id


def _infer_asr_backend_name() -> str:
    try:
        import faster_whisper  # type: ignore  # noqa: F401

        return "faster-whisper"
    except Exception:
        pass
    try:
        import whisper  # type: ignore  # noqa: F401

        return "openai-whisper"
    except Exception:
        return "unknown"


def transcribe_audio(
    audio_path: str | Path,
    *,
    bandpass: bool = False,
    denoise: bool = False,
) -> Transcript:
    """Transcribe an input audio file into a diarized `Transcript`.

    Args:
        audio_path: Path to an input audio file (wav/mp3/etc).
        bandpass: If True, apply a telephone-like bandpass filter (~200–3400 Hz) before ASR.
        denoise: If True, apply lightweight noise reduction before ASR.

    Returns:
        Transcript: diarized segments with start/end timestamps and confidence.
    """

    wave = load_audio(audio_path)
    sr = TARGET_SAMPLE_RATE

    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1:
        raise ValueError("load_audio must return a 1D mono waveform.")

    duration_s = float(wave.shape[0]) / float(sr) if wave.size else 0.0
    if wave.size == 0:
        return Transcript(
            segments=[],
            metadata={
                "duration_s": duration_s,
                "sample_rate_hz": sr,
                "asr": {
                    "backend": _infer_asr_backend_name(),
                    "model": os.getenv("EMS_ASR_MODEL", "base"),
                    "language": "en",
                },
                "preprocess": {"bandpass": bool(bandpass), "denoise": bool(denoise)},
                "diarization": {"backend": "baseline/fallback"},
            },
        )

    if bandpass:
        wave = np.asarray(bandpass_filter(wave, sr), dtype=np.float32)
    if denoise:
        wave = np.asarray(denoise_filter(wave, sr), dtype=np.float32)

    turns = diarize(wave, sr)

    # Chunking: no overlap to keep timestamps consistent with `transcribe_chunks`'s offset logic.
    chunks = chunk_audio(wave, sr=sr, chunk_seconds=30.0, overlap_seconds=0.0)
    chunk_waves = [c for c, _ in chunks] if chunks else [wave]

    items = transcribe_chunks(chunk_waves, sr=sr, language="en")

    segments: list[Segment] = []
    word_buf: list[Word] = []
    word_speaker: str | None = None

    def flush_words() -> None:
        nonlocal word_buf, word_speaker
        if not word_buf:
            return
        segments.extend(asr_items_to_segments(word_buf, speaker=word_speaker or "spk0"))
        word_buf = []
        word_speaker = None

    for item in sorted(items, key=lambda x: (x.start, x.end)):
        if isinstance(item, Word):
            mid = (float(item.start) + float(item.end)) / 2.0
            spk = _speaker_for_time(turns, mid)
            if word_buf and spk != word_speaker:
                flush_words()
            word_speaker = spk
            word_buf.append(item)
            continue

        if isinstance(item, ASRSegment):
            flush_words()
            mid = (float(item.start) + float(item.end)) / 2.0
            spk = _speaker_for_time(turns, mid)
            segments.extend(asr_items_to_segments([item], speaker=spk))
            continue

        raise TypeError(f"Unexpected ASR item type: {type(item)!r}")

    flush_words()

    segments = sorted(segments, key=lambda s: (s.start, s.end))
    return Transcript(
        segments=segments,
        metadata={
            "duration_s": duration_s,
            "sample_rate_hz": sr,
            "asr": {
                "backend": _infer_asr_backend_name(),
                "model": os.getenv("EMS_ASR_MODEL", "base"),
                "language": "en",
            },
            "preprocess": {"bandpass": bool(bandpass), "denoise": bool(denoise)},
            "diarization": {"backend": "baseline/fallback"},
        },
    )
