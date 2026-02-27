"""Audio -> diarized transcript stage.

Current implementation:
  - Offline/local ASR via `ems_pipeline.speech.asr_whisper` (prefers faster-whisper).
  - Optional multi-speaker diarization via `ems_pipeline.speech.diarize`; falls back to a
    single-speaker baseline when no backend is configured.
  - Speaker assignment uses maximum-overlap matching between ASR word intervals and
    diarization turns (falling back to nearest-turn midpoint lookup for out-of-range words).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

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
    """Assign speaker by midpoint lookup (kept for backward compatibility)."""
    if not turns:
        return "spk0"
    for turn in turns:
        if turn.start <= t < turn.end:
            return turn.speaker_id
    return turns[-1].speaker_id


def _speaker_for_interval(
    turns: list[SpeakerTurn] | Iterable[SpeakerTurn],
    start: float,
    end: float,
) -> str:
    """Assign speaker by maximum overlap with the word interval ``[start, end]``.

    Strategy:
    1. Compute the overlap between ``[start, end]`` and every diarization turn.
    2. Return the speaker_id of the turn with the greatest overlap.
    3. If no turn overlaps (e.g. the word timestamp is out-of-range), fall back to a
       midpoint lookup using :func:`_speaker_for_time`.

    Guard rails:
    - If *end* < *start* (malformed ASR output), they are swapped before comparison.
    - Turns may be any iterable; the list is materialised once internally.
    """
    turns_list: list[SpeakerTurn] = list(turns)
    if not turns_list:
        return "spk0"

    if end < start:
        start, end = end, start

    best_turn: SpeakerTurn | None = None
    best_overlap = 0.0
    for turn in turns_list:
        overlap = min(turn.end, end) - max(turn.start, start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = turn

    if best_turn is not None:
        return best_turn.speaker_id

    # No turn overlaps the word — use nearest-turn midpoint fallback
    return _speaker_for_time(turns_list, (start + end) / 2.0)


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
    """Transcribe an input audio file into a diarized :class:`~ems_pipeline.models.Transcript`.

    Args:
        audio_path: Path to an input audio file (wav/mp3/etc).
        bandpass:   If ``True``, apply a telephone-like bandpass filter (~200–3400 Hz)
                    before ASR.
        denoise:    If ``True``, apply lightweight noise reduction before ASR.

    Returns:
        :class:`~ems_pipeline.models.Transcript` with diarized segments, start/end
        timestamps, per-segment confidence, and comprehensive metadata (ASR backend,
        preprocessing flags, diarization backend and overlap policy).
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
                "diarization": {
                    "backend": "baseline/fallback",
                    "policy": "reject",
                    "num_speakers": 0,
                },
            },
        )

    if bandpass:
        wave = np.asarray(bandpass_filter(wave, sr), dtype=np.float32)
    if denoise:
        wave = np.asarray(denoise_filter(wave, sr), dtype=np.float32)

    diarize_result = diarize(wave, sr)
    # Materialise turns into a plain list; works whether diarize_result is a DiarizeResult
    # (production) or a plain list[SpeakerTurn] (monkeypatched in tests).
    turns: list[SpeakerTurn] = list(diarize_result)
    diarize_backend: str = getattr(diarize_result, "backend", "baseline/fallback")
    diarize_policy: str = getattr(diarize_result, "policy", "reject")

    # Chunking: no overlap to keep timestamps consistent with transcribe_chunks offset logic.
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
            spk = _speaker_for_interval(turns, float(item.start), float(item.end))
            if word_buf and spk != word_speaker:
                flush_words()
            word_speaker = spk
            word_buf.append(item)
            continue

        if isinstance(item, ASRSegment):
            flush_words()
            spk = _speaker_for_interval(turns, float(item.start), float(item.end))
            segments.extend(asr_items_to_segments([item], speaker=spk))
            continue

        raise TypeError(f"Unexpected ASR item type: {type(item)!r}")

    flush_words()

    segments = sorted(segments, key=lambda s: (s.start, s.end))
    num_speakers = len({t.speaker_id for t in turns})
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
            "diarization": {
                "backend": diarize_backend,
                "policy": diarize_policy,
                "num_speakers": num_speakers,
            },
        },
    )
