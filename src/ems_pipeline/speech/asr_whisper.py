"""Local/offline ASR via open-source Whisper (prefers faster-whisper).

This module intentionally avoids any network/API calls. If the requested model weights are not
available locally, it raises a clear error instructing the user to install/cache the model.
"""

from __future__ import annotations

import inspect
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ems_pipeline.models import Segment


@dataclass(frozen=True, slots=True)
class Word:
    """A word-level ASR token with timestamps (seconds from audio start)."""

    start: float
    end: float
    text: str
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ASRSegment:
    """A segment-level ASR item without diarization."""

    start: float
    end: float
    text: str
    confidence: float | None = None


class ASRModelNotFoundError(RuntimeError):
    """Raised when the ASR model weights are not available locally."""


_ALLOWED_MODELS: set[str] = {"base", "small", "medium", "large-v3"}


def _get_model_name() -> str:
    model = os.getenv("EMS_ASR_MODEL", "base").strip()
    if model not in _ALLOWED_MODELS:
        allowed = ", ".join(sorted(_ALLOWED_MODELS))
        raise ValueError(f"Invalid EMS_ASR_MODEL={model!r}. Allowed values: {allowed}.")
    return model


def _maybe_to_mono(samples):
    # Accept either shape (n,) or (n, channels) or (channels, n).
    try:
        import numpy as np  # type: ignore
    except Exception:
        return samples

    arr = np.asarray(samples)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        return arr.reshape(-1)
    # Heuristic: if second dim is small, treat as channels; else treat first as channels.
    if arr.shape[1] <= 8:
        return arr.mean(axis=1)
    if arr.shape[0] <= 8:
        return arr.mean(axis=0)
    return arr.reshape(-1)


def _resample_to_16k(samples, sr: int):
    if sr == 16000:
        return samples
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ValueError(
            f"Expected 16kHz audio (got {sr}Hz). Install numpy to enable resampling."
        ) from exc

    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return x
    duration = x.shape[0] / float(sr)
    new_len = max(1, int(round(duration * 16000.0)))
    old_idx = np.linspace(0.0, x.shape[0] - 1.0, num=x.shape[0], dtype=np.float32)
    new_idx = np.linspace(0.0, x.shape[0] - 1.0, num=new_len, dtype=np.float32)
    y = np.interp(new_idx, old_idx, x).astype(np.float32)
    return y


def _clean_joined_text(text: str) -> str:
    # Whisper-style tokens often include leading spaces; normalize without destroying punctuation.
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+([\)\]\}])", r"\1", text)
    return text


def _words_to_segments(
    words: Sequence[Word],
    *,
    speaker: str,
    max_gap_s: float = 0.6,
    max_segment_s: float = 20.0,
) -> list[Segment]:
    if not words:
        return []

    segments: list[Segment] = []
    buf: list[Word] = []
    buf_start = words[0].start

    def flush():
        nonlocal buf, buf_start
        if not buf:
            return
        start = buf_start
        end = max(w.end for w in buf)
        text = _clean_joined_text(" ".join(w.text.strip() for w in buf if w.text.strip()))
        confidences = [w.confidence for w in buf if w.confidence is not None]
        confidence = float(sum(confidences) / len(confidences)) if confidences else 1.0
        segments.append(
            Segment(
                start=float(start),
                end=float(end),
                speaker=speaker,
                text=text,
                confidence=confidence,
            )
        )
        buf = []

    for w in words:
        if not buf:
            buf = [w]
            buf_start = w.start
            continue

        gap = w.start - buf[-1].end
        would_duration = w.end - buf_start
        if gap > max_gap_s or would_duration > max_segment_s:
            flush()
            buf = [w]
            buf_start = w.start
        else:
            buf.append(w)

    flush()
    return segments


def asr_items_to_segments(
    items: Sequence[Word | ASRSegment],
    *,
    speaker: str = "spk0",
    max_gap_s: float = 0.6,
    max_segment_s: float = 20.0,
) -> list[Segment]:
    """Adapt ASR output (word- or segment-level) into pipeline `Segment`s."""

    if not items:
        return []

    words = [x for x in items if isinstance(x, Word)]
    segs = [x for x in items if isinstance(x, ASRSegment)]

    out: list[Segment] = []
    if segs:
        for s in segs:
            confidence = 1.0 if s.confidence is None else float(s.confidence)
            out.append(
                Segment(
                    start=float(s.start),
                    end=float(s.end),
                    speaker=speaker,
                    text=_clean_joined_text(s.text),
                    confidence=confidence,
                )
            )

    if words:
        out.extend(
            _words_to_segments(
                sorted(words, key=lambda w: (w.start, w.end)),
                speaker=speaker,
                max_gap_s=max_gap_s,
                max_segment_s=max_segment_s,
            )
        )

    return sorted(out, key=lambda s: (s.start, s.end))


class _FasterWhisperBackend:
    def __init__(self, model_name: str):
        from faster_whisper import WhisperModel  # type: ignore

        kwargs = {
            "device": "cpu",
            "compute_type": "int8",
        }
        sig = inspect.signature(WhisperModel)
        if "local_files_only" in sig.parameters:
            kwargs["local_files_only"] = True
        self._model = WhisperModel(model_name, **kwargs)

    def transcribe(self, audio_16k, *, language: str) -> list[Word | ASRSegment]:
        # We request word timestamps; if the backend can't provide them for any reason, we
        # gracefully fall back to segment-level timestamps.
        segments, info = self._model.transcribe(
            audio_16k,
            language=language,
            word_timestamps=True,
            vad_filter=False,
        )

        items: list[Word | ASRSegment] = []
        for seg in segments:
            words = getattr(seg, "words", None)
            if words:
                for w in words:
                    try:
                        prob = w.probability
                    except AttributeError:
                        prob = None
                    items.append(
                        Word(
                            start=float(w.start),
                            end=float(w.end),
                            text=str(w.word),
                            confidence=None if prob is None else float(prob),
                        )
                    )
            else:
                try:
                    text = seg.text
                except AttributeError:
                    text = ""
                items.append(
                    ASRSegment(
                        start=float(seg.start),
                        end=float(seg.end),
                        text=str(text),
                        confidence=None,
                    )
                )
        _ = info  # reserved for metadata
        return items


class _OpenAIWhisperBackend:
    def __init__(self, model_name: str):
        import whisper  # type: ignore

        self._whisper = whisper
        self._model = self._load_model_local_only(model_name)

    def _load_model_local_only(self, model_name: str):
        whisper = self._whisper
        model_dir = os.getenv("WHISPER_MODEL_DIR")
        if model_dir is None:
            model_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")

        url = getattr(whisper, "_MODELS", {}).get(model_name)
        if not url:
            raise ValueError(f"Unknown whisper model {model_name!r}.")
        filename = str(url).split("/")[-1]
        expected = os.path.join(model_dir, filename)
        if not os.path.exists(expected):
            raise ASRModelNotFoundError(
                "Whisper model weights are not available locally. "
                f"Expected: {expected}. "
                "Download/cache the model once (while online) or use faster-whisper "
                "with a local cache."
            )

        return whisper.load_model(model_name, download_root=model_dir)

    def transcribe(self, audio_16k, *, language: str) -> list[Word | ASRSegment]:
        # OpenAI Whisper's official Python package does not expose stable word timestamps without
        # extra alignment logic; we return segment-level timestamps only.
        result = self._model.transcribe(audio_16k, language=language, task="transcribe")
        items: list[Word | ASRSegment] = []
        for seg in result.get("segments", []) or []:
            items.append(
                ASRSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=str(seg.get("text", "")),
                    confidence=None,
                )
            )
        return items


_BACKEND = None


def _get_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    model_name = _get_model_name()

    try:
        _BACKEND = _FasterWhisperBackend(model_name)
        return _BACKEND
    except ImportError:
        pass
    except Exception as exc:
        # If faster-whisper is installed but the model isn't cached locally, surface a clearer
        # error.
        raise ASRModelNotFoundError(
            "faster-whisper is available but the requested model is not cached locally. "
            f"Set EMS_ASR_MODEL={model_name!r} and cache it once while online "
            "(or provide a local model path)."
        ) from exc

    try:
        _BACKEND = _OpenAIWhisperBackend(model_name)
        return _BACKEND
    except ImportError as exc:
        raise RuntimeError(
            "No Whisper backend is installed. Install either `faster-whisper` (recommended) "
            "or `openai-whisper` to enable offline transcription."
        ) from exc


def transcribe_chunks(
    chunks: Sequence[Sequence[float] | object],
    sr: int,
    language: str = "en",
) -> list[Word | ASRSegment]:
    """Transcribe audio chunks to word- or segment-level ASR items.

    Notes:
      - Chunks are assumed to be contiguous; timestamps are returned in seconds from the start of
        the *concatenated* audio.
      - Audio is resampled to 16kHz if numpy is available; otherwise non-16k audio raises.
    """

    if sr <= 0:
        raise ValueError(f"Invalid sample rate sr={sr}.")

    backend = _get_backend()

    out: list[Word | ASRSegment] = []
    offset_s = 0.0
    for chunk in chunks:
        if chunk is None:
            continue

        mono = _maybe_to_mono(chunk)
        try:
            mono_len = len(mono)  # type: ignore[arg-type]
        except Exception:
            mono_len = None
        audio_16k = _resample_to_16k(mono, sr)

        try:
            items = backend.transcribe(audio_16k, language=language)
        except ASRModelNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError("ASR transcription failed.") from exc

        for item in items:
            if isinstance(item, Word):
                out.append(
                    Word(
                        start=float(item.start + offset_s),
                        end=float(item.end + offset_s),
                        text=item.text,
                        confidence=item.confidence,
                    )
                )
            else:
                out.append(
                    ASRSegment(
                        start=float(item.start + offset_s),
                        end=float(item.end + offset_s),
                        text=item.text,
                        confidence=item.confidence,
                    )
                )

        if mono_len is None:
            mono_len = len(audio_16k)
        offset_s += float(mono_len) / float(sr)

    return out
