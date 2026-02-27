"""Speaker diarization utilities.

This module exposes a small, dependency-light interface that the rest of the pipeline can
consume today, with a real optional backend (pyannote.audio) and clear fallback behavior.

Supported backends (selected by ``EMS_DIARIZE_BACKEND`` env var):
    - ``"pyannote"`` : pyannote.audio speaker-diarization-3.1 pipeline.  Requires
                       ``pyannote.audio`` installed and ``EMS_PYANNOTE_AUTH_TOKEN`` set to
                       a valid HuggingFace token.  Falls back gracefully if unavailable.
    - unset/other    : Baseline single-speaker fallback (``spk0`` spanning the full audio).

Overlap policy (``EMS_DIARIZE_OVERLAP_POLICY`` env var):
    - ``"reject"``    : Raise ``ValueError`` on any overlap (strict; useful for curated data).
    - ``"normalize"`` : Trim overlaps by moving the later turn's start to the earlier turn's
                        end.  Empty turns are dropped.  **Default.**
    - ``"allow"``     : Pass turns through with overlaps intact (boundary bounds still
                        enforced).  Downstream code must handle overlapping turns explicitly.

The core contract::

    diarize(wave, sr) -> DiarizeResult

where ``wave`` is a mono waveform (any sequence type with a meaningful ``len()`` or
``shape``) and ``sr`` is the sample rate in Hz.

:class:`DiarizeResult` behaves like ``list[SpeakerTurn]`` for backward compatibility but
additionally exposes ``.backend`` and ``.policy`` metadata so callers (e.g.
``transcribe_audio``) can surface diarization information without a separate lookup.

Transcript builder consumption
-------------------------------

Downstream, a transcript builder typically combines:

1. ASR output with word-level timestamps, and
2. diarization turns (``SpeakerTurn``) produced here.

The recommended assignment strategy is **maximum overlap**: for each ASR word interval
``[word.start, word.end]``, pick the speaker turn with the greatest time overlap, falling
back to the nearest turn by midpoint if no turn overlaps::

    from ems_pipeline.speech.diarize import diarize

    result = diarize(wave, sr)
    # result behaves like list[SpeakerTurn]; also has result.backend, result.policy

    def speaker_for_interval(start: float, end: float) -> str:
        best, best_ov = None, 0.0
        for turn in result:
            ov = min(turn.end, end) - max(turn.start, start)
            if ov > best_ov:
                best, best_ov = turn, ov
        if best:
            return best.speaker_id
        mid = (start + end) / 2.0
        for turn in result:
            if turn.start <= mid < turn.end:
                return turn.speaker_id
        return result[-1].speaker_id if result else "unknown"
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Sequence


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A contiguous time span attributed to a single speaker."""

    start: float
    end: float
    speaker_id: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0 (got {self.start})")
        if self.end <= self.start:
            raise ValueError(f"end must be > start (got start={self.start}, end={self.end})")
        if not self.speaker_id:
            raise ValueError("speaker_id must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1] (got {self.confidence})")


class DiarizeResult:
    """Return value of :func:`diarize`, pairing speaker turns with execution metadata.

    Behaves like a ``list[SpeakerTurn]`` for backward compatibility — supports ``len()``,
    integer indexing, and iteration — while also exposing ``.backend`` and ``.policy``
    attributes so callers can surface diarization metadata without a separate lookup call.

    Attributes:
        backend: Name of the backend that produced the turns
                 (e.g. ``"pyannote"`` or ``"baseline/fallback"``).
        policy:  Overlap policy applied during validation
                 (``"reject"``, ``"normalize"``, or ``"allow"``).
    """

    __slots__ = ("_turns", "backend", "policy")

    def __init__(self, turns: list[SpeakerTurn], backend: str, policy: str) -> None:
        self._turns = turns
        self.backend = backend
        self.policy = policy

    @property
    def turns(self) -> list[SpeakerTurn]:
        return self._turns

    # ------------------------------------------------------------------
    # Sequence-like interface for backward compatibility
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[SpeakerTurn]:
        return iter(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def __getitem__(self, idx: int) -> SpeakerTurn:
        return self._turns[idx]

    def __repr__(self) -> str:
        return (
            f"DiarizeResult(turns={self._turns!r}, backend={self.backend!r},"
            f" policy={self.policy!r})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diarize(wave: Sequence[float] | object, sr: int) -> DiarizeResult:
    """Return diarization speaker turns for the input audio.

    Reads ``EMS_DIARIZE_BACKEND`` and ``EMS_DIARIZE_OVERLAP_POLICY`` from the environment.
    Falls back to a single full-duration ``spk0`` turn if no backend is configured or if
    the selected backend fails.

    Args:
        wave: Mono waveform — any sequence or array with a meaningful ``len()`` / ``shape``.
        sr:   Sample rate in Hz.

    Returns:
        :class:`DiarizeResult` — behaves like ``list[SpeakerTurn]`` and also carries
        ``.backend`` and ``.policy`` metadata.
    """

    duration = audio_duration_seconds(wave, sr)
    backend_name, raw_turns = _run_optional_backend(wave, sr)

    if not raw_turns:
        fallback = [SpeakerTurn(start=0.0, end=duration, speaker_id="spk0", confidence=1.0)]
        validated = validate_speaker_turns(fallback, duration=duration, mode="reject")
        return DiarizeResult(turns=validated, backend="baseline/fallback", policy="reject")

    policy = _get_overlap_policy()
    validated = validate_speaker_turns(raw_turns, duration=duration, mode=policy)
    return DiarizeResult(turns=validated, backend=backend_name, policy=policy)


def audio_duration_seconds(wave: Sequence[float] | object, sr: int) -> float:
    """Compute audio duration in seconds from *wave* and sample rate."""

    if sr <= 0:
        raise ValueError(f"sr must be > 0 (got {sr})")
    n_samples = _num_samples(wave)
    if n_samples < 0:
        raise ValueError("wave must have a non-negative number of samples")
    return n_samples / float(sr)


def validate_speaker_turns(
    turns: Iterable[SpeakerTurn],
    *,
    duration: float | None = None,
    mode: Literal["reject", "normalize", "allow"] = "reject",
    tolerance: float = 1e-6,
) -> list[SpeakerTurn]:
    """Validate (and optionally normalize) a sequence of speaker turns.

    Rules enforced in all modes:
        - ``start >= 0``
        - If ``duration`` is provided: ``end <= duration``
        - Turns with non-positive duration are rejected (``reject``) or dropped
          (``normalize``/``allow``).

    Overlap handling depends on ``mode``:
        - ``"reject"``   : raise :exc:`ValueError` on any invalidity or overlap.
        - ``"normalize"`` : clamp bounds to ``[0, duration]``, sort by start time, and trim
          overlaps by moving a turn's start to the previous turn's end; empty turns dropped.
        - ``"allow"``    : same bound enforcement as ``"reject"`` but overlapping turns are
          passed through unchanged.  Downstream code must handle them.

    Args:
        turns:     Proposed turns (any order).
        duration:  Optional audio duration (seconds) used for bounds checking/clamping.
        mode:      One of ``"reject"``, ``"normalize"``, or ``"allow"``.
        tolerance: Float tolerance for overlap and zero-duration checks.

    Returns:
        Validated (and potentially reordered/trimmed) list of :class:`SpeakerTurn`.
    """

    if duration is not None and duration < 0:
        raise ValueError(f"duration must be >= 0 (got {duration})")

    ordered = sorted(list(turns), key=lambda t: (t.start, t.end, t.speaker_id))
    if not ordered:
        return []

    normalized: list[SpeakerTurn] = []
    for idx, turn in enumerate(ordered):
        start = float(turn.start)
        end = float(turn.end)

        if mode == "normalize":
            start = max(0.0, start)
            if duration is not None:
                end = min(float(duration), end)
        else:
            # reject or allow: strict bounds check
            if start < 0:
                raise ValueError(f"turn[{idx}].start must be >= 0 (got {start})")
            if duration is not None and end > duration + tolerance:
                raise ValueError(
                    f"turn[{idx}].end must be <= duration (got {end} > {duration})"
                )

        if end <= start + tolerance:
            if mode == "normalize":
                continue  # drop empty turn after clamping
            raise ValueError(
                f"turn[{idx}] has non-positive duration (start={start}, end={end})"
            )

        if normalized:
            prev = normalized[-1]
            if start < prev.end - tolerance:
                if mode == "reject":
                    raise ValueError(
                        f"turn[{idx}] overlaps previous "
                        f"(prev_end={prev.end}, start={start}, end={end})"
                    )
                elif mode == "normalize":
                    start = max(start, prev.end)
                    if end <= start + tolerance:
                        continue  # trimmed away entirely
                # "allow": keep overlap intact, do not modify start

        normalized.append(
            SpeakerTurn(
                start=start,
                end=end,
                speaker_id=turn.speaker_id,
                confidence=turn.confidence,
            )
        )

    return normalized


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_overlap_policy() -> Literal["reject", "normalize", "allow"]:
    """Read the active overlap policy from ``EMS_DIARIZE_OVERLAP_POLICY``.

    Defaults to ``"normalize"`` if the env var is unset or contains an unrecognised value.
    """
    raw = os.getenv("EMS_DIARIZE_OVERLAP_POLICY", "normalize").lower().strip()
    if raw in ("reject", "normalize", "allow"):
        return raw  # type: ignore[return-value]
    return "normalize"


def _run_optional_backend(
    wave: Sequence[float] | object, sr: int
) -> tuple[str, list[SpeakerTurn] | None]:
    """Dispatch to the configured diarization backend.

    Returns ``(backend_name, turns)`` where *turns* is ``None`` when no backend is
    configured or available (triggering fallback in :func:`diarize`).

    Controlled by the ``EMS_DIARIZE_BACKEND`` environment variable.
    """
    backend = os.getenv("EMS_DIARIZE_BACKEND", "").lower().strip()
    if backend == "pyannote":
        turns = _run_pyannote_backend(wave, sr)
        if turns is not None:
            return "pyannote", turns
        # Backend configured but failed — fall through to baseline/fallback
    return "baseline/fallback", None


def _run_pyannote_backend(
    wave: Sequence[float] | object, sr: int
) -> list[SpeakerTurn] | None:
    """Try to run pyannote.audio speaker diarization.

    Requirements:
        - ``pyannote.audio`` installed (``pip install pyannote.audio``).
        - ``EMS_PYANNOTE_AUTH_TOKEN`` env var set to a valid HuggingFace token with
          access to ``pyannote/speaker-diarization-3.1``.

    Returns ``None`` on any failure (import error, missing token, runtime error) so the
    caller falls back gracefully to the single-speaker baseline.
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore[import]
    except ImportError:
        warnings.warn(
            "EMS_DIARIZE_BACKEND=pyannote but pyannote.audio is not installed. "
            "Falling back to single-speaker baseline. "
            "Install with: pip install pyannote.audio",
            RuntimeWarning,
            stacklevel=4,
        )
        return None

    auth_token = os.getenv("EMS_PYANNOTE_AUTH_TOKEN", "").strip()
    if not auth_token:
        warnings.warn(
            "EMS_DIARIZE_BACKEND=pyannote but EMS_PYANNOTE_AUTH_TOKEN is not set. "
            "Falling back to single-speaker baseline.",
            RuntimeWarning,
            stacklevel=4,
        )
        return None

    try:
        import numpy as np

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token,
        )

        wave_array = np.asarray(wave, dtype=np.float32)
        if wave_array.ndim == 1:
            wave_array = wave_array[np.newaxis, :]  # (1, n_samples) for pyannote

        diarization = pipeline({"waveform": wave_array, "sample_rate": sr})

        turns: list[SpeakerTurn] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            try:
                turns.append(
                    SpeakerTurn(
                        start=float(segment.start),
                        end=float(segment.end),
                        speaker_id=str(speaker),
                        confidence=1.0,
                    )
                )
            except ValueError:
                continue  # skip degenerate segments from the backend

        return turns if turns else None

    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"pyannote diarization failed ({exc!r}). "
            "Falling back to single-speaker baseline.",
            RuntimeWarning,
            stacklevel=4,
        )
        return None


def _num_samples(wave: Sequence[float] | object) -> int:
    """Best-effort extraction of sample count from common waveform containers."""

    shape = getattr(wave, "shape", None)
    if isinstance(shape, tuple) and shape:
        # Common conventions:
        # - numpy/torch: (n_samples,) or (n_channels, n_samples)
        # - soundfile: similar array shapes
        return int(shape[-1])

    return len(wave)  # type: ignore[arg-type]
