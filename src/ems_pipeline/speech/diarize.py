"""Speaker diarization utilities.

This module exposes a small, dependency-light interface that the rest of the pipeline can
consume today, while leaving clear extension points for real diarization backends later
(e.g., pyannote.audio, NVIDIA NeMo).

The core contract:

- `diarize(wave, sr) -> list[SpeakerTurn]`

Where `wave` is a mono waveform (any sequence type with a meaningful `len()` or `shape`)
and `sr` is the sample rate in Hz.

Transcript builder consumption
-----------------------------

Downstream, a transcript builder typically combines:

1) ASR output with word-level timestamps, and
2) diarization turns (`SpeakerTurn`) produced here.

One simple approach is to assign each word to a speaker turn based on the word's midpoint,
then group consecutive words with the same speaker into transcript segments.

Example (pseudo-code):

```python
from ems_pipeline.models import Segment, Transcript
from ems_pipeline.speech.diarize import diarize

turns = diarize(wave, sr)  # -> [SpeakerTurn(...), ...]
words = asr_words(wave, sr)  # list[dict(word=..., start=..., end=..., conf=...)]

def speaker_for_time(t: float) -> str:
    for turn in turns:
        if turn.start <= t < turn.end:
            return turn.speaker_id
    return "unknown"

segments: list[Segment] = []
current: list[dict] = []
current_speaker: str | None = None
for w in words:
    spk = speaker_for_time((w["start"] + w["end"]) / 2)
    if current and spk != current_speaker:
        segments.append(
            Segment(
                start=current[0]["start"],
                end=current[-1]["end"],
                speaker=current_speaker or "unknown",
                text=" ".join(x["word"] for x in current),
                confidence=sum(x["conf"] for x in current) / len(current),
            )
        )
        current = []
    current_speaker = spk
    current.append(w)

if current:
    segments.append(
        Segment(
            start=current[0]["start"],
            end=current[-1]["end"],
            speaker=current_speaker or "unknown",
            text=" ".join(x["word"] for x in current),
            confidence=sum(x["conf"] for x in current) / len(current),
        )
    )

transcript = Transcript(segments=segments, metadata={"diarization": "baseline/fallback"})
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence


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


def diarize(wave: Sequence[float] | object, sr: int) -> list[SpeakerTurn]:
    """Return diarization speaker turns for the input audio.

    Baseline behavior (fallback):
        If no diarization backend is available/configured, return a single speaker turn
        spanning the entire audio.

    Extension points:
        Replace `_run_optional_backend()` with an implementation that returns a list of
        `SpeakerTurn` from a real diarization model (pyannote/NeMo/etc). Keep it optional
        (import inside the function) so this package stays light by default.
    """

    duration = audio_duration_seconds(wave, sr)
    turns = _run_optional_backend(wave, sr)
    if not turns:
        turns = [SpeakerTurn(start=0.0, end=duration, speaker_id="spk0", confidence=1.0)]
    return validate_speaker_turns(turns, duration=duration, mode="reject")


def audio_duration_seconds(wave: Sequence[float] | object, sr: int) -> float:
    """Compute audio duration in seconds from `wave` and sample rate."""

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
    mode: Literal["reject", "normalize"] = "reject",
    tolerance: float = 1e-6,
) -> list[SpeakerTurn]:
    """Validate (and optionally normalize) a sequence of speaker turns.

    Rules enforced:
    - monotonically non-decreasing time, i.e., no overlaps
    - `start >= 0`
    - if `duration` is provided, `end <= duration`

    Args:
        turns: Proposed turns (any order).
        duration: Optional audio duration (seconds) used for bounds checking/clamping.
        mode:
            - "reject": raise `ValueError` on any invalidity/overlap.
            - "normalize": clamp to [0, duration], sort by start time, and trim overlaps
              by moving a turn's start to the previous turn's end (dropping empty turns).
        tolerance: Float tolerance for overlap checks.
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
            if start < 0:
                raise ValueError(f"turn[{idx}].start must be >= 0 (got {start})")
            if duration is not None and end > duration + tolerance:
                raise ValueError(f"turn[{idx}].end must be <= duration (got {end} > {duration})")

        if end <= start + tolerance:
            if mode == "normalize":
                continue
            raise ValueError(f"turn[{idx}] has non-positive duration (start={start}, end={end})")

        if normalized:
            prev = normalized[-1]
            if start < prev.end - tolerance:
                if mode == "reject":
                    raise ValueError(
                        f"turn[{idx}] overlaps previous "
                        f"(prev_end={prev.end}, start={start}, end={end})"
                    )
                start = max(start, prev.end)
                if end <= start + tolerance:
                    continue

        normalized.append(
            SpeakerTurn(
                start=start,
                end=end,
                speaker_id=turn.speaker_id,
                confidence=turn.confidence,
            )
        )

    return normalized


def _run_optional_backend(wave: Sequence[float] | object, sr: int) -> list[SpeakerTurn] | None:
    """Optional diarization backend hook.

    Keep this function lightweight: do not add mandatory heavy dependencies.
    A real implementation might:
    - look for an env var or config flag selecting a backend
    - try importing a backend lazily
    - run diarization and translate outputs into `SpeakerTurn`
    """

    _ = (wave, sr)
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

