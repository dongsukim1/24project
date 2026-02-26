from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from ems_pipeline.models import Entity, Transcript

EventType = Literal[
    "CALL_RECEIVED",
    "DISPATCHED",
    "ARRIVED_SCENE",
    "PATIENT_CONTACT",
    "INTERVENTION",
    "TRANSPORT_BEGIN",
    "ARRIVE_ED",
    "DISPOSITION",
]


class Event(TypedDict):
    type: EventType
    start: float
    end: float
    speaker: str | None
    evidence_segment_ids: list[str]
    payload: dict[str, Any]


_SEGMENT_TIME_RE = re.compile(
    r"(?<!\d)"  # don't start mid-number
    r"(?P<h>\d{1,2}):(?P<m>\d{2})"  # hh:mm or mm:ss
    r"(?::(?P<s>\d{2}))?"  # optional :ss
    r"(?!\d)"  # don't end mid-number
    r"\s*(?P<ampm>am|pm)?\b",
    flags=re.IGNORECASE,
)

_TYPE_PATTERNS: list[tuple[EventType, re.Pattern[str]]] = [
    ("CALL_RECEIVED", re.compile(r"\b(911|call (?:received|came in)|tone(?:d)? out)\b", re.I)),
    ("DISPATCHED", re.compile(r"\b(dispatch(?:ed)?|responding|en route|enroute|assigned)\b", re.I)),
    ("ARRIVED_SCENE", re.compile(r"\b(arrived|on scene|onscene)\b", re.I)),
    (
        "PATIENT_CONTACT",
        re.compile(r"\b(patient contact|pt contact|with patient|contact made)\b", re.I),
    ),
    (
        "TRANSPORT_BEGIN",
        re.compile(
            r"\b(transport(?:ing)?|depart(?:ing)? scene|en route to (?:ed|er|hospital))\b",
            re.I,
        ),
    ),
    ("ARRIVE_ED", re.compile(r"\b(arrived (?:at )?(?:ed|er|hospital)|at (?:ed|er))\b", re.I)),
    (
        "DISPOSITION",
        re.compile(r"\b(transferred care|turn(?:ed)? over|refus(?:al|ed)|ama|pronounced)\b", re.I),
    ),
]


@dataclass(frozen=True)
class _ExplicitTime:
    kind: Literal["relative", "absolute"]
    seconds: float
    raw: str


def _parse_explicit_time(text: str) -> _ExplicitTime | None:
    """Parse an explicit timestamp mention from free text.

    Supported:
    - Relative timecode: mm:ss or hh:mm:ss (interpreted as seconds from start)
    - Absolute time-of-day: hh:mm with optional am/pm, or 24h-like hh:mm where hh >= 13
    """

    match = _SEGMENT_TIME_RE.search(text)
    if not match:
        return None

    h = int(match.group("h"))
    m = int(match.group("m"))
    s_group = match.group("s")
    ampm = match.group("ampm")
    raw = match.group(0).strip()

    if s_group is not None:
        s = int(s_group)
        return _ExplicitTime(kind="relative", seconds=h * 3600 + m * 60 + s, raw=raw)

    # Two-field times are ambiguous (mm:ss vs hh:mm). Prefer time-of-day if clearly marked.
    if ampm:
        ampm_norm = ampm.lower()
        hour = h % 12
        if ampm_norm == "pm":
            hour += 12
        return _ExplicitTime(kind="absolute", seconds=hour * 3600 + m * 60, raw=raw)

    if 13 <= h <= 23:
        return _ExplicitTime(kind="absolute", seconds=h * 3600 + m * 60, raw=raw)

    # Default: treat as mm:ss timecode.
    return _ExplicitTime(kind="relative", seconds=h * 60 + m, raw=raw)


def _segment_ids(transcript: Transcript) -> list[str]:
    seg_id_map = transcript.metadata.get("segment_id_map")
    if isinstance(seg_id_map, dict) and seg_id_map:
        inv: dict[int, str] = {}
        for seg_id, idx in seg_id_map.items():
            if isinstance(seg_id, str) and isinstance(idx, int):
                inv[idx] = seg_id
        return [inv.get(i, f"seg_{i:04d}") for i in range(len(transcript.segments))]
    return [f"seg_{i:04d}" for i in range(len(transcript.segments))]


def _find_evidence_segments(transcript: Transcript, entity: Entity) -> list[int]:
    # Prefer timestamp overlap.
    if entity.start is not None or entity.end is not None:
        start = entity.start if entity.start is not None else entity.end
        end = entity.end if entity.end is not None else entity.start
        if start is not None and end is not None and end < start:
            start, end = end, start
        if start is not None and end is None:
            end = start
        if start is None or end is None:
            return []
        overlaps: list[int] = []
        for i, seg in enumerate(transcript.segments):
            if seg.end >= start and seg.start <= end:
                overlaps.append(i)
        if overlaps:
            return overlaps

    # Fallback: fuzzy text match.
    needle = entity.text.strip().lower()
    if not needle:
        return []
    hits: list[int] = []
    for i, seg in enumerate(transcript.segments):
        if needle in seg.text.lower():
            hits.append(i)
    return hits


def build_events(transcript: Transcript, entities: Iterable[Entity]) -> list[Event]:
    """Build a chronological incident timeline from transcript + extracted entities."""

    segment_ids = _segment_ids(transcript)
    segment_explicit: list[_ExplicitTime | None] = [
        _parse_explicit_time(seg.text) for seg in transcript.segments
    ]
    absolute_anchor: tuple[float, float] | None = None  # (absolute_seconds, segment_start_seconds)
    for i, explicit in enumerate(segment_explicit):
        if explicit and explicit.kind == "absolute":
            absolute_anchor = (explicit.seconds, transcript.segments[i].start)
            break

    def to_timeline_seconds(segment_index: int, at_seconds: float | None = None) -> float:
        seg = transcript.segments[segment_index]
        base = seg.start if at_seconds is None else at_seconds
        if absolute_anchor is None:
            explicit = segment_explicit[segment_index]
            if explicit and explicit.kind == "relative":
                return float(explicit.seconds)
            return float(base)

        absolute_at, seg_start = absolute_anchor
        mapped = absolute_at + (base - seg_start)
        explicit = segment_explicit[segment_index]
        if explicit and explicit.kind == "absolute":
            return float(explicit.seconds)
        return float(mapped)

    def segment_event_bounds(segment_index: int) -> tuple[float, float]:
        seg = transcript.segments[segment_index]
        start = to_timeline_seconds(segment_index)
        # Preserve duration when projecting into absolute mode.
        end = start + float(seg.end - seg.start)
        return start, end

    events: list[Event] = []
    singleton_best: dict[EventType, int] = {}

    def add_event(event: Event, singleton: bool) -> None:
        if not singleton:
            events.append(event)
            return
        t = event["type"]
        if t not in singleton_best:
            singleton_best[t] = len(events)
            events.append(event)
            return
        existing = events[singleton_best[t]]
        if event["start"] < existing["start"]:
            events[singleton_best[t]] = event

    # 1) Segment-triggered operational events.
    for i, seg in enumerate(transcript.segments):
        for event_type, pattern in _TYPE_PATTERNS:
            if not pattern.search(seg.text):
                continue
            start, end = segment_event_bounds(i)
            add_event(
                Event(
                    type=event_type,
                    start=start,
                    end=end,
                    speaker=seg.speaker,
                    evidence_segment_ids=[segment_ids[i]],
                    payload={
                        "source": "segment",
                        "text": seg.text,
                        "explicit_time": segment_explicit[i].raw if segment_explicit[i] else None,
                    },
                ),
                singleton=True,
            )

    # 2) Entity-triggered patient contact + interventions.
    for ent in entities:
        ent_type = (ent.type or "").strip().lower()
        evidence_idxs = _find_evidence_segments(transcript, ent)
        evidence_ids = [segment_ids[j] for j in evidence_idxs] if evidence_idxs else []
        speaker = ent.speaker
        if speaker is None and evidence_idxs:
            speaker = transcript.segments[evidence_idxs[0]].speaker

        # Establish patient contact when symptoms are first mentioned.
        if ent_type in {"symptom", "chief_complaint", "complaint"}:
            if evidence_idxs:
                i = evidence_idxs[0]
                start = (
                    to_timeline_seconds(i, at_seconds=ent.start)
                    if ent.start is not None
                    else to_timeline_seconds(i)
                )
                end = start
                add_event(
                    Event(
                        type="PATIENT_CONTACT",
                        start=float(start),
                        end=float(end),
                        speaker=speaker,
                        evidence_segment_ids=evidence_ids or [segment_ids[i]],
                        payload={
                            "source": "entity",
                            "entity_type": ent.type,
                            "text": ent.text,
                            "normalized": ent.normalized,
                        },
                    ),
                    singleton=True,
                )
            continue

        if ent_type in {"intervention", "procedure", "medication", "treatment"}:
            if evidence_idxs:
                i = evidence_idxs[0]
                start = (
                    to_timeline_seconds(i, at_seconds=ent.start)
                    if ent.start is not None
                    else to_timeline_seconds(i)
                )
                end = start
            elif ent.start is not None:
                # No evidence segment, but a timestamp exists: map via anchor if available.
                if absolute_anchor is None:
                    start = float(ent.start)
                else:
                    absolute_at, seg_start = absolute_anchor
                    start = float(absolute_at + (ent.start - seg_start))
                end = start
            else:
                continue

            add_event(
                Event(
                    type="INTERVENTION",
                    start=float(start),
                    end=float(end),
                    speaker=speaker,
                    evidence_segment_ids=evidence_ids,
                    payload={
                        "source": "entity",
                        "entity_type": ent.type,
                        "text": ent.text,
                        "normalized": ent.normalized,
                        "attributes": cast(dict[str, Any], ent.attributes),
                    },
                ),
                singleton=False,
            )

    events.sort(key=lambda e: (e["start"], e["end"], e["type"]))
    return events
