from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from ems_pipeline.models import Entity, Transcript


@dataclass(frozen=True)
class _MatchWindow:
    text: str
    start: int
    end: int


_NEGATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdenies\b", flags=re.IGNORECASE),
    re.compile(r"\bno\b", flags=re.IGNORECASE),
    re.compile(r"\bnegative\s+for\b", flags=re.IGNORECASE),
    re.compile(r"\bwithout\b", flags=re.IGNORECASE),
)

_UNCERTAINTY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bpossible\b", flags=re.IGNORECASE),
    re.compile(r"\bmaybe\b", flags=re.IGNORECASE),
    re.compile(r"\bsuspected\b", flags=re.IGNORECASE),
    re.compile(r"\bquestion\s+of\b", flags=re.IGNORECASE),
)

_CONTRAST_BOUNDARY = re.compile(r"\b(?:but|however|though|although|yet)\b", flags=re.IGNORECASE)

_TEMPORALITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:started|begin|began|onset)\s+(?:about\s+)?(?P<t>\d+\s+(?:min(?:ute)?s?|hr(?:s)?|hour(?:s)?|day(?:s)?|week(?:s)?)\s+ago)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<t>(?:\d+)\s+(?:min(?:ute)?s?|hr(?:s)?|hour(?:s)?|day(?:s)?|week(?:s)?)\s+ago)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<t>since\s+(?:today|yesterday|last\s+night|this\s+morning|this\s+afternoon|this\s+evening))\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<t>for\s+\d+\s+(?:min(?:ute)?s?|hr(?:s)?|hour(?:s)?|day(?:s)?|week(?:s)?))\b",
        flags=re.IGNORECASE,
    ),
)

_BYSTANDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcaller\s+says\b", flags=re.IGNORECASE),
    re.compile(r"\bmy\s+(?:father|mother|wife|husband|son|daughter)\b", flags=re.IGNORECASE),
)

_PATIENT_PATTERN = re.compile(r"\bpatient\b", flags=re.IGNORECASE)


def _transcript_to_text(transcript: str | Transcript) -> str:
    if isinstance(transcript, str):
        return transcript
    return " ".join(s.text for s in transcript.segments)


def _find_entity_window(transcript_text: str, entity: Entity) -> _MatchWindow | None:
    start = entity.attributes.get("char_start")
    end = entity.attributes.get("char_end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(transcript_text):
        return _MatchWindow(transcript_text, start, end)

    needles: list[str] = []
    if entity.text:
        needles.append(entity.text)
    if entity.normalized and entity.normalized not in needles:
        needles.append(entity.normalized)

    lower = transcript_text.lower()
    for needle in needles:
        n = needle.strip().lower()
        if not n:
            continue
        idx = lower.find(n)
        if idx != -1:
            return _MatchWindow(transcript_text, idx, idx + len(n))
    return None


def _sentence_bounds(text: str, i: int) -> tuple[int, int]:
    left = max(text.rfind(".", 0, i), text.rfind("?", 0, i), text.rfind("!", 0, i))
    left = 0 if left == -1 else left + 1
    right_candidates = [p for p in (text.find(".", i), text.find("?", i), text.find("!", i)) if p != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return left, right


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    # Start from sentence bounds, then break on contrast terms like "but".
    sent_l, sent_r = _sentence_bounds(text, start)
    sent = text[sent_l:sent_r]
    rel_start = start - sent_l
    rel_end = end - sent_l

    boundaries: list[int] = [0, len(sent)]
    for m in _CONTRAST_BOUNDARY.finditer(sent):
        boundaries.append(m.start())
        boundaries.append(m.end())
    boundaries = sorted(set(b for b in boundaries if 0 <= b <= len(sent)))

    clause_l = 0
    clause_r = len(sent)
    for a, b in zip(boundaries, boundaries[1:], strict=False):
        if a <= rel_start <= b:
            clause_l, clause_r = a, b
            break

    return sent_l + clause_l, sent_l + clause_r


def _search_any(patterns: Sequence[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _extract_temporality(text: str) -> str | None:
    for pattern in _TEMPORALITY_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        cue = m.groupdict().get("t") or m.group(0)
        cue = cue.strip(" ,;:-")
        if cue:
            return cue
    return None


def _infer_experiencer(sentence_text: str) -> str:
    # Prefer explicit "patient" over bystander/caller framing.
    if _PATIENT_PATTERN.search(sentence_text):
        return "patient"
    if _search_any(_BYSTANDER_PATTERNS, sentence_text):
        return "bystander"
    return "patient"


def _infer_flags(*, clause_prefix: str) -> tuple[bool, bool]:
    negated = _search_any(_NEGATION_PATTERNS, clause_prefix)
    uncertain = _search_any(_UNCERTAINTY_PATTERNS, clause_prefix)
    return negated, uncertain


def apply_context(entities: Iterable[Entity], transcript: str | Transcript) -> list[Entity]:
    """Annotate entities with simple context attributes.

    Adds (or overwrites) these keys under `entity.attributes`:
      - negated: bool
      - uncertain: bool
      - experiencer: "patient" | "bystander"
      - temporality: {"text": str} | None
    """

    transcript_text = _transcript_to_text(transcript)
    if not transcript_text:
        return [e.model_copy() for e in entities]

    updated: list[Entity] = []
    for entity in entities:
        win = _find_entity_window(transcript_text, entity)
        if win is None:
            attrs = dict(entity.attributes)
            attrs.setdefault("negated", False)
            attrs.setdefault("uncertain", False)
            attrs.setdefault("experiencer", "patient")
            attrs.setdefault("temporality", None)
            updated.append(entity.model_copy(update={"attributes": attrs}))
            continue

        sent_l, sent_r = _sentence_bounds(transcript_text, win.start)
        clause_l, clause_r = _clause_bounds(transcript_text, win.start, win.end)

        clause_prefix = transcript_text[clause_l : win.start]
        # Limit scope to a small local window to avoid far-away cues.
        clause_prefix = clause_prefix[-120:]

        negated, uncertain = _infer_flags(clause_prefix=clause_prefix)
        experiencer = _infer_experiencer(transcript_text[sent_l:sent_r])

        around = transcript_text[max(sent_l, win.start - 80) : min(sent_r, win.end + 120)]
        temporality = _extract_temporality(around)

        attrs = dict(entity.attributes)
        attrs["negated"] = bool(negated)
        attrs["uncertain"] = bool(uncertain)
        attrs["experiencer"] = experiencer
        attrs["temporality"] = {"text": temporality} if temporality else None

        updated.append(entity.model_copy(update={"attributes": attrs}))

    return updated
