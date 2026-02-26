from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ems_pipeline.models import Entity, Transcript
from ems_pipeline.nlp.context import apply_context

TriggerType = Literal["pre_negation", "post_negation", "experiencer"]

# Negation trigger terms that occur before the entity mention.
PRE_NEGATION_TRIGGERS: tuple[str, ...] = (
    "denies",
    "denied",
    "deny",
    "no evidence of",
    "negative for",
    "without",
    "does not have",
    "do not have",
    "did not have",
    "not",
    "no",
)

# Negation trigger terms that occur after the entity mention.
POST_NEGATION_TRIGGERS: tuple[str, ...] = (
    "ruled out",
    "rule out",
    "excluded",
)

# Terms that end negation scope when found between trigger and entity.
SCOPE_TERMINATORS: tuple[str, ...] = (
    "but",
    "however",
    "though",
    "although",
    "yet",
    ".",
    ";",
    "!",
    "?",
)

# Phrases that look like negation but usually indicate uncertainty/affirmation.
PSEUDO_NEGATION: tuple[str, ...] = (
    "not ruled out",
    "cannot rule out",
    "can't rule out",
    "not excluded",
)

# Experiencer cues used to mark family-history context.
EXPERIENCER_TRIGGERS: tuple[str, ...] = (
    "family history of",
    "family history",
    "mother",
    "father",
    "sister",
    "brother",
    "grandmother",
    "grandfather",
    "aunt",
    "uncle",
)


@dataclass(frozen=True, slots=True)
class NegationResult:
    is_negated: bool
    is_family_history: bool
    trigger_term: str | None
    trigger_type: TriggerType | None


@dataclass(frozen=True, slots=True)
class _TriggerMatch:
    term: str
    start: int
    end: int


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags=re.IGNORECASE)


def _find_matches(text: str, terms: tuple[str, ...]) -> list[_TriggerMatch]:
    matches: list[_TriggerMatch] = []
    for term in terms:
        pattern = _term_pattern(term)
        for m in pattern.finditer(text):
            matches.append(_TriggerMatch(term=m.group(0), start=m.start(), end=m.end()))
    return matches


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _in_pseudo(match: _TriggerMatch, pseudo_hits: list[_TriggerMatch]) -> bool:
    return any(
        _overlaps(match.start, match.end, pseudo.start, pseudo.end)
        for pseudo in pseudo_hits
    )


def _span_within_pseudo(span: tuple[int, int], pseudo_hits: list[_TriggerMatch]) -> bool:
    start, end = span
    return any(start >= pseudo.start and end <= pseudo.end for pseudo in pseudo_hits)


def _has_terminator(text: str) -> bool:
    for terminator in SCOPE_TERMINATORS:
        if terminator in {".", ";", "!", "?"}:
            if terminator in text:
                return True
            continue
        if _term_pattern(terminator).search(text):
            return True
    return False


class NegationDetector:
    """Rule-based sentence-level negation detector (stateless).

    Limitations:
    - Uses local token/phrase matching; does not understand syntax or long-distance scope well.
    - Handles one sentence at a time and depends on caller-provided span offsets.

    V2 upgrade path:
    - Replace clause-window heuristics with dependency parsing to improve scope and trigger linking.

    Comment-style doctest examples:
    >>> detector = NegationDetector()
    >>> s = "denies chest pain"
    >>> cp = (s.index("chest pain"), s.index("chest pain") + len("chest pain"))
    >>> detector.check_sentence(s, [cp])[0].is_negated
    True
    >>> s = "denies chest pain but reports nausea"
    >>> cp = (s.index("chest pain"), s.index("chest pain") + len("chest pain"))
    >>> nz = (s.index("nausea"), s.index("nausea") + len("nausea"))
    >>> [r.is_negated for r in detector.check_sentence(s, [cp, nz])]
    [True, False]
    >>> s = "no evidence of fracture"
    >>> fx = (s.index("fracture"), s.index("fracture") + len("fracture"))
    >>> detector.check_sentence(s, [fx])[0].is_negated
    True
    >>> s = "chest pain is ruled out"
    >>> cp = (s.index("chest pain"), s.index("chest pain") + len("chest pain"))
    >>> detector.check_sentence(s, [cp])[0].is_negated
    True
    >>> s = "not ruled out for PE"
    >>> pe = (s.index("PE"), s.index("PE") + len("PE"))
    >>> detector.check_sentence(s, [pe])[0].is_negated
    False
    >>> s = "family history of diabetes"
    >>> dm = (s.index("diabetes"), s.index("diabetes") + len("diabetes"))
    >>> detector.check_sentence(s, [dm])[0].is_family_history
    True
    >>> s = "patient states he does not have chest pain"
    >>> cp = (s.index("chest pain"), s.index("chest pain") + len("chest pain"))
    >>> detector.check_sentence(s, [cp])[0].is_negated
    True
    """

    def check_sentence(self, sentence: str, spans: list[tuple[int, int]]) -> list[NegationResult]:
        text = sentence or ""
        pseudo_hits = _find_matches(text, PSEUDO_NEGATION)
        pre_hits = _find_matches(text, PRE_NEGATION_TRIGGERS)
        post_hits = _find_matches(text, POST_NEGATION_TRIGGERS)
        experiencer_hits = _find_matches(text, EXPERIENCER_TRIGGERS)

        results: list[NegationResult] = []
        for raw_start, raw_end in spans:
            start = max(0, min(len(text), raw_start))
            end = max(start, min(len(text), raw_end))
            span = (start, end)

            family_hit = self._nearest_backward_hit(
                trigger_hits=experiencer_hits,
                span_start=start,
                text=text,
                pseudo_hits=pseudo_hits,
            )
            is_family_history = family_hit is not None

            if _span_within_pseudo(span, pseudo_hits):
                if family_hit:
                    results.append(
                        NegationResult(
                            is_negated=False,
                            is_family_history=True,
                            trigger_term=family_hit.term,
                            trigger_type="experiencer",
                        )
                    )
                else:
                    results.append(
                        NegationResult(
                            is_negated=False,
                            is_family_history=False,
                            trigger_term=None,
                            trigger_type=None,
                        )
                    )
                continue

            pre_hit = self._nearest_backward_hit(
                trigger_hits=pre_hits,
                span_start=start,
                text=text,
                pseudo_hits=pseudo_hits,
            )
            if pre_hit:
                results.append(
                    NegationResult(
                        is_negated=True,
                        is_family_history=is_family_history,
                        trigger_term=pre_hit.term,
                        trigger_type="pre_negation",
                    )
                )
                continue

            post_hit = self._nearest_forward_hit(
                trigger_hits=post_hits,
                span_end=end,
                pseudo_hits=pseudo_hits,
            )
            if post_hit:
                results.append(
                    NegationResult(
                        is_negated=True,
                        is_family_history=is_family_history,
                        trigger_term=post_hit.term,
                        trigger_type="post_negation",
                    )
                )
                continue

            if family_hit:
                results.append(
                    NegationResult(
                        is_negated=False,
                        is_family_history=True,
                        trigger_term=family_hit.term,
                        trigger_type="experiencer",
                    )
                )
            else:
                results.append(
                    NegationResult(
                        is_negated=False,
                        is_family_history=False,
                        trigger_term=None,
                        trigger_type=None,
                    )
                )

        return results

    def _nearest_backward_hit(
        self,
        *,
        trigger_hits: list[_TriggerMatch],
        span_start: int,
        text: str,
        pseudo_hits: list[_TriggerMatch],
    ) -> _TriggerMatch | None:
        candidates = [
            hit
            for hit in trigger_hits
            if hit.end <= span_start and not _in_pseudo(hit, pseudo_hits)
        ]
        candidates.sort(key=lambda hit: hit.end, reverse=True)
        for hit in candidates:
            between = text[hit.end:span_start]
            if _has_terminator(between):
                continue
            return hit
        return None

    def _nearest_forward_hit(
        self,
        *,
        trigger_hits: list[_TriggerMatch],
        span_end: int,
        pseudo_hits: list[_TriggerMatch],
    ) -> _TriggerMatch | None:
        candidates = [
            hit
            for hit in trigger_hits
            if hit.start >= span_end and not _in_pseudo(hit, pseudo_hits)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda hit: hit.start)


def apply_negation(entities: list[Entity], transcript: Transcript) -> list[Entity]:
    """Compatibility wrapper that reuses the existing context annotator."""

    return apply_context(entities, transcript)


__all__ = [
    "EXPERIENCER_TRIGGERS",
    "NegationDetector",
    "NegationResult",
    "POST_NEGATION_TRIGGERS",
    "PRE_NEGATION_TRIGGERS",
    "PSEUDO_NEGATION",
    "SCOPE_TERMINATORS",
    "apply_negation",
]
