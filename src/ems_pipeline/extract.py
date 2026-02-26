"""Transcript -> EMS entity extraction stage.

This is a lightweight, rule-based extractor intended for the skeleton project.
It favors deterministic behavior and human-readable outputs over completeness.

TODO:
- Expand the ontology and lexicon coverage.
- Add negation/uncertainty detection (NegEx-like rules, classifier).
- Add phrase-level timing alignment (beyond segment-level attribution).
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from typing import Any

from ems_pipeline.models import EntitiesDocument, Entity, Transcript
from ems_pipeline.nlp.negation import apply_negation
from ems_pipeline.nlp.nlp_extractor import NlpExtractor
from ems_pipeline.nlp.normalize import UnitIdRule, normalize_text
from ems_pipeline.nlp.normalize import _compile_unit_id_rules as _compile_unit_id_rules_from_lexicon
from ems_pipeline.nlp.normalize import _load_lexicon as _load_lexicon_from_resources

_nlp_extractor: NlpExtractor | None = None
_nlp_extractor_initialized = False
_EXTRACT_DEBUG_ENV_VAR = "EMS_EXTRACT_DEBUG"


def get_nlp_extractor() -> NlpExtractor | None:
    global _nlp_extractor, _nlp_extractor_initialized
    if _nlp_extractor_initialized:
        return _nlp_extractor

    _nlp_extractor_initialized = True
    try:
        _nlp_extractor = NlpExtractor()
    except Exception as exc:
        warnings.warn(
            f"NLP extractor initialization failed; NLP extraction disabled: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        _nlp_extractor = None

    return _nlp_extractor


def _get_nlp_extractor() -> NlpExtractor | None:
    # Backward-compatible alias for older tests/callers.
    return get_nlp_extractor()


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _extract_debug_enabled(debug: bool | None) -> bool:
    if debug is not None:
        return debug
    return _is_truthy(os.getenv(_EXTRACT_DEBUG_ENV_VAR))


def _entity_debug_payload(entity: Entity) -> dict[str, Any]:
    attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
    source = attrs.get("source")
    segment_id = attrs.get("segment_id")
    return {
        "type": entity.type,
        "text": entity.text,
        "normalized": entity.normalized,
        "source": source if isinstance(source, str) else "unknown",
        "segment_id": segment_id if isinstance(segment_id, str) else None,
        "confidence": entity.confidence,
    }


def _emit_extract_debug(label: str, payload: Any) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    print(f"[extract.debug] {label}: {encoded}", file=sys.stderr)


def merge_entities(
    rule_entities: list[Entity],
    nlp_entities: list[Entity],
    confidence_threshold: float = 0.5,
) -> list[Entity]:
    def _segment_id(entity: Entity) -> str | None:
        segment_id = entity.attributes.get("segment_id")
        return segment_id if isinstance(segment_id, str) else None

    def _span(entity: Entity) -> tuple[float, float] | None:
        if entity.start is None or entity.end is None:
            return None
        if entity.end <= entity.start:
            return None
        return (entity.start, entity.end)

    known_types = {entity.type for entity in rule_entities}
    known_types.update(entity.type for entity in nlp_entities)

    related_type_groups: list[set[str]] = [
        {"SYMPTOM", "CONDITION"},
    ]
    if {"MEDICATION", "DRUG"} <= known_types:
        related_type_groups.append({"MEDICATION", "DRUG"})

    def types_are_related(type_a: str, type_b: str) -> bool:
        if type_a == type_b:
            return True
        return any(type_a in group and type_b in group for group in related_type_groups)

    occupied: dict[str, list[tuple[float, float, str]]] = {}
    for entity in rule_entities:
        segment_id = _segment_id(entity)
        span = _span(entity)
        if segment_id is None or span is None:
            continue
        occupied.setdefault(segment_id, []).append((span[0], span[1], entity.type))

    accepted_nlp_entities: list[Entity] = []
    for entity in nlp_entities:
        confidence = entity.confidence if entity.confidence is not None else 0.0
        if confidence < confidence_threshold:
            continue

        segment_id = _segment_id(entity)
        span = _span(entity)
        if segment_id is None or span is None:
            accepted_nlp_entities.append(entity)
            continue

        span_start, span_end = span
        overlaps_related = False
        for other_start, other_end, other_type in occupied.get(segment_id, []):
            if not types_are_related(entity.type, other_type):
                continue
            if span_start < other_end and span_end > other_start:
                overlaps_related = True
                break

        if overlaps_related:
            continue

        accepted_nlp_entities.append(entity)
        occupied.setdefault(segment_id, []).append((span_start, span_end, entity.type))

    return [*rule_entities, *accepted_nlp_entities]


def extract_entities(
    transcript: Transcript,
    *,
    debug: bool | None = None,
) -> EntitiesDocument:
    """Extract EMS entities + context from a transcript.

    Args:
        transcript: A diarized transcript (JSON from `ems_pipeline transcribe`).
        debug: When `True` (or env `EMS_EXTRACT_DEBUG=1`), emits debug summaries
            for rule entities, NLP pre-merge entities, negation drops, and merged
            output.

    Returns:
        EntitiesDocument: list of entities with context and optional timing info.
    """

    global _nlp_extractor
    debug_enabled = _extract_debug_enabled(debug)

    segment_id_map = (
        transcript.metadata.get("segment_id_map")
        if isinstance(transcript.metadata, dict)
        else None
    )
    index_to_segment_id: dict[int, str] = {}
    if isinstance(segment_id_map, dict):
        for seg_id, idx in segment_id_map.items():
            if isinstance(seg_id, str) and isinstance(idx, int):
                index_to_segment_id[idx] = seg_id

    for i in range(len(transcript.segments)):
        index_to_segment_id.setdefault(i, f"seg_{i:04d}")

    lexicon = _load_lexicon_from_resources()
    unit_id_rules: list[UnitIdRule] = _compile_unit_id_rules_from_lexicon(lexicon)

    entities: list[Entity] = []
    segment_normalization: dict[str, dict[str, Any]] = {}
    nlp_segment_inputs: list[tuple[str, str, Any]] = []

    spo2_re = re.compile(r"(?<!\w)SpO2\s+(?P<value>\d{2,3})%(?!\w)")
    bp_re = re.compile(r"(?<!\w)(?P<sys>\d{2,3})/(?P<dia>\d{2,3})(?!\w)")
    gcs_re = re.compile(r"(?<!\w)GCS\s*(?P<score>\d{1,2})(?!\w)", flags=re.IGNORECASE)
    sob_re = re.compile(r"(?<!\w)SOB(?!\w)")
    stemi_re = re.compile(r"(?<!\w)STEMI(?!\w)")
    cpr_re = re.compile(r"(?<!\w)CPR(?!\w)")
    als_bls_re = re.compile(r"(?<!\w)ALS/BLS(?!\w)", flags=re.IGNORECASE)
    naloxone_re = re.compile(r"(?<!\w)naloxone(?!\w)", flags=re.IGNORECASE)

    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }

    def _words_to_int(tok: str) -> int | None:
        toks = [t for t in re.split(r"[-\s]+", tok.lower().strip()) if t]
        if not toks:
            return None
        if len(toks) == 1 and toks[0] in number_words:
            return number_words[toks[0]]
        if len(toks) == 2 and toks[0] in number_words and toks[1] in number_words:
            if number_words[toks[0]] >= 20 and number_words[toks[1]] < 10:
                return number_words[toks[0]] + number_words[toks[1]]
        return None

    eta_re = re.compile(
        r"(?<!\w)ETA(?!\w)\s+(?P<num>\d+|[a-z]+)\s*(?P<unit>minutes?|mins?|hours?|hrs?)(?!\w)",
        flags=re.IGNORECASE,
    )

    def _emit(
        *,
        entity_type: str,
        text: str,
        normalized: str | None,
        seg_id: str,
        seg: Any,
    ) -> None:
        entities.append(
            Entity(
                type=entity_type,
                text=text,
                normalized=normalized,
                start=getattr(seg, "start", None),
                end=getattr(seg, "end", None),
                speaker=getattr(seg, "speaker", None),
                confidence=getattr(seg, "confidence", None),
                attributes={"segment_id": seg_id, "source": "rule"},
            )
        )

    for i, seg in enumerate(transcript.segments):
        seg_id = index_to_segment_id[i]
        seg_text = seg.text
        seg_norm, seg_map = normalize_text(seg_text, unit_id_rules=unit_id_rules)
        segment_normalization[seg_id] = seg_map
        nlp_segment_inputs.append((seg_id, seg_norm, seg))

        for rule in unit_id_rules:
            for m in rule.pattern.finditer(seg_text):
                surface = m.group(0)
                norm_unit, _ = normalize_text(surface, unit_id_rules=[rule])
                _emit(
                    entity_type="UNIT_ID",
                    text=surface,
                    normalized=norm_unit,
                    seg_id=seg_id,
                    seg=seg,
                )

        for m in spo2_re.finditer(seg_norm):
            v = f"SpO2 {m.group('value')}%"
            _emit(entity_type="VITAL_SPO2", text=v, normalized=v, seg_id=seg_id, seg=seg)

        for m in bp_re.finditer(seg_norm):
            v = f"{m.group('sys')}/{m.group('dia')}"
            _emit(entity_type="VITAL_BP", text=v, normalized=v, seg_id=seg_id, seg=seg)

        if sob_re.search(seg_norm):
            _emit(entity_type="SYMPTOM", text="SOB", normalized="SOB", seg_id=seg_id, seg=seg)
        if stemi_re.search(seg_norm):
            _emit(entity_type="CONDITION", text="STEMI", normalized="STEMI", seg_id=seg_id, seg=seg)
        if cpr_re.search(seg_norm):
            _emit(entity_type="PROCEDURE", text="CPR", normalized="CPR", seg_id=seg_id, seg=seg)
        if als_bls_re.search(seg_norm):
            _emit(
                entity_type="RESOURCE",
                text="ALS/BLS",
                normalized="ALS/BLS",
                seg_id=seg_id,
                seg=seg,
            )
        if naloxone_re.search(seg_norm):
            _emit(
                entity_type="MEDICATION",
                text="naloxone",
                normalized="naloxone",
                seg_id=seg_id,
                seg=seg,
            )

        for m in gcs_re.finditer(seg_norm):
            v = f"GCS {m.group('score')}"
            _emit(entity_type="ASSESSMENT", text=v, normalized=v, seg_id=seg_id, seg=seg)

        for m in eta_re.finditer(seg_norm):
            num_raw = m.group("num")
            unit = m.group("unit").lower()
            num = int(num_raw) if num_raw.isdigit() else _words_to_int(num_raw)
            if num is None:
                continue
            canonical_unit = "minutes" if unit.startswith("min") else "hours"
            v = f"{num} {canonical_unit}"
            _emit(entity_type="ETA", text=v, normalized=v, seg_id=seg_id, seg=seg)

    full_text_original = " ".join(s.text for s in transcript.segments).strip()
    full_text_normalized, full_map = normalize_text(full_text_original, unit_id_rules=unit_id_rules)

    nlp_entities: list[Entity] = []
    negation_drops: list[dict[str, Any]] = []
    merge_confidence_threshold = 0.5
    extractor: NlpExtractor | None = None
    used_transcript_fallback = False
    for seg_id, seg_norm, seg in nlp_segment_inputs:
        if extractor is None:
            extractor = _get_nlp_extractor()
            if extractor is None:
                break
            merge_confidence_threshold = float(getattr(extractor, "confidence_threshold", 0.5))

        try:
            extracted = extractor.extract(seg_norm, seg)
        except TypeError:
            try:
                extracted = extractor.extract(transcript, index_to_segment_id=index_to_segment_id)
                used_transcript_fallback = True
            except ImportError as exc:
                warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
                _nlp_extractor = None
                nlp_entities = []
                break
            except Exception:
                nlp_entities = []
                break
        except ImportError as exc:
            warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
            _nlp_extractor = None
            nlp_entities = []
            break
        except Exception:
            nlp_entities = []
            break

        pop_negation_drops = getattr(extractor, "pop_negation_drops", None)
        if callable(pop_negation_drops):
            try:
                dropped = pop_negation_drops()
                if isinstance(dropped, list):
                    negation_drops.extend(
                        item for item in dropped if isinstance(item, dict)
                    )
            except Exception:
                pass

        for entity in extracted:
            if not used_transcript_fallback:
                entity.attributes["segment_id"] = seg_id
            entity.attributes.setdefault("source", "nlp")
        nlp_entities.extend(extracted)
        if used_transcript_fallback:
            break

    if nlp_entities:
        try:
            nlp_entities = apply_negation(nlp_entities, transcript)
        except Exception:
            nlp_entities = []

    rule_entities = list(entities)
    entities = merge_entities(
        rule_entities,
        nlp_entities,
        confidence_threshold=merge_confidence_threshold,
    )

    if debug_enabled:
        _emit_extract_debug("rule_entities", [_entity_debug_payload(e) for e in rule_entities])
        _emit_extract_debug(
            "nlp_entities_pre_merge",
            [_entity_debug_payload(e) for e in nlp_entities],
        )
        _emit_extract_debug("entities_removed_by_negation", negation_drops)
        _emit_extract_debug("final_merged_entities", [_entity_debug_payload(e) for e in entities])

    return EntitiesDocument(
        entities=entities,
        metadata={
            "source": "rule_based_v0",
            "segment_id_map": {v: k for k, v in index_to_segment_id.items()},
            "transcript_original": full_text_original,
            "transcript_normalized": full_text_normalized,
            "transcript_normalization": full_map,
            "segment_normalization": segment_normalization,
        },
    )
