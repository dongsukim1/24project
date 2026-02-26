"""Transcript -> EMS entity extraction stage.

This is a lightweight, rule-based extractor intended for the skeleton project.
It favors deterministic behavior and human-readable outputs over completeness.

TODO:
- Expand the ontology and lexicon coverage.
- Add negation/uncertainty detection (NegEx-like rules, classifier).
- Add phrase-level timing alignment (beyond segment-level attribution).
"""

from __future__ import annotations

import re
from typing import Any

from ems_pipeline.models import EntitiesDocument, Entity, Transcript
from ems_pipeline.nlp.normalize import UnitIdRule, normalize_text
from ems_pipeline.nlp.normalize import _compile_unit_id_rules as _compile_unit_id_rules_from_lexicon
from ems_pipeline.nlp.normalize import _load_lexicon as _load_lexicon_from_resources


def extract_entities(transcript: Transcript) -> EntitiesDocument:
    """Extract EMS entities + context from a transcript.

    Args:
        transcript: A diarized transcript (JSON from `ems_pipeline transcribe`).

    Returns:
        EntitiesDocument: list of entities with context and optional timing info.
    """

    segment_id_map = transcript.metadata.get("segment_id_map") if isinstance(transcript.metadata, dict) else None
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

    def _emit(*, entity_type: str, text: str, normalized: str | None, seg_id: str, seg: Any) -> None:
        entities.append(
            Entity(
                type=entity_type,
                text=text,
                normalized=normalized,
                start=getattr(seg, "start", None),
                end=getattr(seg, "end", None),
                speaker=getattr(seg, "speaker", None),
                confidence=getattr(seg, "confidence", None),
                attributes={"segment_id": seg_id},
            )
        )

    for i, seg in enumerate(transcript.segments):
        seg_id = index_to_segment_id[i]
        seg_text = seg.text
        seg_norm, seg_map = normalize_text(seg_text, unit_id_rules=unit_id_rules)
        segment_normalization[seg_id] = seg_map

        for rule in unit_id_rules:
            for m in rule.pattern.finditer(seg_text):
                surface = m.group(0)
                norm_unit, _ = normalize_text(surface, unit_id_rules=[rule])
                _emit(entity_type="UNIT_ID", text=surface, normalized=norm_unit, seg_id=seg_id, seg=seg)

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
            _emit(entity_type="RESOURCE", text="ALS/BLS", normalized="ALS/BLS", seg_id=seg_id, seg=seg)
        if naloxone_re.search(seg_norm):
            _emit(entity_type="MEDICATION", text="naloxone", normalized="naloxone", seg_id=seg_id, seg=seg)

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
