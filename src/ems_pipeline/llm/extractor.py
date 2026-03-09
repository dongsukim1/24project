"""LLM-based EMS entity extractor using Claude tool_use."""

from __future__ import annotations

import os
from typing import Any

from ems_pipeline.models import Entity, EntitiesDocument, Segment, Transcript
from ems_pipeline.normalize import UnitIdRule, normalize_text
from ems_pipeline.normalize import _compile_unit_id_rules
from ems_pipeline.normalize import _load_lexicon

ENTITY_TYPES = [
    "VITAL_SPO2", "VITAL_BP", "UNIT_ID", "SYMPTOM", "CONDITION",
    "PROCEDURE", "MEDICATION", "ASSESSMENT", "ETA", "ANATOMY", "RESOURCE",
]

EMS_SYSTEM_PROMPT = """\
You are an EMS clinical entity extractor. You analyze Emergency Medical Services \
radio transcripts and extract all clinically relevant entities.

## Entity Types and Extraction Rules

- **VITAL_SPO2**: Oxygen saturation readings (e.g., "SpO2 88%", "sat 94")
- **VITAL_BP**: Blood pressure readings as systolic/diastolic (e.g., "150/90", "BP 110 over 70")
- **UNIT_ID**: EMS unit callsigns (e.g., "MEDIC 12", "ENGINE 3", "RESCUE 7")
- **SYMPTOM**: Physical symptoms reported or observed (e.g., "chest pain", "SOB", "nausea", "altered mental status")
- **CONDITION**: Medical conditions or diagnoses (e.g., "STEMI", "stroke", "sepsis", "cardiac arrest")
- **PROCEDURE**: Clinical procedures performed (e.g., "CPR", "intubation", "IV access", "12-lead")
- **MEDICATION**: Drug names administered or considered (e.g., "naloxone", "epinephrine", "aspirin")
- **ASSESSMENT**: Clinical assessment scales (e.g., "GCS 14", "APGAR 8", "pain scale 7")
- **ETA**: Estimated time of arrival (e.g., "5 minutes", "ETA 10 mins")
- **ANATOMY**: Body locations referenced (e.g., "chest", "left arm", "abdomen")
- **RESOURCE**: Resources requested or dispatched (e.g., "ALS", "BLS", "helicopter")

## Context Rules

- If an entity is **negated** (e.g., "denies chest pain", "no SOB", "pain resolved"), set `negated: true`.
- If an entity is **uncertain** (e.g., "possible STEMI", "questionable fracture"), set `uncertain: true`.
- If mentioned about a **family member or bystander** (e.g., "caller says", "patient's wife reports"), set `experiencer: "bystander"`.
- If **temporal context** exists (e.g., "2 hours ago", "since yesterday", "onset 30 minutes prior"), capture in `temporality`.

## Instructions

Extract ALL entities from the provided transcript segments. Each entity must reference the \
segment_id where it was found. Assign a confidence score (0.0-1.0) based on how clearly the \
entity is stated. Use the `record_entities` tool to return your results.
"""

ENTITY_TOOL_DEFINITION: dict = {
    "name": "record_entities",
    "description": "Record all clinical entities extracted from EMS transcript segments.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ENTITY_TYPES},
                        "text": {"type": "string", "description": "Surface text as it appears in the transcript"},
                        "normalized": {"type": ["string", "null"], "description": "Canonical/normalized form, or null if same as text"},
                        "segment_id": {"type": "string", "description": "ID of the segment where this entity was found"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "negated": {"type": "boolean", "default": False},
                        "uncertain": {"type": "boolean", "default": False},
                        "experiencer": {"type": "string", "enum": ["patient", "bystander"], "default": "patient"},
                        "temporality": {"type": ["string", "null"], "default": None},
                    },
                    "required": ["type", "text", "segment_id", "confidence"],
                },
            },
        },
        "required": ["entities"],
    },
}


def build_extraction_prompt(
    segments: list[Segment],
    segment_ids: list[str],
) -> str:
    """Build the user message content for a chunk of segments."""
    lines = ["Extract all EMS entities from these transcript segments:\n"]
    for seg, seg_id in zip(segments, segment_ids):
        lines.append(f'[{seg_id}] ({seg.speaker}): "{seg.text}"')
    return "\n".join(lines)


def adaptive_chunk(
    segments: list[Segment],
    max_segments: int = 15,
) -> list[list[Segment]]:
    """Split segments into chunks for LLM extraction.

    If len(segments) <= max_segments, returns a single chunk.
    Otherwise, splits at speaker turn boundaries with 1-segment overlap.
    """
    if len(segments) <= max_segments:
        return [list(segments)]

    chunks: list[list[Segment]] = []
    start = 0

    while start < len(segments):
        end = min(start + max_segments, len(segments))

        if end < len(segments):
            # Look for speaker turn boundary, searching back up to half the window
            best_split = end
            search_limit = max(end - max_segments // 2, start + 1)
            for candidate in range(end, search_limit - 1, -1):
                if (
                    segments[candidate].speaker != segments[candidate - 1].speaker
                ):
                    best_split = candidate
                    break
            end = best_split

        chunk = segments[start:end]
        chunks.append(chunk)

        # Next chunk starts 1 before end (overlap segment)
        start = end - 1 if end < len(segments) else end

    return chunks


def merge_chunks(chunk_results: list[list[Entity]]) -> list[Entity]:
    """Merge entity lists from multiple chunks, deduplicating overlaps.

    Dedup key: (segment_id, type, text). Keeps the entity with highest confidence.
    """
    seen: dict[tuple[str, str, str], Entity] = {}

    for entities in chunk_results:
        for entity in entities:
            segment_id = entity.attributes.get("segment_id", "")
            key = (segment_id, entity.type, entity.text)
            existing = seen.get(key)
            if existing is None:
                seen[key] = entity
            else:
                existing_conf = existing.confidence if existing.confidence is not None else 0.0
                new_conf = entity.confidence if entity.confidence is not None else 0.0
                if new_conf > existing_conf:
                    seen[key] = entity

    return list(seen.values())


class LlmExtractor:
    """EMS entity extractor using Claude tool_use."""

    def __init__(
        self,
        client: Any = None,
        model: str | None = None,
    ) -> None:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self._client = client
        self._model = model or os.getenv("EMS_EXTRACT_MODEL", "claude-sonnet-4-20250514")

    def extract(self, transcript: Transcript) -> EntitiesDocument:
        """Extract entities from a transcript using Claude."""
        # Build segment ID map
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

        # Pre-normalize
        lexicon = _load_lexicon()
        unit_id_rules: list[UnitIdRule] = _compile_unit_id_rules(lexicon)
        normalized_segments: list[Segment] = []
        segment_normalization: dict[str, dict[str, Any]] = {}
        for i, seg in enumerate(transcript.segments):
            seg_id = index_to_segment_id[i]
            norm_text, norm_map = normalize_text(seg.text, unit_id_rules=unit_id_rules)
            segment_normalization[seg_id] = norm_map
            normalized_segments.append(
                Segment(
                    start=seg.start, end=seg.end, speaker=seg.speaker,
                    text=norm_text, confidence=seg.confidence,
                )
            )

        # Chunk
        chunks = adaptive_chunk(normalized_segments)
        chunk_segment_ids = self._build_chunk_segment_ids(
            normalized_segments, index_to_segment_id, chunks,
        )

        # Extract per chunk
        all_chunk_entities: list[list[Entity]] = []
        for chunk_segs, chunk_ids in zip(chunks, chunk_segment_ids):
            raw_entities = self._call_llm(chunk_segs, chunk_ids)
            hydrated = self._hydrate_entities(raw_entities, transcript.segments, index_to_segment_id)
            all_chunk_entities.append(hydrated)

        entities = merge_chunks(all_chunk_entities)

        # Build metadata
        full_text_original = " ".join(s.text for s in transcript.segments).strip()
        full_text_normalized = " ".join(s.text for s in normalized_segments).strip()

        return EntitiesDocument(
            entities=entities,
            metadata={
                "source": "llm_v1",
                "segment_id_map": {v: k for k, v in index_to_segment_id.items()},
                "transcript_original": full_text_original,
                "transcript_normalized": full_text_normalized,
                "segment_normalization": segment_normalization,
            },
        )

    def _build_chunk_segment_ids(
        self,
        all_segments: list[Segment],
        index_to_segment_id: dict[int, str],
        chunks: list[list[Segment]],
    ) -> list[list[str]]:
        seg_to_id: dict[int, str] = {}
        for i, seg in enumerate(all_segments):
            seg_to_id[id(seg)] = index_to_segment_id[i]
        result: list[list[str]] = []
        for chunk in chunks:
            result.append([seg_to_id[id(seg)] for seg in chunk])
        return result

    def _call_llm(
        self,
        segments: list[Segment],
        segment_ids: list[str],
    ) -> list[dict[str, Any]]:
        user_prompt = build_extraction_prompt(segments, segment_ids)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=EMS_SYSTEM_PROMPT,
            tools=[ENTITY_TOOL_DEFINITION],
            tool_choice={"type": "tool", "name": "record_entities"},
            messages=[{"role": "user", "content": user_prompt}],
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "record_entities":
                return block.input.get("entities", [])
        return []

    def _hydrate_entities(
        self,
        raw_entities: list[dict[str, Any]],
        original_segments: list[Segment],
        index_to_segment_id: dict[int, str],
    ) -> list[Entity]:
        """Convert raw LLM output dicts into Entity objects with segment metadata."""
        id_to_index: dict[str, int] = {v: k for k, v in index_to_segment_id.items()}
        entities: list[Entity] = []

        for raw in raw_entities:
            seg_id = raw.get("segment_id", "")
            seg_idx = id_to_index.get(seg_id)
            seg = original_segments[seg_idx] if seg_idx is not None and seg_idx < len(original_segments) else None

            attributes: dict[str, Any] = {
                "segment_id": seg_id,
                "source": "llm",
                "negated": raw.get("negated", False),
                "uncertain": raw.get("uncertain", False),
                "experiencer": raw.get("experiencer", "patient"),
            }
            temporality = raw.get("temporality")
            if temporality:
                attributes["temporality"] = temporality

            entities.append(Entity(
                type=raw.get("type", ""),
                text=raw.get("text", ""),
                normalized=raw.get("normalized"),
                start=seg.start if seg else None,
                end=seg.end if seg else None,
                speaker=seg.speaker if seg else None,
                confidence=raw.get("confidence"),
                attributes=attributes,
            ))

        return entities
