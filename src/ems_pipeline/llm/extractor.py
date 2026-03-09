"""LLM-based EMS entity extractor using Claude tool_use."""

from __future__ import annotations

from ems_pipeline.models import Entity, Segment

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
