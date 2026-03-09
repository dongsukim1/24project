"""LLM-based EMS entity extractor using Claude tool_use."""

from __future__ import annotations

from ems_pipeline.models import Entity, Segment


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
