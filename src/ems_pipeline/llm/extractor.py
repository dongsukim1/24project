"""LLM-based EMS entity extractor using Claude tool_use."""

from __future__ import annotations

from ems_pipeline.models import Segment


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
