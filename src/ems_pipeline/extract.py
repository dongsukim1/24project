"""Transcript -> EMS entity extraction stage.

Uses LLM-based extraction via Claude tool_use with EMS-specific vocabulary.
Text normalization (lexicon-based) is applied as a deterministic pre-processing step.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from ems_pipeline.llm.extractor import LlmExtractor
from ems_pipeline.models import EntitiesDocument, Entity, Transcript

_EXTRACT_DEBUG_ENV_VAR = "EMS_EXTRACT_DEBUG"

_llm_extractor: LlmExtractor | None = None


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


def extract_entities(
    transcript: Transcript,
    *,
    debug: bool | None = None,
) -> EntitiesDocument:
    """Extract EMS entities from a transcript using LLM-based extraction.

    Args:
        transcript: A diarized transcript (JSON from `ems_pipeline transcribe`).
        debug: When `True` (or env `EMS_EXTRACT_DEBUG=1`), emits debug summaries.

    Returns:
        EntitiesDocument: list of entities with context and optional timing info.
    """
    global _llm_extractor
    debug_enabled = _extract_debug_enabled(debug)

    if _llm_extractor is None:
        _llm_extractor = LlmExtractor()

    result = _llm_extractor.extract(transcript)

    if debug_enabled:
        _emit_extract_debug(
            "llm_entities",
            [_entity_debug_payload(e) for e in result.entities],
        )

    return result
