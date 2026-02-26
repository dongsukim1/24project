"""Pydantic models for the EMS Ambient Pipeline.

These schemas are the contract between pipeline stages:

- `transcribe` produces a `Transcript`
- `extract` produces a list of `Entity` (wrapped in `EntitiesDocument`)
- `build-claim` produces a `Claim`

The actual ML/NLP implementations are intentionally omitted. See the stage modules for TODOs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Segment(BaseModel):
    """A diarized transcript segment.

    Times are seconds from the start of the input audio.
    """

    model_config = ConfigDict(extra="forbid")

    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    speaker: str
    text: str
    confidence: float = Field(..., ge=0, le=1)


class Transcript(BaseModel):
    """A full transcript consisting of diarized segments plus free-form metadata."""

    model_config = ConfigDict(extra="forbid")

    segments: list[Segment]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """An extracted EMS entity mention with context.

    `start`/`end` refer to seconds from the beginning of the audio.
    `attributes` may include negation/uncertainty cues, section labels, etc.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    text: str
    normalized: str | None = None
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    speaker: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    attributes: dict[str, Any] = Field(default_factory=dict)


class EntitiesDocument(BaseModel):
    """Container for entity extraction output, including linkable provenance."""

    model_config = ConfigDict(extra="forbid")

    entities: list[Entity]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProvenanceLink(BaseModel):
    """Links a derived claim field back to transcript evidence."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str
    entity_index: int | None = Field(default=None, ge=0)
    note: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class Claim(BaseModel):
    """Structured proto-claim JSON built from extracted entities.

    `fields` is a shallow JSON object for early-stage prototyping.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "0.1"
    claim_id: str
    fields: dict[str, Any] = Field(default_factory=dict)
    provenance: list[ProvenanceLink] = Field(default_factory=list)

