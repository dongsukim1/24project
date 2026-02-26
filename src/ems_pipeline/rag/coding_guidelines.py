"""Index 3: Coding Guidelines RAG scaffold.

Provides a file-backed JSON store for ICD-10/CPT coding rules, a retrieval
interface keyed by entity type and optional specialty/code filters, and the
ingestion contract used by ``scripts/ingest_coding_guidelines.py``.

Upgrade path
------------
The backing store is a plain JSON file (simple linear scan).  When the data
volume grows beyond a few thousand entries, swap ``CodingGuidelinesIndex``
for a vector-database backend (e.g., ChromaDB, Qdrant, pgvector):

1. Replace ``self._entries: list[CodingGuidelineEntry]`` with a VDB client.
2. Replace ``add_entry()`` with a VDB upsert call.
3. Replace the linear ``retrieve()`` loop with an ANN search query.
4. ``save()`` / ``load()`` become no-ops (VDB manages persistence).
5. Keep ``CodingGuidelineEntry`` and ``ENTITY_TYPE_TO_BODY_SYSTEM`` as-is —
   they are the stable data contract regardless of backend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity-type → body-system mapping
# ---------------------------------------------------------------------------

# Covers all entity types emitted by src/ems_pipeline/extract.py.
# None means "no specific body system" — entries without a body_system
# restriction are not filtered out by the body-system filter.
ENTITY_TYPE_TO_BODY_SYSTEM: dict[str, str | None] = {
    "VITAL_SPO2": "respiratory",    # SpO2 measures oxygen saturation
    "VITAL_BP": "cardiovascular",   # blood pressure
    "SYMPTOM": None,                # too varied for a single body system
    "CONDITION": None,              # too varied (STEMI, sepsis, etc.)
    "PROCEDURE": None,              # too varied (CPR, intubation, etc.)
    "RESOURCE": None,               # operational (ALS/BLS) — not clinical
    "MEDICATION": None,             # too varied (aspirin, naloxone, etc.)
    "ASSESSMENT": None,             # too varied (GCS, AVPU, etc.)
    "UNIT_ID": None,                # operational — not clinical
    "ETA": None,                    # operational — not clinical
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class CodingGuidelineEntry(BaseModel):
    """A single ICD-10 or CPT coding guideline record."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="ICD-10 or CPT code, e.g. 'R07.9' or '92950'.")
    code_type: Literal["ICD10", "CPT"]
    description: str = Field(..., description="Human-readable code description.")
    specialty: str | None = Field(
        default=None,
        description="Clinical specialty scope, e.g. 'emergency_medicine', 'cardiology'.",
    )
    body_system: str | None = Field(
        default=None,
        description="Anatomical system, e.g. 'cardiovascular', 'respiratory'.",
    )
    required_documentation: list[str] = Field(
        default_factory=list,
        description="Documentation elements required to support this code.",
    )
    exclusion_codes: list[str] = Field(
        default_factory=list,
        description="Codes that cannot be billed together with this code.",
    )
    notes: str | None = Field(default=None, description="Additional coding notes.")
    source: str = Field(..., description="Provenance of this guideline entry.")


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class CodingGuidelinesIndex:
    """File-backed JSON store for coding guideline entries.

    # STUB: backing store is a JSON file with linear scan retrieval.
    # Replace with a vector-database client (ChromaDB, Qdrant, pgvector) when
    # scale requires it.  See module docstring for the upgrade path.

    Args:
        store_path: Path to the JSON file that persists the index.
                    If the file already exists it is loaded on construction.
    """

    def __init__(self, store_path: str | Path) -> None:
        self._path = Path(store_path)
        self._entries: list[CodingGuidelineEntry] = []
        if self._path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_entry(self, entry: CodingGuidelineEntry) -> None:
        """Append a guideline entry to the in-memory store.

        Call ``save()`` afterwards to persist.
        """
        self._entries.append(entry)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        entity_types: list[str],
        codes: list[str] | None = None,
        specialty: str | None = None,
        top_k: int = 5,
    ) -> list[CodingGuidelineEntry]:
        """Return up to *top_k* guideline entries relevant to the query.

        Filters are applied in this order:

        1. **Code prefix** — if *codes* is provided, keep only entries whose
           ``code`` starts with one of the supplied prefixes (case-insensitive).
        2. **Specialty** — if *specialty* is provided, keep entries where
           ``entry.specialty`` is ``None`` (generic) or equals *specialty*.
        3. **Body system** — map *entity_types* to body systems via
           :data:`ENTITY_TYPE_TO_BODY_SYSTEM`; keep entries where
           ``entry.body_system`` is ``None`` (generic) or in the derived set.
           If all entity types map to ``None``, this filter is skipped.

        Linear scan; acceptable for file-backed stores up to ~10 k entries.

        Args:
            entity_types: Entity type strings from the session (e.g. ``["VITAL_BP"]``).
            codes:        Optional list of code prefixes to narrow the result set.
            specialty:    Optional clinical specialty filter.
            top_k:        Maximum number of entries to return.

        Returns:
            Filtered list of :class:`CodingGuidelineEntry`, length ≤ *top_k*.
        """
        results: list[CodingGuidelineEntry] = list(self._entries)

        # --- Filter 1: code prefix match ---
        if codes:
            prefixes = [c.upper() for c in codes]
            results = [
                e for e in results
                if any(e.code.upper().startswith(p) for p in prefixes)
            ]

        # --- Filter 2: specialty ---
        if specialty:
            results = [
                e for e in results
                if e.specialty is None or e.specialty == specialty
            ]

        # --- Filter 3: body system from entity types ---
        if entity_types:
            body_systems: set[str] = {
                bs
                for et in entity_types
                if (bs := ENTITY_TYPE_TO_BODY_SYSTEM.get(et.upper())) is not None
            }
            if body_systems:
                results = [
                    e for e in results
                    if e.body_system is None or e.body_system in body_systems
                ]

        return results[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Serialize the in-memory store to the JSON file at ``store_path``."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [e.model_dump(mode="json") for e in self._entries]
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug("CodingGuidelinesIndex saved %d entries to %s", len(self._entries), self._path)

    def load(self) -> None:
        """Deserialize the JSON file at ``store_path`` into the in-memory store."""
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._entries = [CodingGuidelineEntry.model_validate(d) for d in raw]
        logger.debug("CodingGuidelinesIndex loaded %d entries from %s", len(self._entries), self._path)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)
