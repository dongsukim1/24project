"""Index 2: Payer Rules & Policy Knowledge Base RAG scaffold.

Architecture principle: build inductively from denial data.
Every denial is a data point.  The primary ingestion pipeline is denial
records — not parsed policy PDFs.  PayerRuleEntry objects are *synthesized*
from denial patterns by ``scripts/synthesize_payer_rules.py``.

Data flow
---------
1. Denial EOBs / appeal outcomes → ``DenialRecord`` (raw)
2. ``scripts/ingest_denial.py`` → ``PayerRulesIndex.add_denial()``
3. ``scripts/synthesize_payer_rules.py`` → groups denials → ``PayerRuleEntry``
4. ``retrieve_payer_requirements()`` in ``rag/__init__.py`` → ``PayerRulesIndex.retrieve_for_payer()``
5. Agent-3 ``validate_claim()`` consumes the returned context dict

Upgrade path
------------
The backing store is a plain JSON file with linear scan retrieval.
When data volume grows beyond a few thousand entries, swap
``PayerRulesIndex`` for a vector-database backend (ChromaDB, Qdrant, pgvector):

1. Replace ``self._denials`` / ``self._rules`` with VDB collections.
2. Replace ``add_denial()`` / ``add_rule()`` with VDB upsert calls.
3. Replace the linear ``retrieve_for_payer()`` loop with an ANN search query.
4. ``save()`` / ``load()`` become no-ops (VDB manages persistence).
5. Keep ``DenialRecord`` and ``PayerRuleEntry`` as-is — they are the stable
   data contract regardless of backend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AppealAttempt(BaseModel):
    """A single appeal attempt attached to a denial record."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(..., description="Unique identifier for this appeal attempt.")
    strategy: str = Field(..., description="Appeal strategy used for this attempt.")
    outcome: Literal["success", "failure", "pending", "withdrawn"] | None = Field(
        default=None,
        description="Outcome of the attempt; None when not yet known.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional free-text notes about the attempt.",
    )
    timestamp: str = Field(
        ...,
        description="ISO timestamp when this attempt was recorded.",
    )


class DenialRecord(BaseModel):
    """A single payer denial event — the primary data unit for Index 2.

    # STUB: no real denial data yet.  Add records via scripts/ingest_denial.py.
    """

    model_config = ConfigDict(extra="forbid")

    denial_id: str = Field(..., description="UUID identifying this denial record.")
    encounter_id: str | None = Field(
        default=None,
        description="Optional pipeline encounter_id associated with this denial.",
    )
    payer_id: str = Field(..., description="Payer identifier (e.g. 'BCBS_TX').")
    cpt_codes: list[str] = Field(
        default_factory=list,
        description="CPT codes on the denied claim.",
    )
    icd_codes: list[str] = Field(
        default_factory=list,
        description="ICD-10 codes on the denied claim.",
    )
    denial_reason: str = Field(..., description="Human-readable denial reason.")
    denial_code: str | None = Field(
        default=None,
        description="Payer-specific denial code (e.g. 'CO-4', 'PR-96').",
    )
    policy_citation: str | None = Field(
        default=None,
        description="Policy reference cited in the denial (e.g. 'LCD L34462').",
    )
    date: str | None = Field(
        default=None,
        description="ISO date of the denial (e.g. '2024-03-15').",
    )
    resolved: bool = Field(
        default=False,
        description="True when the denial was successfully appealed / resolved.",
    )
    resolution: str | None = Field(
        default=None,
        description="Outcome of the appeal or resolution action.",
    )
    source: Literal["denial_eob", "manual_entry", "appeal_outcome"] = Field(
        ...,
        description="Provenance of this denial record.",
    )
    appeal_attempts: list[AppealAttempt] = Field(
        default_factory=list,
        description="Chronological appeal attempts tied to this denial.",
    )


class PayerRuleEntry(BaseModel):
    """A synthesized payer rule derived from one or more denial records.

    Instances are created by ``scripts/synthesize_payer_rules.py`` when
    enough denial records with the same (payer_id, denial_reason,
    codes_affected) signature have been accumulated.

    # STUB: simple frequency-based synthesis — replace with LLM-assisted
    # rule extraction when real denial data is available.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., description="UUID identifying this rule.")
    payer_id: str = Field(..., description="Payer this rule applies to.")
    rule_type: Literal[
        "coverage",
        "medical_necessity",
        "authorization",
        "bundling",
        "documentation",
    ] = Field(..., description="Category of payer rule.")
    codes_affected: list[str] = Field(
        default_factory=list,
        description="CPT or ICD-10 codes governed by this rule.",
    )
    description: str = Field(..., description="Plain-English rule description.")
    common_denial_reason: str | None = Field(
        default=None,
        description="The denial reason most often associated with this rule.",
    )
    required_documentation: list[str] = Field(
        default_factory=list,
        description="Documentation elements required to satisfy this rule.",
    )
    derived_from_denial_ids: list[str] = Field(
        default_factory=list,
        description="denial_ids of DenialRecords this rule was synthesized from.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How well-established this rule is (0–1). "
            "More supporting denials → higher confidence."
        ),
    )
    source: str = Field(..., description="Provenance of this rule entry.")


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class PayerRulesIndex:
    """File-backed JSON store for denial records and synthesized payer rules.

    # STUB: backing store is a JSON file with linear scan retrieval.
    # Replace with a vector-database client when scale requires it.
    # See module docstring for the upgrade path.

    Args:
        store_path: Path to the JSON file that persists the index.
                    If the file already exists it is loaded on construction.
    """

    def __init__(self, store_path: str | Path) -> None:
        self._path = Path(store_path)
        self._denials: list[DenialRecord] = []
        self._rules: list[PayerRuleEntry] = []
        if self._path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_denial(self, record: DenialRecord) -> None:
        """Append a denial record to the in-memory store.

        Call ``save()`` afterwards to persist.
        """
        self._denials.append(record)

    def add_rule(self, rule: PayerRuleEntry) -> None:
        """Append a synthesized payer rule to the in-memory store.

        Call ``save()`` afterwards to persist.
        """
        self._rules.append(rule)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_for_payer(
        self,
        payer_id: str,
        codes: list[str] | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Return rules and denial patterns for the given payer.

        Filters:
        1. **Payer match** — only rules whose ``rule.payer_id == payer_id``.
        2. **Code overlap** — if *codes* is provided, keep rules where
           ``rule.codes_affected`` shares at least one code with *codes*
           (case-insensitive prefix match, same convention as Index 3).

        Returns a context dict structured for Agent-3 consumption::

            {
                "payer_id":              str,
                "rules":                 list[PayerRuleEntry],
                "common_denial_reasons": list[str],   # deduplicated
                "required_documentation": list[str],  # deduplicated union
            }

        Returns an empty-context dict (same shape, empty lists) when no
        rules are found for the payer — callers continue normally.

        Args:
            payer_id: Payer identifier to query.
            codes:    Optional list of CPT/ICD-10 code prefixes to narrow
                      results (case-insensitive prefix match).
            top_k:    Maximum number of rule entries to return.

        Returns:
            Context dict ready for Agent-3 ``validate_claim()`` consumption.
        """
        rules: list[PayerRuleEntry] = [
            r for r in self._rules if r.payer_id == payer_id
        ]

        if codes:
            upper_codes = [c.upper() for c in codes]
            rules = [
                r for r in rules
                if any(
                    affected.upper().startswith(prefix)
                    for affected in r.codes_affected
                    for prefix in upper_codes
                )
            ]

        rules = rules[:top_k]

        # Build deduplicated auxiliary lists across matched rules.
        denial_reasons: list[str] = []
        req_docs: list[str] = []
        seen_reasons: set[str] = set()
        seen_docs: set[str] = set()
        for r in rules:
            if r.common_denial_reason and r.common_denial_reason not in seen_reasons:
                denial_reasons.append(r.common_denial_reason)
                seen_reasons.add(r.common_denial_reason)
            for doc in r.required_documentation:
                if doc not in seen_docs:
                    req_docs.append(doc)
                    seen_docs.add(doc)

        return {
            "payer_id": payer_id,
            "rules": rules,
            "common_denial_reasons": denial_reasons,
            "required_documentation": req_docs,
        }

    def get_denial_patterns(
        self,
        payer_id: str | None = None,
        denial_reason: str | None = None,
        code: str | None = None,
    ) -> list[DenialRecord]:
        """Return raw denial records filtered by payer/reason/code.

        Args:
            payer_id: Optional payer identifier to filter by.
            denial_reason:
                      Optional denial reason string (case-insensitive exact match).
            code:     Optional CPT or ICD-10 code prefix.  Records whose
                      ``cpt_codes`` or ``icd_codes`` contain a code that
                      starts with this prefix (case-insensitive) are kept.

        Returns:
            Filtered list of :class:`DenialRecord`.
        """
        records = list(self._denials)

        if payer_id:
            records = [d for d in records if d.payer_id == payer_id]

        if denial_reason:
            target_reason = denial_reason.strip().casefold()
            records = [
                d for d in records if d.denial_reason.strip().casefold() == target_reason
            ]

        if code:
            prefix = code.upper()
            records = [
                d for d in records
                if any(c.upper().startswith(prefix) for c in d.cpt_codes + d.icd_codes)
            ]
        return records

    def get_denial_by_id(self, denial_id: str) -> DenialRecord | None:
        """Return the denial record with matching ``denial_id``, if present."""
        for denial in self._denials:
            if denial.denial_id == denial_id:
                return denial
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Serialize the in-memory store to the JSON file at ``store_path``."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "denials": [d.model_dump(mode="json") for d in self._denials],
            "rules": [r.model_dump(mode="json") for r in self._rules],
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug(
            "PayerRulesIndex saved %d denials, %d rules to %s",
            len(self._denials),
            len(self._rules),
            self._path,
        )

    def load(self) -> None:
        """Deserialize the JSON file at ``store_path`` into the in-memory store."""
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._denials = [DenialRecord.model_validate(d) for d in raw.get("denials", [])]
        self._rules = [PayerRuleEntry.model_validate(r) for r in raw.get("rules", [])]
        logger.debug(
            "PayerRulesIndex loaded %d denials, %d rules from %s",
            len(self._denials),
            len(self._rules),
            self._path,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return total number of stored objects (denials + rules)."""
        return len(self._denials) + len(self._rules)

    @property
    def denial_count(self) -> int:
        return len(self._denials)

    @property
    def rule_count(self) -> int:
        return len(self._rules)
