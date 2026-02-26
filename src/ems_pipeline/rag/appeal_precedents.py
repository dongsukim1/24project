"""Index 4: Appeal precedent retrieval on top of shared denial records.

This module intentionally wraps :class:`ems_pipeline.rag.payer_rules.PayerRulesIndex`
instead of introducing a separate Index 4 store. Appeal outcomes are written to
``DenialRecord.appeal_attempts`` so Index 2 and Index 4 share one source of truth.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ems_pipeline.rag.payer_rules import AppealAttempt, PayerRulesIndex


class AppealPrecedentQuery(BaseModel):
    """Query parameters for appeal precedent retrieval."""

    model_config = ConfigDict(extra="forbid")

    payer_id: str | None = None
    denial_reason: str | None = None
    cpt_codes: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1)


def _parse_iso_timestamp(timestamp: str) -> datetime:
    """Best-effort ISO parser used for descending sort."""
    value = timestamp.strip()
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


class AppealPrecedentsIndex:
    """Read/write appeal precedents by wrapping an existing ``PayerRulesIndex``."""

    def __init__(self, payer_rules_index: PayerRulesIndex) -> None:
        self._payer_rules_index = payer_rules_index

    def retrieve(self, query: AppealPrecedentQuery) -> list[dict[str, Any]]:
        """Return non-pending appeal outcomes for denials matching the query."""
        matching_denials = self._payer_rules_index.get_denial_patterns(
            payer_id=query.payer_id,
            denial_reason=query.denial_reason,
        )

        if query.cpt_codes:
            prefixes = [code.upper() for code in query.cpt_codes if code]
            matching_denials = [
                denial
                for denial in matching_denials
                if any(
                    cpt.upper().startswith(prefix)
                    for cpt in denial.cpt_codes
                    for prefix in prefixes
                )
            ]

        precedents: list[dict[str, Any]] = []
        for denial in matching_denials:
            codes = list(dict.fromkeys([*denial.cpt_codes, *denial.icd_codes]))
            for attempt in denial.appeal_attempts:
                if attempt.outcome == "pending":
                    continue
                precedents.append(
                    {
                        "strategy": attempt.strategy,
                        "outcome": attempt.outcome,
                        "payer_id": denial.payer_id,
                        "denial_reason": denial.denial_reason,
                        "codes": codes,
                        "notes": attempt.notes,
                        "timestamp": attempt.timestamp,
                    }
                )

        precedents.sort(
            key=lambda precedent: _parse_iso_timestamp(precedent["timestamp"]),
            reverse=True,
        )
        return precedents[:query.top_k]

    def record_outcome(self, denial_id: str, attempt: AppealAttempt) -> None:
        """Append a new appeal attempt to a denial and persist the shared store."""
        denial = self._payer_rules_index.get_denial_by_id(denial_id)
        if denial is None:
            denial = next(
                (
                    record
                    for record in self._payer_rules_index.get_denial_patterns()
                    if record.encounter_id == denial_id
                ),
                None,
            )
        if denial is None:
            raise KeyError(f"Denial not found: {denial_id}")
        denial.appeal_attempts.append(attempt)

        if attempt.outcome == "success":
            denial.resolved = True
            if denial.resolution is None:
                denial.resolution = "Successfully appealed"

        self._payer_rules_index.save()
