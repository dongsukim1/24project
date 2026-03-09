"""EMS Pipeline interoperability exporters.

Each exporter accepts a ``CanonicalClaim`` (v1.0), validates the fields
required by the target standard, and returns an ``ExportResult`` containing:

- ``data``:              The structured output (dict / JSON-serialisable).
- ``missing_required``:  Field paths that *must* be present but are absent.
- ``missing_optional``:  Field paths that are recommended but absent.
- ``warnings``:          Non-fatal observations about data quality.

Available exporters:
    ``ems_pipeline.exporters.nemsis``   — NEMSIS v3.5 data-element mapping
    ``ems_pipeline.exporters.x12_837``  — X12 837P professional claim mapping
    ``ems_pipeline.exporters.fhir``     — FHIR R4 Bundle (Patient / Coverage /
                                          Encounter / Claim / Observations)
    ``ems_pipeline.exporters.coverage`` — Cross-format coverage report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExportResult:
    """Container for an exporter's output.

    Attributes:
        format:           Short format name ("nemsis", "x12_837", "fhir").
        data:             The exported payload (JSON-serialisable dict).
        missing_required: Dotted field paths that are required but absent.
        missing_optional: Dotted field paths that are useful but absent.
        warnings:         Non-fatal diagnostic messages.
    """

    format: str
    data: dict[str, Any]
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when no required fields are missing."""
        return len(self.missing_required) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "is_valid": self.is_valid,
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
            "warnings": self.warnings,
            "data": self.data,
        }


def fmt_dt(dt: Any) -> str | None:
    """Format a datetime as ISO-8601, or return None if absent."""
    if dt is None:
        return None
    return dt.isoformat()


def fmt_date(dt: Any, *, separator: bool = True) -> str | None:
    """Format a date as YYYY-MM-DD (default) or YYYYMMDD (separator=False)."""
    if dt is None:
        return None
    fmt = "%Y-%m-%d" if separator else "%Y%m%d"
    try:
        return dt.strftime(fmt)
    except AttributeError:
        s = str(dt)
        if separator:
            return s[:10] if len(s) >= 10 else s
        return s[:10].replace("-", "")


def fmt_name(full_name: str | None) -> tuple[str | None, str | None]:
    """Split 'First Last' into (last, first)."""
    if not full_name:
        return None, None
    parts = full_name.strip().split(" ", 1)
    if len(parts) == 2:
        return parts[1], parts[0]
    return parts[0], None


__all__ = ["ExportResult", "fmt_dt", "fmt_date", "fmt_name"]
