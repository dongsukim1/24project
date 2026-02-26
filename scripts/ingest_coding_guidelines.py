"""Ingestion script for Index 3: Coding Guidelines RAG.

# STUB: no real ICD-10/CPT data yet.  This script is fully wired and ready
# to process data files once a curated source is available.

Usage
-----
    python scripts/ingest_coding_guidelines.py <input_file> --index <store.json>

    # JSON input
    python scripts/ingest_coding_guidelines.py data/guidelines.json \\
        --index /var/ems/coding_guidelines_index.json

    # CSV input
    python scripts/ingest_coding_guidelines.py data/guidelines.csv \\
        --index /var/ems/coding_guidelines_index.json --format csv

Expected input formats
----------------------
JSON
    A JSON array of objects.  Each object maps 1-to-1 with CodingGuidelineEntry
    fields::

        [
          {
            "code": "R07.9",
            "code_type": "ICD10",
            "description": "Chest pain, unspecified",
            "specialty": "emergency_medicine",
            "body_system": "cardiovascular",
            "required_documentation": ["onset", "character", "severity"],
            "exclusion_codes": ["R07.1", "R07.2"],
            "notes": null,
            "source": "ICD-10-CM 2024"
          },
          ...
        ]

CSV
    Columns (header row required)::

        code, code_type, description, specialty, body_system,
        required_documentation, exclusion_codes, notes, source

    - ``required_documentation`` and ``exclusion_codes`` are pipe-separated
      strings: ``"onset|character|severity"`` → ``["onset", "character", "severity"]``.
    - Empty pipe-separated fields are treated as empty lists.
    - ``specialty``, ``body_system``, and ``notes`` may be blank (stored as None).

Exit codes
----------
    0  All rows ingested successfully (or with some skipped — see summary).
    1  Fatal error (index not writable, input file not found, etc.).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _parse_pipe_list(value: str) -> list[str]:
    """Split a pipe-separated string into a list, dropping blank entries."""
    return [v.strip() for v in value.split("|") if v.strip()]


def _none_if_blank(value: str) -> str | None:
    stripped = value.strip()
    return stripped if stripped else None


def _load_json_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"JSON input must be a top-level array, got {type(data).__name__}")
    return data


def _load_csv_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "code": row.get("code", "").strip(),
                    "code_type": row.get("code_type", "").strip(),
                    "description": row.get("description", "").strip(),
                    "specialty": _none_if_blank(row.get("specialty", "")),
                    "body_system": _none_if_blank(row.get("body_system", "")),
                    "required_documentation": _parse_pipe_list(
                        row.get("required_documentation", "")
                    ),
                    "exclusion_codes": _parse_pipe_list(
                        row.get("exclusion_codes", "")
                    ),
                    "notes": _none_if_blank(row.get("notes", "")),
                    "source": row.get("source", "").strip(),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest ICD-10/CPT coding guidelines into the EMS pipeline Index 3 store.",
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input file (JSON array or CSV).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        metavar="STORE_PATH",
        help="Path to the JSON index file (created if absent, appended if existing).",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default=None,
        help="Input format.  Inferred from file extension if not provided.",
    )
    args = parser.parse_args(argv)

    input_path: Path = args.input_file
    index_path: Path = args.index

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    # Infer format from extension if not given explicitly.
    fmt: str = args.format or input_path.suffix.lstrip(".").lower()
    if fmt not in ("json", "csv"):
        print(
            f"ERROR: cannot infer format from extension '{input_path.suffix}'. "
            "Use --format json or --format csv.",
            file=sys.stderr,
        )
        return 1

    # --- Load raw rows ---
    try:
        rows = _load_json_rows(input_path) if fmt == "json" else _load_csv_rows(input_path)
    except Exception as exc:
        print(f"ERROR: failed to load input: {exc}", file=sys.stderr)
        return 1

    # --- Import after arg parsing so script is importable without ems_pipeline installed ---
    try:
        from ems_pipeline.rag.coding_guidelines import CodingGuidelineEntry, CodingGuidelinesIndex
    except ImportError as exc:
        print(
            f"ERROR: ems_pipeline not installed or not on PYTHONPATH: {exc}",
            file=sys.stderr,
        )
        return 1

    index = CodingGuidelinesIndex(index_path)
    ingested = 0
    failed = 0

    for i, row in enumerate(rows, start=1):
        try:
            entry = CodingGuidelineEntry.model_validate(row)
            index.add_entry(entry)
            ingested += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP row {i}: {exc} — row data: {row}", file=sys.stderr)
            failed += 1

    index.save()

    print(f"Ingestion complete: {ingested} entries ingested, {failed} failed validation.")
    print(f"Index written to: {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
