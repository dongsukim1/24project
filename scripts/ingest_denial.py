"""Ingestion script for Index 2: Payer Rules RAG — denial records.

# STUB: no real denial data yet.  This script is fully wired and ready
# to process denial EOB files once a curated source is available.

Usage
-----
    python scripts/ingest_denial.py <input_file.json> --index <store.json>

    # Example
    python scripts/ingest_denial.py data/denials.json \\
        --index /var/ems/payer_rules_index.json

    # Associate ingested denials with a specific pipeline session
    python scripts/ingest_denial.py data/denials.json \\
        --index /var/ems/payer_rules_index.json \\
        --associate-encounter-id enc-2024-03-15-001

Expected input format
----------------------
A JSON array of objects.  Each object maps 1-to-1 with DenialRecord fields::

    [
      {
        "denial_id":      "550e8400-e29b-41d4-a716-446655440000",
        "encounter_id":   "enc-2024-03-15-001",
        "payer_id":       "BCBS_TX",
        "cpt_codes":      ["92950", "A0427"],
        "icd_codes":      ["I46.9"],
        "denial_reason":  "Medical necessity not established",
        "denial_code":    "CO-50",
        "policy_citation": "LCD L34462",
        "date":           "2024-03-15",
        "resolved":       false,
        "resolution":     null,
        "source":         "denial_eob"
      },
      ...
    ]

Required fields per record: denial_id, payer_id, denial_reason, source.
source must be one of: "denial_eob", "manual_entry", "appeal_outcome".

Exit codes
----------
    0  All records ingested successfully (or with some skipped — see summary).
    1  Fatal error (index not writable, input file not found, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest denial EOB records into the EMS pipeline Index 2 "
            "(Payer Rules) store."
        ),
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the JSON input file (array of DenialRecord objects).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        metavar="STORE_PATH",
        help=(
            "Path to the JSON index file "
            "(created if absent, appended if existing)."
        ),
    )
    parser.add_argument(
        "--associate-encounter-id",
        type=str,
        default=None,
        metavar="ENCOUNTER_ID",
        help=(
            "Optional encounter_id to stamp onto each ingested denial record, "
            "linking EOB denials to a pipeline session."
        ),
    )
    args = parser.parse_args(argv)

    input_path: Path = args.input_file
    index_path: Path = args.index
    associate_encounter_id: str | None = args.associate_encounter_id

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    # --- Load raw rows ---
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: failed to parse JSON input: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, list):
        print(
            f"ERROR: JSON input must be a top-level array, "
            f"got {type(data).__name__}",
            file=sys.stderr,
        )
        return 1

    # --- Import after arg parsing so script is importable without ems_pipeline installed ---
    try:
        from ems_pipeline.rag.payer_rules import DenialRecord, PayerRulesIndex
    except ImportError as exc:
        print(
            f"ERROR: ems_pipeline not installed or not on PYTHONPATH: {exc}",
            file=sys.stderr,
        )
        return 1

    index = PayerRulesIndex(index_path)
    ingested = 0
    failed = 0
    payer_counter: Counter[str] = Counter()

    for i, row in enumerate(data, start=1):
        try:
            record_payload = dict(row) if isinstance(row, dict) else row
            if isinstance(record_payload, dict) and associate_encounter_id:
                record_payload["encounter_id"] = associate_encounter_id

            record = DenialRecord.model_validate(record_payload)
            index.add_denial(record)
            payer_counter[record.payer_id] += 1
            ingested += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP row {i}: {exc} — row data: {row}", file=sys.stderr)
            failed += 1

    index.save()

    print(f"Ingestion complete: {ingested} records ingested, {failed} failed validation.")
    print(f"Index written to: {index_path}")
    if payer_counter:
        print("Payer distribution:")
        for payer_id, count in sorted(payer_counter.items()):
            print(f"  {payer_id}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
