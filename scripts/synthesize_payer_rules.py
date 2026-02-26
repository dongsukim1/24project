"""Synthesize PayerRuleEntry objects inductively from accumulated DenialRecords.

# STUB: simple grouping / frequency logic — replace with LLM-assisted
# rule extraction when enough real denial data exists.

Architecture principle
----------------------
Every denial is a data point.  When the same (payer_id, denial_reason,
frozen codes_affected) combination appears in >= 2 distinct denial records,
we have enough evidence to synthesize a PayerRuleEntry with a confidence
score proportional to the count.

Confidence formula
------------------
    confidence = min(1.0, count / CONFIDENCE_DENOMINATOR)

CONFIDENCE_DENOMINATOR defaults to 10 (i.e. 10 identical denials → 1.0).
Adjust via --confidence-denominator.

Usage
-----
    python scripts/synthesize_payer_rules.py --index <store.json>

    # With custom thresholds
    python scripts/synthesize_payer_rules.py \\
        --index /var/ems/payer_rules_index.json \\
        --min-denials 3 \\
        --confidence-denominator 20

    # Dry-run (print synthesized rules without writing)
    python scripts/synthesize_payer_rules.py --index store.json --dry-run

Exit codes
----------
    0  Synthesis complete (0 rules synthesized is not an error).
    1  Fatal error (index not found, not readable, etc.).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

# Default thresholds.
_DEFAULT_MIN_DENIALS = 2
_DEFAULT_CONFIDENCE_DENOMINATOR = 10
_SUCCESS_CONFIDENCE_THRESHOLD = 0.7


def _has_successful_appeal(record: Any) -> bool:
    """Return True when a denial has at least one successful appeal signal."""
    if record.resolved:
        return True
    return any(attempt.outcome == "success" for attempt in record.appeal_attempts)


def _infer_rule_type(denial_reason: str) -> str:
    """Heuristically map a denial reason string to a rule_type Literal.

    # STUB: keyword heuristics — replace with classifier or LLM when live.
    """
    lower = denial_reason.lower()
    if any(k in lower for k in ("bundl", "unbundl", "duplicate")):
        return "bundling"
    if any(k in lower for k in ("auth", "prior auth", "referral", "pre-auth")):
        return "authorization"
    if any(k in lower for k in ("document", "record", "note", "report")):
        return "documentation"
    if any(k in lower for k in ("not covered", "non-covered", "coverage", "benefit")):
        return "coverage"
    # Default: medical_necessity (most common EMS denial category)
    return "medical_necessity"


def synthesize(
    index_path: Path,
    min_denials: int = _DEFAULT_MIN_DENIALS,
    confidence_denominator: float = _DEFAULT_CONFIDENCE_DENOMINATOR,
    dry_run: bool = False,
) -> int:
    """Core synthesis logic.  Returns number of new rules written (or that would be written)."""
    try:
        from ems_pipeline.rag.payer_rules import DenialRecord, PayerRuleEntry, PayerRulesIndex
    except ImportError as exc:
        print(
            f"ERROR: ems_pipeline not installed or not on PYTHONPATH: {exc}",
            file=sys.stderr,
        )
        return -1

    if not index_path.exists():
        print(f"ERROR: index file not found: {index_path}", file=sys.stderr)
        return -1

    index = PayerRulesIndex(index_path)

    if index.denial_count == 0:
        print("No denial records in index — nothing to synthesize.")
        return 0

    # Group denials by (payer_id, denial_reason, sorted_codes).
    # sorted_codes is a tuple so it's hashable and order-independent.
    groups: dict[tuple[str, str, tuple[str, ...]], list[DenialRecord]] = defaultdict(list)
    for record in index._denials:
        all_codes = tuple(sorted(c.upper() for c in record.cpt_codes + record.icd_codes))
        key = (record.payer_id, record.denial_reason, all_codes)
        groups[key].append(record)

    # Synthesize a rule for each group that meets the threshold.
    new_rules: list[PayerRuleEntry] = []
    successful_denial_ids_by_rule: dict[str, list[str]] = {}
    for (payer_id, denial_reason, codes_tuple), records in groups.items():
        if len(records) < min_denials:
            continue

        count = len(records)
        confidence = min(1.0, count / confidence_denominator)
        denial_ids = [r.denial_id for r in records]
        successful_denial_ids = [r.denial_id for r in records if _has_successful_appeal(r)]
        codes_affected = list(codes_tuple)
        successfully_appealed = (
            confidence >= _SUCCESS_CONFIDENCE_THRESHOLD and bool(successful_denial_ids)
        )

        rule = PayerRuleEntry(
            rule_id=str(uuid.uuid4()),
            payer_id=payer_id,
            rule_type=_infer_rule_type(denial_reason),
            codes_affected=codes_affected,
            description=(
                f"Synthesized from {count} denial record(s): {denial_reason}"
                f"{' (successfully appealed)' if successfully_appealed else ''}"
            ),
            common_denial_reason=denial_reason,
            required_documentation=[],  # STUB: extract from denial notes when live
            derived_from_denial_ids=denial_ids,
            confidence=confidence,
            source="synthesize_payer_rules",
        )
        new_rules.append(rule)
        successful_denial_ids_by_rule[rule.rule_id] = successful_denial_ids

    if not new_rules:
        print(
            f"No groups met the minimum threshold of {min_denials} denial(s). "
            "0 rules synthesized."
        )
        return 0

    if dry_run:
        print(f"[DRY RUN] Would synthesize {len(new_rules)} rule(s):")
        for r in new_rules:
            success_ids = successful_denial_ids_by_rule.get(r.rule_id, [])
            print(
                f"  payer={r.payer_id} type={r.rule_type} "
                f"codes={r.codes_affected} confidence={r.confidence:.2f} "
                f"derived_from={len(r.derived_from_denial_ids)} denial(s)"
            )
            if success_ids:
                print(
                    "    successful_appeal_denials="
                    + ", ".join(success_ids)
                )
        return len(new_rules)

    for rule in new_rules:
        index.add_rule(rule)

    index.save()

    print(f"Synthesis complete: {len(new_rules)} rule(s) synthesized and written.")
    print(f"Index written to: {index_path}")
    for r in new_rules:
        success_ids = successful_denial_ids_by_rule.get(r.rule_id, [])
        print(
            f"  [{r.rule_type}] payer={r.payer_id} "
            f"codes={r.codes_affected} confidence={r.confidence:.2f}"
        )
        if success_ids:
            print("    successful_appeal_denials=" + ", ".join(success_ids))
    return len(new_rules)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize PayerRuleEntry objects from denial records in the "
            "EMS pipeline Index 2 store."
        ),
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        metavar="STORE_PATH",
        help="Path to the JSON index file (must already exist with denial records).",
    )
    parser.add_argument(
        "--min-denials",
        type=int,
        default=_DEFAULT_MIN_DENIALS,
        metavar="N",
        help=(
            "Minimum number of matching denials to synthesize a rule "
            f"(default: {_DEFAULT_MIN_DENIALS})."
        ),
    )
    parser.add_argument(
        "--confidence-denominator",
        type=float,
        default=_DEFAULT_CONFIDENCE_DENOMINATOR,
        metavar="D",
        help=(
            f"Denominator for confidence formula: min(1.0, count/D) "
            f"(default: {_DEFAULT_CONFIDENCE_DENOMINATOR})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print synthesized rules without writing to the index.",
    )
    args = parser.parse_args(argv)

    result = synthesize(
        index_path=args.index,
        min_denials=args.min_denials,
        confidence_denominator=args.confidence_denominator,
        dry_run=args.dry_run,
    )
    return 0 if result >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
