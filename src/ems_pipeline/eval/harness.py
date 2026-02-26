from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EntityKey:
    entity_type: str
    value: str


def _canonicalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _entity_to_key(entity: dict[str, Any]) -> EntityKey:
    entity_type = str(entity.get("type") or "")
    if not entity_type:
        raise ValueError("Entity is missing required field 'type'")

    value = entity.get("normalized") or entity.get("text") or ""
    value = str(value)
    if not value.strip():
        raise ValueError("Entity is missing required field 'text' (or 'normalized')")

    return EntityKey(entity_type=entity_type, value=_canonicalize_text(value))


def load_gold(path: str | Path) -> list[dict[str, Any]]:
    """Load a gold dataset JSON file.

    Expected shape:
      { "examples": [ { "id": str, "transcript": str, "entities": [ {type,text,...} ] } ] }
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = data.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Gold JSON must contain top-level key 'examples' as a list")
    return examples


def load_predictions(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load predicted entities JSON.

    Supported shapes:
      - { "examples": [ { "id": str, "entities": [ ... ] } ] }
      - { "<id>": [ ... ], ... }
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "examples" in data:
        examples = data.get("examples")
        if not isinstance(examples, list):
            raise ValueError("Pred JSON 'examples' must be a list")
        out: dict[str, list[dict[str, Any]]] = {}
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            ex_id = str(ex.get("id") or "")
            if not ex_id:
                continue
            entities = ex.get("entities") or ex.get("pred_entities") or []
            if not isinstance(entities, list):
                raise ValueError(f"Pred JSON example {ex_id!r} has non-list 'entities'")
            out[ex_id] = [e for e in entities if isinstance(e, dict)]
        return out

    if isinstance(data, dict):
        out2: dict[str, list[dict[str, Any]]] = {}
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, list):
                raise ValueError(f"Pred JSON id {k!r} must map to a list of entities")
            out2[k] = [e for e in v if isinstance(e, dict)]
        return out2

    raise ValueError("Pred JSON must be an object")


def entity_prf_by_type(
    gold_examples: list[dict[str, Any]],
    pred_by_id: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compute precision/recall/F1 for entity mentions, grouped by entity type.

    Matching is exact after canonicalization, and de-duplicated per (type, value) per example.
    """

    by_type: dict[str, dict[str, int]] = {}

    for ex in gold_examples:
        if not isinstance(ex, dict):
            continue
        ex_id = str(ex.get("id") or "")
        gold_entities = ex.get("entities") or ex.get("gold_entities") or []
        if not isinstance(gold_entities, list):
            raise ValueError(f"Gold example {ex_id!r} has non-list 'entities'")

        pred_entities = pred_by_id.get(ex_id, [])

        gold_keys = {_entity_to_key(e) for e in gold_entities if isinstance(e, dict)}
        pred_keys = {_entity_to_key(e) for e in pred_entities if isinstance(e, dict)}

        # Tally by type.
        types = {k.entity_type for k in gold_keys} | {k.entity_type for k in pred_keys}
        for t in types:
            gold_t = {k for k in gold_keys if k.entity_type == t}
            pred_t = {k for k in pred_keys if k.entity_type == t}
            tp = len(gold_t & pred_t)
            fp = len(pred_t - gold_t)
            fn = len(gold_t - pred_t)
            stats = by_type.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
            stats["tp"] += tp
            stats["fp"] += fp
            stats["fn"] += fn

    def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    per_type: dict[str, Any] = {}
    micro_tp = micro_fp = micro_fn = 0
    for t, stats in sorted(by_type.items(), key=lambda kv: kv[0]):
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        per_type[t] = {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}

    return {
        "by_type": per_type,
        "micro": {
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            **prf(micro_tp, micro_fp, micro_fn),
        },
    }


def _load_ems_key_terms(lexicon_resource: str = "ems_lexicon.yml") -> dict[str, list[str]]:
    """Return {canonical_term: [variants...]} for a small set of key EMS terms."""

    raw = resources.files("ems_pipeline.resources").joinpath(lexicon_resource).read_text(
        encoding="utf-8"
    )
    lexicon = yaml.safe_load(raw) or {}
    if not isinstance(lexicon, dict):
        return {}

    terms: dict[str, list[str]] = {}
    for group_name in ("acronyms", "medications"):
        group = lexicon.get(group_name) or {}
        if not isinstance(group, dict):
            continue
        for canonical, variants in group.items():
            if not isinstance(canonical, str) or not canonical.strip():
                continue
            out_variants: list[str] = [canonical]
            if isinstance(variants, list):
                out_variants.extend([v for v in variants if isinstance(v, str) and v.strip()])
            # De-dupe, preserve order.
            deduped: list[str] = []
            seen: set[str] = set()
            for v in out_variants:
                key = v.lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(v)
            terms[canonical] = deduped

    # Common vitals not present in lexicon.
    terms.setdefault("SpO2", ["SpO2", "pulse ox", "pulseox", "o2 sat", "oxygen sat"])
    terms.setdefault("BP", ["BP", "blood pressure"])
    return terms


def key_term_coverage(
    gold_examples: list[dict[str, Any]],
    pred_by_id: dict[str, list[dict[str, Any]]],
    *,
    terms: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Compute coverage of key EMS terms mentioned in the transcript text.

    A term is "present" if any of its variants appears in the transcript. It is "covered" if any
    predicted entity's text/normalized matches one of the variants (substring match after
    canonicalization).
    """

    if terms is None:
        terms = _load_ems_key_terms()

    term_patterns: dict[str, list[re.Pattern[str]]] = {}
    for canonical, variants in terms.items():
        patterns: list[re.Pattern[str]] = []
        for v in variants:
            v = v.strip()
            if not v:
                continue
            patterns.append(re.compile(rf"(?<!\\w){re.escape(v)}(?!\\w)", flags=re.IGNORECASE))
        if canonical == "SpO2":
            patterns.append(re.compile(r"\\bspo2\\b", flags=re.IGNORECASE))
            patterns.append(re.compile(r"\\bpulse\\s*ox\\b", flags=re.IGNORECASE))
            patterns.append(re.compile(r"\\boxygen\\s*sat", flags=re.IGNORECASE))
        if canonical == "BP":
            patterns.append(
                re.compile(
                    r"\\b\\d{2,3}\\s*(?:/|over)\\s*\\d{2,3}\\b",
                    flags=re.IGNORECASE,
                )
            )
        term_patterns[canonical] = patterns

    total_present = 0
    total_covered = 0
    per_term: dict[str, dict[str, Any]] = {t: {"present": 0, "covered": 0} for t in term_patterns}

    for ex in gold_examples:
        if not isinstance(ex, dict):
            continue
        ex_id = str(ex.get("id") or "")
        transcript = str(ex.get("transcript") or "")
        pred_entities = pred_by_id.get(ex_id, [])
        pred_values = {_entity_to_key(e).value for e in pred_entities if isinstance(e, dict)}

        for term, patterns in term_patterns.items():
            is_present = any(p.search(transcript) for p in patterns)
            if not is_present:
                continue
            total_present += 1
            per_term[term]["present"] += 1

            term_variants = terms.get(term, [term])
            variant_keys = {_canonicalize_text(v) for v in term_variants if v.strip()}
            covered = any(any(vk in pv for vk in variant_keys) for pv in pred_values)
            if covered:
                total_covered += 1
                per_term[term]["covered"] += 1

    overall = (total_covered / total_present) if total_present else 0.0
    for _term, stats in per_term.items():
        p = stats["present"]
        c = stats["covered"]
        stats["coverage"] = (c / p) if p else 0.0

    return {
        "overall": overall,
        "present": total_present,
        "covered": total_covered,
        "by_term": per_term,
    }


def evaluate(gold_path: str | Path, pred_path: str | Path) -> dict[str, Any]:
    gold_examples = load_gold(gold_path)
    pred_by_id = load_predictions(pred_path)
    return {
        "entity_metrics": entity_prf_by_type(gold_examples, pred_by_id),
        "key_term_coverage": key_term_coverage(gold_examples, pred_by_id),
    }


def format_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
