from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml


@dataclass(frozen=True)
class UnitIdRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str | Callable[[re.Match[str]], str]


Number = int


_ONES: dict[str, Number] = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_TENS: dict[str, Number] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _words_to_int(words: str) -> int | None:
    tokens = [t for t in re.split(r"[-\s]+", words.lower().strip()) if t]
    if not tokens:
        return None

    total = 0
    current = 0
    saw_any = False

    for token in tokens:
        if token in _ONES:
            current += _ONES[token]
            saw_any = True
            continue
        if token in _TENS:
            current += _TENS[token]
            saw_any = True
            continue
        if token == "hundred":
            if current == 0:
                current = 1
            current *= 100
            saw_any = True
            continue
        return None

    if not saw_any:
        return None

    total += current
    return total


def _words_bp_to_int(words: str) -> int | None:
    """Handle common EMS BP pronunciations like 'one fifty' => 150."""
    w = words.lower().strip()
    tokens = [t for t in re.split(r"[-\s]+", w) if t]
    if len(tokens) == 2 and tokens[0] in ("one", "two") and (
        tokens[1] in _TENS or tokens[1] in _ONES
    ):
        base = 100 if tokens[0] == "one" else 200
        rest = _TENS.get(tokens[1], _ONES.get(tokens[1]))
        if rest is None:
            return None
        return base + rest

    if (
        len(tokens) == 3
        and tokens[0] in ("one", "two")
        and tokens[1] in _TENS
        and tokens[2] in _ONES
    ):
        base = 100 if tokens[0] == "one" else 200
        return base + _TENS[tokens[1]] + _ONES[tokens[2]]

    v = _words_to_int(w)
    if v is not None:
        return v

    return None


def _load_lexicon(lexicon_resource: str = "ems_lexicon.yml") -> dict[str, Any]:
    data = resources.files("ems_pipeline.resources").joinpath(lexicon_resource).read_text(
        encoding="utf-8"
    )
    parsed = yaml.safe_load(data) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Lexicon {lexicon_resource} must be a mapping")
    return parsed


def _apply_regex_subs(
    text: str,
    *,
    kind: str,
    name: str,
    pattern: re.Pattern[str],
    repl: str | Callable[[re.Match[str]], str],
    edits: list[dict[str, Any]],
) -> str:
    matches: list[dict[str, Any]] = []

    def _repl(m: re.Match[str]) -> str:
        before = m.group(0)
        after = repl(m) if callable(repl) else m.expand(repl)
        matches.append({"match": before, "replacement": after, "start": m.start(), "end": m.end()})
        return after

    new_text, count = pattern.subn(_repl, text)
    if count:
        edits.append({"kind": kind, "name": name, "count": count, "matches": matches})
    return new_text


def _compile_lexicon_subs(lexicon: dict[str, Any]) -> list[tuple[str, re.Pattern[str], str]]:
    subs: list[tuple[str, re.Pattern[str], str]] = []

    def add_group(group_name: str) -> None:
        group = lexicon.get(group_name, {}) or {}
        if not isinstance(group, dict):
            return
        for canonical, variants in group.items():
            if not isinstance(canonical, str) or not isinstance(variants, list):
                continue
            for variant in variants:
                if not isinstance(variant, str) or not variant.strip():
                    continue
                # Use non-word lookarounds (instead of \b) so variants like "s.o.b." match.
                pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", flags=re.IGNORECASE)
                subs.append((f"{group_name}:{canonical}", pattern, canonical))

    add_group("acronyms")
    add_group("medications")
    return subs


_NUM_WORD_RE = r"(?:{})".format("|".join(
    sorted({*list(_ONES.keys()), *list(_TENS.keys()), "hundred"}, key=len, reverse=True)
))


def _compile_unit_id_rules(lexicon: dict[str, Any]) -> list[UnitIdRule]:
    rules_raw = lexicon.get("unit_id_rules", []) or []
    if not isinstance(rules_raw, list):
        return []

    rules: list[UnitIdRule] = []
    for item in rules_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "unit_id_rule")
        pattern_s = item.get("pattern")
        replacement = item.get("replacement")
        if not isinstance(pattern_s, str) or not isinstance(replacement, str):
            continue
        pattern = re.compile(pattern_s, flags=re.IGNORECASE)

        def _make_repl(template: str) -> Callable[[re.Match[str]], str]:
            def _repl(m: re.Match[str]) -> str:
                gd = {k: (v or "") for k, v in m.groupdict().items()}
                if "prefix" in gd:
                    gd["prefix_upper"] = str(gd["prefix"]).upper()
                return template.format(**gd)

            return _repl

        rules.append(UnitIdRule(name=name, pattern=pattern, replacement=_make_repl(replacement)))
    return rules


def normalize_text(
    text: str,
    *,
    unit_id_rules: list[UnitIdRule] | None = None,
    lexicon_resource: str = "ems_lexicon.yml",
) -> tuple[str, dict[str, Any]]:
    """Normalize EMS transcript text.

    Returns (normalized_text, mapping) where mapping contains:
      - original: original input text
      - normalized: normalized text (same as first return value)
      - edits: list of edit summaries in application order
    """

    original = text
    normalized = text
    edits: list[dict[str, Any]] = []

    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized != original:
        edits.append({"kind": "whitespace", "name": "collapse_spaces", "count": 1})

    lexicon = _load_lexicon(lexicon_resource)

    for name, pattern, replacement in _compile_lexicon_subs(lexicon):
        normalized = _apply_regex_subs(
            normalized, kind="lexicon", name=name, pattern=pattern, repl=replacement, edits=edits
        )

    # Vitals: SpO2
    spo2_pattern = re.compile(
        r"\b(pulse\s*ox(?:imeter)?|pulseox|spo2|o2\s*sat(?:s|uration)?|oxygen\s*sat(?:s|uration)?)\s*(?:is\s*)?(?P<value>\d{2,3})\s*%?(?!\w)",
        flags=re.IGNORECASE,
    )

    def _spo2_repl(m: re.Match[str]) -> str:
        return f"SpO2 {m.group('value')}%"

    normalized = _apply_regex_subs(
        normalized, kind="vitals", name="spo2", pattern=spo2_pattern, repl=_spo2_repl, edits=edits
    )

    # Vitals: BP as digits ("150 over 90" => "150/90")
    bp_digits_pattern = re.compile(
        r"\b(?P<sys>\d{2,3})\s*(?:/|over)\s*(?P<dia>\d{2,3})\b", flags=re.IGNORECASE
    )

    def _bp_digits_repl(m: re.Match[str]) -> str:
        return f"{m.group('sys')}/{m.group('dia')}"

    normalized = _apply_regex_subs(
        normalized,
        kind="vitals",
        name="bp_digits",
        pattern=bp_digits_pattern,
        repl=_bp_digits_repl,
        edits=edits,
    )

    # Vitals: BP as words ("one fifty over ninety" => "150/90")
    bp_words_pattern = re.compile(
        rf"(?<!\w)(?P<sys_words>{_NUM_WORD_RE}(?:[ -]{_NUM_WORD_RE})*)\s+over\s+"
        rf"(?P<dia_words>{_NUM_WORD_RE}(?:[ -]{_NUM_WORD_RE})*)(?!\w)",
        flags=re.IGNORECASE,
    )

    def _bp_words_repl(m: re.Match[str]) -> str:
        sys_v = _words_bp_to_int(m.group("sys_words"))
        dia_v = _words_bp_to_int(m.group("dia_words"))
        if sys_v is None or dia_v is None:
            return m.group(0)
        return f"{sys_v}/{dia_v}"

    normalized = _apply_regex_subs(
        normalized,
        kind="vitals",
        name="bp_words",
        pattern=bp_words_pattern,
        repl=_bp_words_repl,
        edits=edits,
    )

    if unit_id_rules is None:
        unit_id_rules = _compile_unit_id_rules(lexicon)

    for rule in unit_id_rules:
        pattern = rule.pattern
        # Be tolerant of over-escaped regexes (common when authoring patterns in YAML or raw
        # strings).
        if "\\\\" in pattern.pattern:
            pattern = re.compile(pattern.pattern.replace("\\\\", "\\"), flags=pattern.flags)
        normalized = _apply_regex_subs(
            normalized,
            kind="unit_id",
            name=rule.name,
            pattern=pattern,
            repl=rule.replacement,
            edits=edits,
        )

    mapping: dict[str, Any] = {"original": original, "normalized": normalized, "edits": edits}
    return normalized, mapping
