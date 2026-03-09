from __future__ import annotations

import re

from ems_pipeline.normalize import UnitIdRule, normalize_text


def test_normalize_examples() -> None:
    cases: list[tuple[str, str]] = [
        ("patient has sob", "patient has SOB"),
        ("S.O.B. with exertion", "SOB with exertion"),
        ("gcs 15 on arrival", "GCS 15 on arrival"),
        ("als/bLs intercept", "ALS/BLS intercept"),
        ("eta 5 minutes", "ETA 5 minutes"),
        ("started cpr", "started CPR"),
        ("stemi alert", "STEMI alert"),
        ("give narcan now", "give naloxone now"),
        ("Naloxone administered", "naloxone administered"),
        ("BP one fifty over ninety", "BP 150/90"),
        ("blood pressure 150 over 90", "blood pressure 150/90"),
        ("one hundred ten over seventy", "110/70"),
        ("pulse ox 92", "SpO2 92%"),
        ("PulseOx 88%", "SpO2 88%"),
        ("spo2 100", "SpO2 100%"),
        ("O2 sat 95", "SpO2 95%"),
        ("unit 12 responding", "UNIT12 responding"),
        ("Medic-3 en route", "MEDIC3 en route"),
    ]

    for original, expected in cases:
        normalized, mapping = normalize_text(original)
        assert normalized == expected
        assert mapping["original"] == original
        assert mapping["normalized"] == normalized
        assert isinstance(mapping["edits"], list)


def test_unit_id_rules_are_configurable() -> None:
    rule = UnitIdRule(
        name="car_call_sign",
        pattern=re.compile(r"(?<!\w)C(?P<number>\d{1,4})(?!\w)", flags=re.IGNORECASE),
        replacement=lambda m: f"CAR{m.group('number')}",
    )
    normalized, mapping = normalize_text("c12 staging", unit_id_rules=[rule])
    assert normalized == "CAR12 staging"
    assert mapping["original"] == "c12 staging"
    assert mapping["normalized"] == normalized
