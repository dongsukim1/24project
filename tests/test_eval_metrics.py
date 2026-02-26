from __future__ import annotations

import pytest

from ems_pipeline.eval.harness import entity_prf_by_type, key_term_coverage


def test_entity_prf_by_type_dedupes_and_scores() -> None:
    gold = [
        {
            "id": "ex1",
            "transcript": "patient has SOB, got naloxone",
            "entities": [
                {"type": "SYMPTOM", "text": "SOB"},
                {"type": "MEDICATION", "text": "naloxone"},
            ],
        }
    ]
    pred_by_id = {
        "ex1": [
            {"type": "SYMPTOM", "text": "sob"},
            {"type": "SYMPTOM", "text": "SOB"},
            {"type": "MEDICATION", "text": "naloxone"},
            {"type": "MEDICATION", "text": "aspirin"},
        ]
    }

    report = entity_prf_by_type(gold, pred_by_id)
    by_type = report["by_type"]

    assert by_type["SYMPTOM"]["tp"] == 1
    assert by_type["SYMPTOM"]["fp"] == 0
    assert by_type["SYMPTOM"]["fn"] == 0
    assert by_type["SYMPTOM"]["precision"] == 1.0
    assert by_type["SYMPTOM"]["recall"] == 1.0
    assert by_type["SYMPTOM"]["f1"] == 1.0

    assert by_type["MEDICATION"]["tp"] == 1
    assert by_type["MEDICATION"]["fp"] == 1
    assert by_type["MEDICATION"]["fn"] == 0
    assert by_type["MEDICATION"]["precision"] == pytest.approx(0.5)
    assert by_type["MEDICATION"]["recall"] == 1.0
    assert by_type["MEDICATION"]["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)

    micro = report["micro"]
    assert micro["tp"] == 2
    assert micro["fp"] == 1
    assert micro["fn"] == 0
    assert micro["precision"] == pytest.approx(2 / 3)
    assert micro["recall"] == 1.0


def test_key_term_coverage_counts_present_and_covered() -> None:
    gold = [
        {
            "id": "ex1",
            "transcript": "patient has SOB; started CPR",
            "entities": [],
        }
    ]
    pred_by_id = {"ex1": [{"type": "PROCEDURE", "text": "CPR"}]}

    terms = {"SOB": ["SOB", "sob"], "CPR": ["CPR", "cpr"]}
    report = key_term_coverage(gold, pred_by_id, terms=terms)

    assert report["present"] == 2
    assert report["covered"] == 1
    assert report["overall"] == pytest.approx(0.5)
    assert report["by_term"]["SOB"]["present"] == 1
    assert report["by_term"]["SOB"]["covered"] == 0
    assert report["by_term"]["CPR"]["present"] == 1
    assert report["by_term"]["CPR"]["covered"] == 1

