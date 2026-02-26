from __future__ import annotations

import pytest

from ems_pipeline.models import Entity
from ems_pipeline.nlp.context import apply_context


def _one(entities: list[Entity], transcript: str) -> list[Entity]:
    out = apply_context(entities, transcript)
    assert len(out) == len(entities)
    return out


@pytest.mark.parametrize(
    ("transcript", "entities", "expect"),
    [
        (
            "Denies chest pain but has SOB.",
            [Entity(type="SYMPTOM", text="chest pain"), Entity(type="SYMPTOM", text="SOB")],
            [
                {"negated": True, "uncertain": False},
                {"negated": False, "uncertain": False},
            ],
        ),
        (
            "Possible stroke.",
            [Entity(type="CONDITION", text="stroke")],
            [{"negated": False, "uncertain": True}],
        ),
        (
            "Negative for fever, chills, or cough.",
            [
                Entity(type="SYMPTOM", text="fever"),
                Entity(type="SYMPTOM", text="chills"),
                Entity(type="SYMPTOM", text="cough"),
            ],
            [{"negated": True, "uncertain": False}] * 3,
        ),
        (
            "Without nausea or vomiting.",
            [Entity(type="SYMPTOM", text="nausea"), Entity(type="SYMPTOM", text="vomiting")],
            [{"negated": True, "uncertain": False}] * 2,
        ),
        (
            "Question of sepsis.",
            [Entity(type="CONDITION", text="sepsis")],
            [{"negated": False, "uncertain": True}],
        ),
        (
            "Suspected overdose.",
            [Entity(type="CONDITION", text="overdose")],
            [{"negated": False, "uncertain": True}],
        ),
        (
            "Maybe allergic to penicillin.",
            [Entity(type="MEDICATION", text="penicillin")],
            [{"negated": False, "uncertain": True}],
        ),
        (
            "Caller says my father is having chest pain.",
            [Entity(type="SYMPTOM", text="chest pain")],
            [{"experiencer": "bystander"}],
        ),
        (
            "Caller says patient is having chest pain.",
            [Entity(type="SYMPTOM", text="chest pain")],
            [{"experiencer": "patient"}],
        ),
        (
            "Patient denies headache.",
            [Entity(type="SYMPTOM", text="headache")],
            [{"negated": True, "experiencer": "patient"}],
        ),
        (
            "No meds given prior to arrival.",
            [Entity(type="MEDICATION", text="meds")],
            [{"negated": True}],
        ),
        (
            "Chest pain started 2 hours ago; denies nausea.",
            [Entity(type="SYMPTOM", text="chest pain"), Entity(type="SYMPTOM", text="nausea")],
            [
                {"temporality_text": "2 hours ago"},
                {"negated": True},
            ],
        ),
    ],
)
def test_apply_context_rules(
    transcript: str, entities: list[Entity], expect: list[dict[str, object]]
) -> None:
    out = _one(entities, transcript)
    for entity, checks in zip(out, expect, strict=True):
        attrs = entity.attributes
        if "negated" in checks:
            assert attrs["negated"] is checks["negated"]
        if "uncertain" in checks:
            assert attrs["uncertain"] is checks["uncertain"]
        if "experiencer" in checks:
            assert attrs["experiencer"] == checks["experiencer"]
        if "temporality_text" in checks:
            temporality = attrs.get("temporality")
            assert temporality and temporality.get("text") == checks["temporality_text"]

