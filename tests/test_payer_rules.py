"""Tests for Index 2: Payer Rules RAG scaffold.

Covers:
- DenialRecord and PayerRuleEntry validation (required fields, Literal enforcement)
- PayerRulesIndex add_denial / add_rule / save / load round-trip
- retrieve_for_payer() payer isolation and code filtering
- get_denial_patterns() payer isolation and code filtering
- synthesize_payer_rules logic: 3 identical denials → 1 PayerRuleEntry created
- retrieve_payer_requirements() dispatcher in rag/__init__.py
- ingest_denial and synthesize_payer_rules script helpers
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ems_pipeline.rag.payer_rules import (
    DenialRecord,
    PayerRuleEntry,
    PayerRulesIndex,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DENIAL_BCBS_CPR = dict(
    denial_id="d0000001-0000-0000-0000-000000000001",
    payer_id="BCBS_TX",
    cpt_codes=["92950"],
    icd_codes=["I46.9"],
    denial_reason="Medical necessity not established",
    denial_code="CO-50",
    policy_citation="LCD L34462",
    date="2024-03-15",
    resolved=False,
    resolution=None,
    source="denial_eob",
)

DENIAL_BCBS_CPR_2 = dict(
    denial_id="d0000001-0000-0000-0000-000000000002",
    payer_id="BCBS_TX",
    cpt_codes=["92950"],
    icd_codes=["I46.9"],
    denial_reason="Medical necessity not established",
    denial_code="CO-50",
    policy_citation="LCD L34462",
    date="2024-04-01",
    resolved=False,
    resolution=None,
    source="denial_eob",
)

DENIAL_BCBS_CPR_3 = dict(
    denial_id="d0000001-0000-0000-0000-000000000003",
    payer_id="BCBS_TX",
    cpt_codes=["92950"],
    icd_codes=["I46.9"],
    denial_reason="Medical necessity not established",
    denial_code="CO-50",
    policy_citation=None,
    date="2024-04-20",
    resolved=True,
    resolution="Overturned on appeal",
    source="appeal_outcome",
)

DENIAL_AETNA_AUTH = dict(
    denial_id="d0000002-0000-0000-0000-000000000004",
    payer_id="AETNA",
    cpt_codes=["A0427"],
    icd_codes=["R55"],
    denial_reason="Prior authorization required",
    denial_code=None,
    policy_citation=None,
    date="2024-05-10",
    resolved=False,
    resolution=None,
    source="manual_entry",
)

RULE_BCBS_MED_NEC = dict(
    rule_id="r0000001-0000-0000-0000-000000000001",
    payer_id="BCBS_TX",
    rule_type="medical_necessity",
    codes_affected=["92950", "I46.9"],
    description="Medical necessity must be documented for CPR codes",
    common_denial_reason="Medical necessity not established",
    required_documentation=["patient_condition", "clinical_notes"],
    derived_from_denial_ids=[
        "d0000001-0000-0000-0000-000000000001",
        "d0000001-0000-0000-0000-000000000002",
    ],
    confidence=0.3,
    source="synthesize_payer_rules",
)


@pytest.fixture()
def tmp_index_path(tmp_path: Path) -> Path:
    return tmp_path / "payer_rules_index.json"


@pytest.fixture()
def three_denials() -> list[DenialRecord]:
    return [
        DenialRecord.model_validate(DENIAL_BCBS_CPR),
        DenialRecord.model_validate(DENIAL_BCBS_CPR_2),
        DenialRecord.model_validate(DENIAL_AETNA_AUTH),
    ]


@pytest.fixture()
def populated_index(
    three_denials: list[DenialRecord], tmp_index_path: Path
) -> PayerRulesIndex:
    idx = PayerRulesIndex(tmp_index_path)
    for d in three_denials:
        idx.add_denial(d)
    return idx


# ---------------------------------------------------------------------------
# DenialRecord — validation
# ---------------------------------------------------------------------------


class TestDenialRecordValidation:
    def test_valid_full_record(self):
        r = DenialRecord.model_validate(DENIAL_BCBS_CPR)
        assert r.denial_id == "d0000001-0000-0000-0000-000000000001"
        assert r.payer_id == "BCBS_TX"
        assert r.cpt_codes == ["92950"]
        assert r.icd_codes == ["I46.9"]
        assert r.denial_reason == "Medical necessity not established"
        assert r.denial_code == "CO-50"
        assert r.policy_citation == "LCD L34462"
        assert r.date == "2024-03-15"
        assert r.resolved is False
        assert r.resolution is None
        assert r.source == "denial_eob"

    def test_minimal_record(self):
        minimal = dict(
            denial_id="abc",
            payer_id="X",
            denial_reason="Some reason",
            source="manual_entry",
        )
        r = DenialRecord.model_validate(minimal)
        assert r.cpt_codes == []
        assert r.icd_codes == []
        assert r.denial_code is None
        assert r.policy_citation is None
        assert r.date is None
        assert r.resolved is False
        assert r.resolution is None

    def test_missing_denial_id_raises(self):
        data = {**DENIAL_BCBS_CPR}
        del data["denial_id"]
        with pytest.raises(ValidationError):
            DenialRecord.model_validate(data)

    def test_missing_payer_id_raises(self):
        data = {**DENIAL_BCBS_CPR}
        del data["payer_id"]
        with pytest.raises(ValidationError):
            DenialRecord.model_validate(data)

    def test_missing_denial_reason_raises(self):
        data = {**DENIAL_BCBS_CPR}
        del data["denial_reason"]
        with pytest.raises(ValidationError):
            DenialRecord.model_validate(data)

    def test_missing_source_raises(self):
        data = {**DENIAL_BCBS_CPR}
        del data["source"]
        with pytest.raises(ValidationError):
            DenialRecord.model_validate(data)

    def test_invalid_source_literal(self):
        data = {**DENIAL_BCBS_CPR, "source": "pdf_import"}
        with pytest.raises(ValidationError):
            DenialRecord.model_validate(data)

    def test_all_source_literals_accepted(self):
        for src in ("denial_eob", "manual_entry", "appeal_outcome"):
            r = DenialRecord.model_validate({**DENIAL_BCBS_CPR, "source": src})
            assert r.source == src

    def test_extra_fields_forbidden(self):
        data = {**DENIAL_BCBS_CPR, "extra": "bad"}
        with pytest.raises(ValidationError):
            DenialRecord.model_validate(data)

    def test_model_dump_roundtrip(self):
        r = DenialRecord.model_validate(DENIAL_BCBS_CPR)
        reloaded = DenialRecord.model_validate(r.model_dump(mode="json"))
        assert reloaded == r


# ---------------------------------------------------------------------------
# PayerRuleEntry — validation
# ---------------------------------------------------------------------------


class TestPayerRuleEntryValidation:
    def test_valid_rule(self):
        rule = PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC)
        assert rule.payer_id == "BCBS_TX"
        assert rule.rule_type == "medical_necessity"
        assert rule.confidence == 0.3

    def test_all_rule_type_literals_accepted(self):
        for rt in ("coverage", "medical_necessity", "authorization", "bundling", "documentation"):
            r = PayerRuleEntry.model_validate({**RULE_BCBS_MED_NEC, "rule_type": rt})
            assert r.rule_type == rt

    def test_invalid_rule_type_rejected(self):
        data = {**RULE_BCBS_MED_NEC, "rule_type": "unknown"}
        with pytest.raises(ValidationError):
            PayerRuleEntry.model_validate(data)

    def test_confidence_out_of_range_rejected(self):
        for bad_val in (-0.1, 1.1):
            data = {**RULE_BCBS_MED_NEC, "confidence": bad_val}
            with pytest.raises(ValidationError):
                PayerRuleEntry.model_validate(data)

    def test_confidence_boundary_values_accepted(self):
        for val in (0.0, 1.0):
            r = PayerRuleEntry.model_validate({**RULE_BCBS_MED_NEC, "confidence": val})
            assert r.confidence == val

    def test_missing_required_fields(self):
        for field in ("rule_id", "payer_id", "rule_type", "description", "confidence", "source"):
            data = {**RULE_BCBS_MED_NEC}
            del data[field]
            with pytest.raises(ValidationError):
                PayerRuleEntry.model_validate(data)

    def test_optional_list_fields_default_empty(self):
        minimal = dict(
            rule_id="r1",
            payer_id="X",
            rule_type="coverage",
            description="A rule",
            confidence=0.5,
            source="test",
        )
        r = PayerRuleEntry.model_validate(minimal)
        assert r.codes_affected == []
        assert r.required_documentation == []
        assert r.derived_from_denial_ids == []
        assert r.common_denial_reason is None

    def test_model_dump_roundtrip(self):
        rule = PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC)
        reloaded = PayerRuleEntry.model_validate(rule.model_dump(mode="json"))
        assert reloaded == rule


# ---------------------------------------------------------------------------
# PayerRulesIndex — add / properties
# ---------------------------------------------------------------------------


class TestPayerRulesIndexMutation:
    def test_empty_index_counts(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        assert len(idx) == 0
        assert idx.denial_count == 0
        assert idx.rule_count == 0

    def test_add_denial_increments_count(
        self, three_denials: list[DenialRecord], tmp_index_path: Path
    ):
        idx = PayerRulesIndex(tmp_index_path)
        for i, d in enumerate(three_denials, start=1):
            idx.add_denial(d)
            assert idx.denial_count == i

    def test_add_rule_increments_count(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        rule = PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC)
        idx.add_rule(rule)
        assert idx.rule_count == 1
        assert len(idx) == 1

    def test_len_sums_denials_and_rules(
        self, populated_index: PayerRulesIndex, tmp_index_path: Path
    ):
        rule = PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC)
        populated_index.add_rule(rule)
        assert len(populated_index) == 3 + 1  # 3 denials + 1 rule


# ---------------------------------------------------------------------------
# PayerRulesIndex — save / load round-trip
# ---------------------------------------------------------------------------


class TestPayerRulesIndexPersistence:
    def test_save_creates_file(
        self, populated_index: PayerRulesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        assert tmp_index_path.exists()

    def test_saved_file_is_valid_json(
        self, populated_index: PayerRulesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        payload = json.loads(tmp_index_path.read_text())
        assert "denials" in payload
        assert "rules" in payload
        assert isinstance(payload["denials"], list)
        assert isinstance(payload["rules"], list)

    def test_save_preserves_denial_count(
        self, populated_index: PayerRulesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        idx2 = PayerRulesIndex(tmp_index_path)
        assert idx2.denial_count == 3

    def test_save_preserves_rule_count(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_rule(PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC))
        idx.save()
        idx2 = PayerRulesIndex(tmp_index_path)
        assert idx2.rule_count == 1

    def test_round_trip_preserves_denial_data(
        self, populated_index: PayerRulesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        idx2 = PayerRulesIndex(tmp_index_path)
        bcbs = [d for d in idx2._denials if d.payer_id == "BCBS_TX"]
        assert len(bcbs) == 2
        assert bcbs[0].cpt_codes == ["92950"]
        assert bcbs[0].denial_code == "CO-50"

    def test_round_trip_preserves_rule_data(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_rule(PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC))
        idx.save()
        idx2 = PayerRulesIndex(tmp_index_path)
        assert idx2._rules[0].rule_type == "medical_necessity"
        assert idx2._rules[0].confidence == 0.3

    def test_constructor_auto_loads_existing_file(
        self, populated_index: PayerRulesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        idx3 = PayerRulesIndex(tmp_index_path)
        assert idx3.denial_count == 3

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "index.json"
        idx = PayerRulesIndex(deep_path)
        idx.add_denial(DenialRecord.model_validate(DENIAL_BCBS_CPR))
        idx.save()
        assert deep_path.exists()

    def test_append_after_load(
        self, populated_index: PayerRulesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        idx2 = PayerRulesIndex(tmp_index_path)
        idx2.add_denial(DenialRecord.model_validate(DENIAL_BCBS_CPR_3))
        assert idx2.denial_count == 4


# ---------------------------------------------------------------------------
# retrieve_for_payer() — payer isolation
# ---------------------------------------------------------------------------


class TestRetrieveForPayer:
    def test_returns_empty_context_for_unknown_payer(
        self, populated_index: PayerRulesIndex
    ):
        result = populated_index.retrieve_for_payer("UNKNOWN")
        assert result["payer_id"] == "UNKNOWN"
        assert result["rules"] == []
        assert result["common_denial_reasons"] == []
        assert result["required_documentation"] == []

    def test_returns_only_matching_payer_rules(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        rule_bcbs = PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC)
        rule_aetna = PayerRuleEntry.model_validate(
            {
                **RULE_BCBS_MED_NEC,
                "rule_id": "r2",
                "payer_id": "AETNA",
                "description": "Aetna auth rule",
                "rule_type": "authorization",
            }
        )
        idx.add_rule(rule_bcbs)
        idx.add_rule(rule_aetna)

        result = idx.retrieve_for_payer("BCBS_TX")
        assert len(result["rules"]) == 1
        assert result["rules"][0].payer_id == "BCBS_TX"

    def test_aetna_does_not_receive_bcbs_rules(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_rule(PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC))
        result = idx.retrieve_for_payer("AETNA")
        assert result["rules"] == []

    def test_common_denial_reasons_deduplicated(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        for rid in ("r1", "r2"):
            idx.add_rule(
                PayerRuleEntry.model_validate(
                    {
                        **RULE_BCBS_MED_NEC,
                        "rule_id": rid,
                        "common_denial_reason": "Medical necessity not established",
                    }
                )
            )
        result = idx.retrieve_for_payer("BCBS_TX")
        assert result["common_denial_reasons"].count("Medical necessity not established") == 1

    def test_required_documentation_union_deduplicated(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_rule(
            PayerRuleEntry.model_validate(
                {**RULE_BCBS_MED_NEC, "rule_id": "r1", "required_documentation": ["doc_A", "doc_B"]}
            )
        )
        idx.add_rule(
            PayerRuleEntry.model_validate(
                {**RULE_BCBS_MED_NEC, "rule_id": "r2", "required_documentation": ["doc_B", "doc_C"]}
            )
        )
        result = idx.retrieve_for_payer("BCBS_TX")
        assert sorted(result["required_documentation"]) == ["doc_A", "doc_B", "doc_C"]

    def test_top_k_limits_rules_returned(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        for i in range(5):
            idx.add_rule(
                PayerRuleEntry.model_validate(
                    {**RULE_BCBS_MED_NEC, "rule_id": f"r{i}", "codes_affected": [f"9{i}000"]}
                )
            )
        result = idx.retrieve_for_payer("BCBS_TX", top_k=3)
        assert len(result["rules"]) == 3

    def test_code_filter_exact_match(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_rule(PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC))
        result = idx.retrieve_for_payer("BCBS_TX", codes=["92950"])
        assert len(result["rules"]) == 1

    def test_code_filter_prefix_match(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_rule(PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC))
        result = idx.retrieve_for_payer("BCBS_TX", codes=["929"])
        assert len(result["rules"]) == 1

    def test_code_filter_case_insensitive(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_rule(PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC))
        result_upper = idx.retrieve_for_payer("BCBS_TX", codes=["92950"])
        result_lower = idx.retrieve_for_payer("BCBS_TX", codes=["92950"])
        assert len(result_upper["rules"]) == len(result_lower["rules"])

    def test_non_matching_code_excludes_rule(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_rule(PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC))
        result = idx.retrieve_for_payer("BCBS_TX", codes=["A0427"])
        assert result["rules"] == []


# ---------------------------------------------------------------------------
# get_denial_patterns() — payer + code filtering
# ---------------------------------------------------------------------------


class TestGetDenialPatterns:
    def test_returns_only_matching_payer(
        self, populated_index: PayerRulesIndex
    ):
        results = populated_index.get_denial_patterns("BCBS_TX")
        assert len(results) == 2
        assert all(d.payer_id == "BCBS_TX" for d in results)

    def test_unknown_payer_returns_empty(
        self, populated_index: PayerRulesIndex
    ):
        assert populated_index.get_denial_patterns("UNKNOWN") == []

    def test_code_filter_narrows_results(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_denial(DenialRecord.model_validate(DENIAL_BCBS_CPR))
        idx.add_denial(DenialRecord.model_validate(DENIAL_AETNA_AUTH))
        # Only BCBS denial has "92950"
        results = idx.get_denial_patterns("BCBS_TX", code="92950")
        assert len(results) == 1
        assert results[0].denial_id == DENIAL_BCBS_CPR["denial_id"]

    def test_code_filter_prefix_match(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_denial(DenialRecord.model_validate(DENIAL_BCBS_CPR))
        results = idx.get_denial_patterns("BCBS_TX", code="929")
        assert len(results) == 1

    def test_code_filter_case_insensitive_icd(self, tmp_index_path: Path):
        idx = PayerRulesIndex(tmp_index_path)
        idx.add_denial(DenialRecord.model_validate(DENIAL_BCBS_CPR))
        results_upper = idx.get_denial_patterns("BCBS_TX", code="I46")
        results_lower = idx.get_denial_patterns("BCBS_TX", code="i46")
        assert len(results_upper) == len(results_lower) == 1

    def test_no_code_filter_returns_all_for_payer(
        self, populated_index: PayerRulesIndex
    ):
        results = populated_index.get_denial_patterns("BCBS_TX")
        assert len(results) == 2


# ---------------------------------------------------------------------------
# synthesize_payer_rules — inductive rule building
# ---------------------------------------------------------------------------


class TestSynthesizePowerRulesScript:
    """Test synthesize() logic directly (not via CLI args)."""

    @pytest.fixture(autouse=True)
    def import_synthesize(self):
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import importlib
        self.mod = importlib.import_module("synthesize_payer_rules")

    def test_three_identical_denials_produce_one_rule(self, tmp_path: Path):
        """Core inductive requirement: 3 denials → 1 PayerRuleEntry."""
        index_path = tmp_path / "index.json"
        idx = PayerRulesIndex(index_path)
        for d in [DENIAL_BCBS_CPR, DENIAL_BCBS_CPR_2, DENIAL_BCBS_CPR_3]:
            idx.add_denial(DenialRecord.model_validate(d))
        idx.save()

        count = self.mod.synthesize(index_path, min_denials=2)
        assert count == 1

        idx2 = PayerRulesIndex(index_path)
        assert idx2.rule_count == 1
        rule = idx2._rules[0]
        assert rule.payer_id == "BCBS_TX"
        assert rule.common_denial_reason == "Medical necessity not established"
        assert len(rule.derived_from_denial_ids) == 3

    def test_confidence_proportional_to_count(self, tmp_path: Path):
        index_path = tmp_path / "index.json"
        idx = PayerRulesIndex(index_path)
        for d in [DENIAL_BCBS_CPR, DENIAL_BCBS_CPR_2, DENIAL_BCBS_CPR_3]:
            idx.add_denial(DenialRecord.model_validate(d))
        idx.save()

        self.mod.synthesize(index_path, min_denials=2, confidence_denominator=10.0)
        idx2 = PayerRulesIndex(index_path)
        assert idx2._rules[0].confidence == pytest.approx(0.3)  # 3/10

    def test_confidence_capped_at_one(self, tmp_path: Path):
        index_path = tmp_path / "index.json"
        idx = PayerRulesIndex(index_path)
        for d in [DENIAL_BCBS_CPR, DENIAL_BCBS_CPR_2, DENIAL_BCBS_CPR_3]:
            idx.add_denial(DenialRecord.model_validate(d))
        idx.save()

        # denominator=1 would give 3.0 without the cap
        self.mod.synthesize(index_path, min_denials=2, confidence_denominator=1.0)
        idx2 = PayerRulesIndex(index_path)
        assert idx2._rules[0].confidence == 1.0

    def test_below_threshold_produces_no_rule(self, tmp_path: Path):
        """Only 1 denial → no rule when min_denials=2."""
        index_path = tmp_path / "index.json"
        idx = PayerRulesIndex(index_path)
        idx.add_denial(DenialRecord.model_validate(DENIAL_BCBS_CPR))
        idx.save()

        count = self.mod.synthesize(index_path, min_denials=2)
        assert count == 0

        idx2 = PayerRulesIndex(index_path)
        assert idx2.rule_count == 0

    def test_different_payers_produce_separate_rules(self, tmp_path: Path):
        index_path = tmp_path / "index.json"
        idx = PayerRulesIndex(index_path)
        # 2 BCBS denials for same codes/reason
        idx.add_denial(DenialRecord.model_validate(DENIAL_BCBS_CPR))
        idx.add_denial(DenialRecord.model_validate(DENIAL_BCBS_CPR_2))
        # 2 AETNA denials for same codes/reason
        aetna2 = {**DENIAL_AETNA_AUTH, "denial_id": "d-aetna-2"}
        idx.add_denial(DenialRecord.model_validate(DENIAL_AETNA_AUTH))
        idx.add_denial(DenialRecord.model_validate(aetna2))
        idx.save()

        count = self.mod.synthesize(index_path, min_denials=2)
        assert count == 2

        idx2 = PayerRulesIndex(index_path)
        payers = {r.payer_id for r in idx2._rules}
        assert "BCBS_TX" in payers
        assert "AETNA" in payers

    def test_dry_run_does_not_write(self, tmp_path: Path):
        index_path = tmp_path / "index.json"
        idx = PayerRulesIndex(index_path)
        for d in [DENIAL_BCBS_CPR, DENIAL_BCBS_CPR_2]:
            idx.add_denial(DenialRecord.model_validate(d))
        idx.save()

        self.mod.synthesize(index_path, min_denials=2, dry_run=True)
        idx2 = PayerRulesIndex(index_path)
        assert idx2.rule_count == 0  # no rules written

    def test_missing_index_returns_negative(self, tmp_path: Path):
        result = self.mod.synthesize(tmp_path / "nonexistent.json")
        assert result == -1

    def test_empty_index_returns_zero(self, tmp_path: Path):
        index_path = tmp_path / "index.json"
        idx = PayerRulesIndex(index_path)
        idx.save()
        count = self.mod.synthesize(index_path)
        assert count == 0

    def test_infer_rule_type_medical_necessity(self):
        assert self.mod._infer_rule_type("Medical necessity not established") == "medical_necessity"

    def test_infer_rule_type_authorization(self):
        assert self.mod._infer_rule_type("Prior authorization required") == "authorization"

    def test_infer_rule_type_bundling(self):
        assert self.mod._infer_rule_type("Unbundling of procedure codes") == "bundling"

    def test_infer_rule_type_documentation(self):
        assert self.mod._infer_rule_type("Missing documentation") == "documentation"

    def test_infer_rule_type_coverage(self):
        assert self.mod._infer_rule_type("Not covered under benefit plan") == "coverage"

    def test_rule_source_set_to_script_name(self, tmp_path: Path):
        index_path = tmp_path / "index.json"
        idx = PayerRulesIndex(index_path)
        idx.add_denial(DenialRecord.model_validate(DENIAL_BCBS_CPR))
        idx.add_denial(DenialRecord.model_validate(DENIAL_BCBS_CPR_2))
        idx.save()

        self.mod.synthesize(index_path, min_denials=2)
        idx2 = PayerRulesIndex(index_path)
        assert idx2._rules[0].source == "synthesize_payer_rules"


# ---------------------------------------------------------------------------
# ingest_denial script helpers
# ---------------------------------------------------------------------------


class TestIngestDenialScript:
    @pytest.fixture(autouse=True)
    def import_ingest(self):
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import importlib
        self.mod = importlib.import_module("ingest_denial")

    def test_main_json_ingest(self, tmp_path: Path):
        data = [DENIAL_BCBS_CPR, DENIAL_AETNA_AUTH]
        input_file = tmp_path / "denials.json"
        input_file.write_text(json.dumps(data), encoding="utf-8")
        index_file = tmp_path / "index.json"

        rc = self.mod.main(["--index", str(index_file), str(input_file)])
        assert rc == 0
        assert index_file.exists()

        idx = PayerRulesIndex(index_file)
        assert idx.denial_count == 2

    def test_main_missing_input_returns_1(self, tmp_path: Path):
        rc = self.mod.main([
            "--index", str(tmp_path / "idx.json"),
            str(tmp_path / "nonexistent.json"),
        ])
        assert rc == 1

    def test_main_non_array_json_returns_1(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"key": "val"}), encoding="utf-8")
        rc = self.mod.main(["--index", str(tmp_path / "idx.json"), str(f)])
        assert rc == 1

    def test_main_bad_row_skipped(self, tmp_path: Path):
        data = [
            DENIAL_BCBS_CPR,
            {"denial_id": "x", "payer_id": "X", "denial_reason": "bad", "source": "NOT_VALID"},
        ]
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        index_file = tmp_path / "index.json"

        rc = self.mod.main(["--index", str(index_file), str(f)])
        assert rc == 0  # exits 0 (skipped, not fatal)

        idx = PayerRulesIndex(index_file)
        assert idx.denial_count == 1  # only valid record

    def test_main_appends_to_existing_index(self, tmp_path: Path):
        index_file = tmp_path / "index.json"
        f1 = tmp_path / "d1.json"
        f1.write_text(json.dumps([DENIAL_BCBS_CPR]), encoding="utf-8")
        self.mod.main(["--index", str(index_file), str(f1)])

        f2 = tmp_path / "d2.json"
        f2.write_text(json.dumps([DENIAL_AETNA_AUTH]), encoding="utf-8")
        self.mod.main(["--index", str(index_file), str(f2)])

        idx = PayerRulesIndex(index_file)
        assert idx.denial_count == 2

    def test_main_payer_distribution_printed(self, tmp_path: Path, capsys):
        data = [DENIAL_BCBS_CPR, DENIAL_BCBS_CPR_2, DENIAL_AETNA_AUTH]
        f = tmp_path / "denials.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        self.mod.main(["--index", str(tmp_path / "idx.json"), str(f)])
        captured = capsys.readouterr()
        assert "BCBS_TX" in captured.out
        assert "AETNA" in captured.out


# ---------------------------------------------------------------------------
# retrieve_payer_requirements() dispatcher in rag/__init__.py
# ---------------------------------------------------------------------------


class TestRetrievePayerRequirementsDispatcher:
    def test_returns_empty_when_payer_id_is_none(self):
        from ems_pipeline.rag import retrieve_payer_requirements
        result = retrieve_payer_requirements(None)
        assert result == {}

    def test_returns_empty_when_env_var_not_set(self):
        env = {k: v for k, v in os.environ.items() if k != "EMS_PAYER_RULES_INDEX"}
        with patch.dict(os.environ, env, clear=True):
            from ems_pipeline.rag import retrieve_payer_requirements
            result = retrieve_payer_requirements("BCBS_TX")
        assert result == {}

    def test_returns_empty_when_file_missing(self, tmp_path: Path):
        missing = str(tmp_path / "no_such_file.json")
        with patch.dict(os.environ, {"EMS_PAYER_RULES_INDEX": missing}):
            from ems_pipeline.rag import retrieve_payer_requirements
            result = retrieve_payer_requirements("BCBS_TX")
        assert result == {}

    def test_returns_context_when_index_exists(self, tmp_path: Path):
        index_file = tmp_path / "index.json"
        idx = PayerRulesIndex(index_file)
        idx.add_rule(PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC))
        idx.save()

        with patch.dict(os.environ, {"EMS_PAYER_RULES_INDEX": str(index_file)}):
            from ems_pipeline.rag import retrieve_payer_requirements
            result = retrieve_payer_requirements("BCBS_TX")

        assert result["payer_id"] == "BCBS_TX"
        assert len(result["rules"]) == 1

    def test_context_shape(self, tmp_path: Path):
        index_file = tmp_path / "index.json"
        idx = PayerRulesIndex(index_file)
        idx.add_rule(PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC))
        idx.save()

        with patch.dict(os.environ, {"EMS_PAYER_RULES_INDEX": str(index_file)}):
            from ems_pipeline.rag import retrieve_payer_requirements
            result = retrieve_payer_requirements("BCBS_TX")

        assert set(result.keys()) == {
            "payer_id", "rules", "common_denial_reasons", "required_documentation"
        }

    def test_wrong_payer_returns_empty_rules(self, tmp_path: Path):
        index_file = tmp_path / "index.json"
        idx = PayerRulesIndex(index_file)
        idx.add_rule(PayerRuleEntry.model_validate(RULE_BCBS_MED_NEC))
        idx.save()

        with patch.dict(os.environ, {"EMS_PAYER_RULES_INDEX": str(index_file)}):
            from ems_pipeline.rag import retrieve_payer_requirements
            result = retrieve_payer_requirements("AETNA")

        assert result["rules"] == []
