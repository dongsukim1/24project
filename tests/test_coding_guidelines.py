"""Tests for Index 3: Coding Guidelines RAG scaffold.

Covers:
- CodingGuidelineEntry validation (required fields, Literal enforcement)
- CodingGuidelinesIndex add/retrieve/save/load round-trip
- retrieve() filtering by specialty
- retrieve() filtering by entity_types via ENTITY_TYPE_TO_BODY_SYSTEM
- retrieve_coding_guidelines() dispatcher in rag/__init__.py
- ingest_coding_guidelines script helpers
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ems_pipeline.rag.coding_guidelines import (
    ENTITY_TYPE_TO_BODY_SYSTEM,
    CodingGuidelineEntry,
    CodingGuidelinesIndex,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ENTRY_CHEST_PAIN = dict(
    code="R07.9",
    code_type="ICD10",
    description="Chest pain, unspecified",
    specialty="emergency_medicine",
    body_system="cardiovascular",
    required_documentation=["onset", "character", "severity"],
    exclusion_codes=["R07.1", "R07.2"],
    notes=None,
    source="ICD-10-CM 2024",
)

ENTRY_RESP_FAILURE = dict(
    code="J96.00",
    code_type="ICD10",
    description="Acute respiratory failure, unspecified",
    specialty="emergency_medicine",
    body_system="respiratory",
    required_documentation=["onset", "SpO2", "O2_support"],
    exclusion_codes=[],
    notes="Document cause when known.",
    source="ICD-10-CM 2024",
)

ENTRY_CPR = dict(
    code="92950",
    code_type="CPT",
    description="Cardiopulmonary resuscitation",
    specialty="emergency_medicine",
    body_system="cardiovascular",
    required_documentation=["duration", "outcome"],
    exclusion_codes=[],
    notes=None,
    source="AMA CPT 2024",
)


@pytest.fixture()
def three_entries() -> list[CodingGuidelineEntry]:
    return [
        CodingGuidelineEntry.model_validate(ENTRY_CHEST_PAIN),
        CodingGuidelineEntry.model_validate(ENTRY_RESP_FAILURE),
        CodingGuidelineEntry.model_validate(ENTRY_CPR),
    ]


@pytest.fixture()
def tmp_index_path(tmp_path: Path) -> Path:
    return tmp_path / "test_index.json"


@pytest.fixture()
def populated_index(
    three_entries: list[CodingGuidelineEntry], tmp_index_path: Path
) -> CodingGuidelinesIndex:
    idx = CodingGuidelinesIndex(tmp_index_path)
    for e in three_entries:
        idx.add_entry(e)
    return idx


# ---------------------------------------------------------------------------
# CodingGuidelineEntry — validation
# ---------------------------------------------------------------------------


class TestCodingGuidelineEntryValidation:
    def test_valid_icd10_entry(self):
        e = CodingGuidelineEntry.model_validate(ENTRY_CHEST_PAIN)
        assert e.code == "R07.9"
        assert e.code_type == "ICD10"
        assert e.description == "Chest pain, unspecified"
        assert e.specialty == "emergency_medicine"
        assert e.body_system == "cardiovascular"
        assert e.required_documentation == ["onset", "character", "severity"]
        assert e.exclusion_codes == ["R07.1", "R07.2"]
        assert e.notes is None
        assert e.source == "ICD-10-CM 2024"

    def test_valid_cpt_entry(self):
        e = CodingGuidelineEntry.model_validate(ENTRY_CPR)
        assert e.code == "92950"
        assert e.code_type == "CPT"

    def test_missing_code_raises(self):
        data = {**ENTRY_CHEST_PAIN}
        del data["code"]
        with pytest.raises(ValidationError):
            CodingGuidelineEntry.model_validate(data)

    def test_missing_code_type_raises(self):
        data = {**ENTRY_CHEST_PAIN}
        del data["code_type"]
        with pytest.raises(ValidationError):
            CodingGuidelineEntry.model_validate(data)

    def test_missing_description_raises(self):
        data = {**ENTRY_CHEST_PAIN}
        del data["description"]
        with pytest.raises(ValidationError):
            CodingGuidelineEntry.model_validate(data)

    def test_missing_source_raises(self):
        data = {**ENTRY_CHEST_PAIN}
        del data["source"]
        with pytest.raises(ValidationError):
            CodingGuidelineEntry.model_validate(data)

    def test_invalid_code_type_literal(self):
        data = {**ENTRY_CHEST_PAIN, "code_type": "SNOMED"}
        with pytest.raises(ValidationError):
            CodingGuidelineEntry.model_validate(data)

    def test_code_type_case_sensitive(self):
        """Literal["ICD10", "CPT"] is case-sensitive."""
        data = {**ENTRY_CHEST_PAIN, "code_type": "icd10"}
        with pytest.raises(ValidationError):
            CodingGuidelineEntry.model_validate(data)

    def test_optional_fields_default_to_none_or_empty(self):
        minimal = dict(
            code="Z99.9",
            code_type="ICD10",
            description="Minimal entry",
            source="test",
        )
        e = CodingGuidelineEntry.model_validate(minimal)
        assert e.specialty is None
        assert e.body_system is None
        assert e.required_documentation == []
        assert e.exclusion_codes == []
        assert e.notes is None

    def test_extra_fields_forbidden(self):
        data = {**ENTRY_CHEST_PAIN, "unexpected_field": "oops"}
        with pytest.raises(ValidationError):
            CodingGuidelineEntry.model_validate(data)

    def test_notes_can_be_string(self):
        data = {**ENTRY_CHEST_PAIN, "notes": "Some note"}
        e = CodingGuidelineEntry.model_validate(data)
        assert e.notes == "Some note"

    def test_model_dump_roundtrip(self):
        e = CodingGuidelineEntry.model_validate(ENTRY_RESP_FAILURE)
        dumped = e.model_dump(mode="json")
        reloaded = CodingGuidelineEntry.model_validate(dumped)
        assert reloaded == e


# ---------------------------------------------------------------------------
# CodingGuidelinesIndex — add / len
# ---------------------------------------------------------------------------


class TestCodingGuidelinesIndexMutation:
    def test_empty_index_has_len_zero(self, tmp_index_path: Path):
        idx = CodingGuidelinesIndex(tmp_index_path)
        assert len(idx) == 0

    def test_add_entry_increments_len(
        self, three_entries: list[CodingGuidelineEntry], tmp_index_path: Path
    ):
        idx = CodingGuidelinesIndex(tmp_index_path)
        for i, e in enumerate(three_entries, start=1):
            idx.add_entry(e)
            assert len(idx) == i

    def test_add_entry_stores_in_order(
        self, populated_index: CodingGuidelinesIndex,
    ):
        results = populated_index.retrieve(entity_types=[], top_k=10)
        codes = [e.code for e in results]
        assert codes == ["R07.9", "J96.00", "92950"]


# ---------------------------------------------------------------------------
# CodingGuidelinesIndex — save / load round-trip
# ---------------------------------------------------------------------------


class TestCodingGuidelinesIndexPersistence:
    def test_save_creates_file(
        self, populated_index: CodingGuidelinesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        assert tmp_index_path.exists()

    def test_saved_file_is_valid_json_array(
        self, populated_index: CodingGuidelinesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        payload = json.loads(tmp_index_path.read_text())
        assert isinstance(payload, list)
        assert len(payload) == 3

    def test_load_restores_entries(
        self, populated_index: CodingGuidelinesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        idx2 = CodingGuidelinesIndex(tmp_index_path)
        assert len(idx2) == 3

    def test_round_trip_preserves_data(
        self, populated_index: CodingGuidelinesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        idx2 = CodingGuidelinesIndex(tmp_index_path)
        results = idx2.retrieve(entity_types=[], top_k=10)
        assert results[0].code == "R07.9"
        assert results[0].code_type == "ICD10"
        assert results[0].required_documentation == ["onset", "character", "severity"]
        assert results[0].exclusion_codes == ["R07.1", "R07.2"]
        assert results[1].code == "J96.00"
        assert results[1].notes == "Document cause when known."
        assert results[2].code == "92950"
        assert results[2].code_type == "CPT"

    def test_constructor_auto_loads_existing_file(
        self, populated_index: CodingGuidelinesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        idx3 = CodingGuidelinesIndex(tmp_index_path)
        assert len(idx3) == 3

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "index.json"
        idx = CodingGuidelinesIndex(deep_path)
        idx.add_entry(CodingGuidelineEntry.model_validate(ENTRY_CPR))
        idx.save()
        assert deep_path.exists()

    def test_append_after_load(
        self, populated_index: CodingGuidelinesIndex, tmp_index_path: Path
    ):
        populated_index.save()
        idx2 = CodingGuidelinesIndex(tmp_index_path)
        new_entry = CodingGuidelineEntry.model_validate(
            dict(code="Z00.00", code_type="ICD10", description="Extra", source="test")
        )
        idx2.add_entry(new_entry)
        assert len(idx2) == 4


# ---------------------------------------------------------------------------
# retrieve() — no filters (top_k)
# ---------------------------------------------------------------------------


class TestRetrieveNoFilters:
    def test_returns_all_when_top_k_covers(
        self, populated_index: CodingGuidelinesIndex
    ):
        results = populated_index.retrieve(entity_types=[], top_k=10)
        assert len(results) == 3

    def test_top_k_limits_results(self, populated_index: CodingGuidelinesIndex):
        results = populated_index.retrieve(entity_types=[], top_k=2)
        assert len(results) == 2

    def test_top_k_zero_returns_empty(self, populated_index: CodingGuidelinesIndex):
        results = populated_index.retrieve(entity_types=[], top_k=0)
        assert results == []

    def test_empty_index_returns_empty(self, tmp_index_path: Path):
        idx = CodingGuidelinesIndex(tmp_index_path)
        assert idx.retrieve(entity_types=[]) == []


# ---------------------------------------------------------------------------
# retrieve() — code prefix filter
# ---------------------------------------------------------------------------


class TestRetrieveCodeFilter:
    def test_exact_code_match(self, populated_index: CodingGuidelinesIndex):
        results = populated_index.retrieve(entity_types=[], codes=["R07.9"])
        assert len(results) == 1
        assert results[0].code == "R07.9"

    def test_prefix_match(self, populated_index: CodingGuidelinesIndex):
        results = populated_index.retrieve(entity_types=[], codes=["R07"])
        assert len(results) == 1
        assert results[0].code == "R07.9"

    def test_prefix_match_case_insensitive(self, populated_index: CodingGuidelinesIndex):
        results = populated_index.retrieve(entity_types=[], codes=["r07"])
        assert len(results) == 1

    def test_multiple_prefixes(self, populated_index: CodingGuidelinesIndex):
        results = populated_index.retrieve(entity_types=[], codes=["R07", "929"])
        codes = {e.code for e in results}
        assert codes == {"R07.9", "92950"}

    def test_no_matching_prefix(self, populated_index: CodingGuidelinesIndex):
        results = populated_index.retrieve(entity_types=[], codes=["Z99"])
        assert results == []


# ---------------------------------------------------------------------------
# retrieve() — specialty filter
# ---------------------------------------------------------------------------


class TestRetrieveSpecialtyFilter:
    def test_matching_specialty_returns_entries(
        self, populated_index: CodingGuidelinesIndex
    ):
        results = populated_index.retrieve(
            entity_types=[], specialty="emergency_medicine"
        )
        assert len(results) == 3

    def test_non_matching_specialty_returns_none(
        self, populated_index: CodingGuidelinesIndex
    ):
        results = populated_index.retrieve(entity_types=[], specialty="cardiology")
        assert results == []

    def test_generic_entry_passes_specialty_filter(self, tmp_index_path: Path):
        """An entry with specialty=None is returned for any specialty query."""
        generic = CodingGuidelineEntry.model_validate(
            dict(
                code="Z00.00",
                code_type="ICD10",
                description="Generic",
                specialty=None,
                source="test",
            )
        )
        specific = CodingGuidelineEntry.model_validate(
            dict(
                code="R07.9",
                code_type="ICD10",
                description="Chest pain",
                specialty="emergency_medicine",
                source="test",
            )
        )
        idx = CodingGuidelinesIndex(tmp_index_path)
        idx.add_entry(generic)
        idx.add_entry(specific)

        results = idx.retrieve(entity_types=[], specialty="cardiology", top_k=10)
        # generic passes (specialty=None), specific does not (wrong specialty)
        assert len(results) == 1
        assert results[0].code == "Z00.00"

    def test_specialty_filter_combined_with_code(
        self, populated_index: CodingGuidelinesIndex
    ):
        results = populated_index.retrieve(
            entity_types=[], codes=["R07"], specialty="emergency_medicine"
        )
        assert len(results) == 1
        assert results[0].code == "R07.9"


# ---------------------------------------------------------------------------
# retrieve() — entity_types / body-system filter
# ---------------------------------------------------------------------------


class TestRetrieveEntityTypeFilter:
    def test_vital_bp_returns_cardiovascular(
        self, populated_index: CodingGuidelinesIndex
    ):
        results = populated_index.retrieve(entity_types=["VITAL_BP"], top_k=10)
        # VITAL_BP → cardiovascular; entries with body_system=None also pass
        codes = {e.code for e in results}
        assert "R07.9" in codes   # cardiovascular
        assert "92950" in codes   # cardiovascular
        assert "J96.00" not in codes  # respiratory

    def test_vital_spo2_returns_respiratory(
        self, populated_index: CodingGuidelinesIndex
    ):
        results = populated_index.retrieve(entity_types=["VITAL_SPO2"], top_k=10)
        codes = {e.code for e in results}
        assert "J96.00" in codes  # respiratory
        assert "R07.9" not in codes  # cardiovascular
        assert "92950" not in codes  # cardiovascular

    def test_multiple_entity_types_union_body_systems(
        self, populated_index: CodingGuidelinesIndex
    ):
        results = populated_index.retrieve(
            entity_types=["VITAL_BP", "VITAL_SPO2"], top_k=10
        )
        codes = {e.code for e in results}
        assert codes == {"R07.9", "J96.00", "92950"}

    def test_entity_types_mapping_to_none_skips_filter(
        self, populated_index: CodingGuidelinesIndex
    ):
        """When all entity_types map to None, body-system filter is skipped."""
        results = populated_index.retrieve(
            entity_types=["SYMPTOM", "CONDITION"], top_k=10
        )
        assert len(results) == 3  # no body-system filter applied

    def test_unknown_entity_type_maps_to_none(
        self, populated_index: CodingGuidelinesIndex
    ):
        results = populated_index.retrieve(
            entity_types=["UNKNOWN_TYPE"], top_k=10
        )
        assert len(results) == 3  # filter skipped

    def test_entry_with_null_body_system_always_passes(self, tmp_index_path: Path):
        """Entries with body_system=None are not excluded by body-system filter."""
        generic = CodingGuidelineEntry.model_validate(
            dict(
                code="Z00.00",
                code_type="ICD10",
                description="Generic — no body system",
                body_system=None,
                source="test",
            )
        )
        idx = CodingGuidelinesIndex(tmp_index_path)
        idx.add_entry(generic)

        # VITAL_BP maps to "cardiovascular"; generic entry has body_system=None → passes
        results = idx.retrieve(entity_types=["VITAL_BP"], top_k=10)
        assert len(results) == 1
        assert results[0].code == "Z00.00"

    def test_empty_entity_types_list_skips_filter(
        self, populated_index: CodingGuidelinesIndex
    ):
        results = populated_index.retrieve(entity_types=[], top_k=10)
        assert len(results) == 3

    def test_entity_type_uppercase_normalization(
        self, populated_index: CodingGuidelinesIndex
    ):
        """entity_types are uppercased before lookup."""
        results_upper = populated_index.retrieve(entity_types=["VITAL_BP"], top_k=10)
        results_lower = populated_index.retrieve(entity_types=["vital_bp"], top_k=10)
        assert {e.code for e in results_upper} == {e.code for e in results_lower}


# ---------------------------------------------------------------------------
# ENTITY_TYPE_TO_BODY_SYSTEM mapping
# ---------------------------------------------------------------------------


class TestEntityTypeToBodySystemMapping:
    def test_vital_bp_maps_to_cardiovascular(self):
        assert ENTITY_TYPE_TO_BODY_SYSTEM["VITAL_BP"] == "cardiovascular"

    def test_vital_spo2_maps_to_respiratory(self):
        assert ENTITY_TYPE_TO_BODY_SYSTEM["VITAL_SPO2"] == "respiratory"

    def test_operational_types_map_to_none(self):
        for et in ("UNIT_ID", "ETA", "RESOURCE"):
            assert ENTITY_TYPE_TO_BODY_SYSTEM[et] is None, f"{et} should map to None"

    def test_clinical_generic_types_map_to_none(self):
        for et in ("SYMPTOM", "CONDITION", "PROCEDURE", "MEDICATION", "ASSESSMENT"):
            assert ENTITY_TYPE_TO_BODY_SYSTEM[et] is None, f"{et} should map to None"

    def test_all_extract_entity_types_present(self):
        expected_keys = {
            "VITAL_SPO2", "VITAL_BP", "SYMPTOM", "CONDITION",
            "PROCEDURE", "RESOURCE", "MEDICATION", "ASSESSMENT",
            "UNIT_ID", "ETA",
        }
        assert expected_keys == set(ENTITY_TYPE_TO_BODY_SYSTEM.keys())


# ---------------------------------------------------------------------------
# retrieve_coding_guidelines() dispatcher in rag/__init__.py
# ---------------------------------------------------------------------------


class TestRetrieveCodingGuidelinesDispatcher:
    def test_returns_empty_when_env_var_not_set(self):
        env = {k: v for k, v in os.environ.items() if k != "EMS_CODING_GUIDELINES_INDEX"}
        with patch.dict(os.environ, env, clear=True):
            from ems_pipeline.rag import retrieve_coding_guidelines
            result = retrieve_coding_guidelines(["VITAL_BP"])
        assert result == {}

    def test_returns_empty_when_file_missing(self, tmp_path: Path):
        missing = str(tmp_path / "no_such_file.json")
        with patch.dict(os.environ, {"EMS_CODING_GUIDELINES_INDEX": missing}):
            from ems_pipeline.rag import retrieve_coding_guidelines
            result = retrieve_coding_guidelines(["VITAL_BP"])
        assert result == {}

    def test_returns_dict_when_index_exists(
        self,
        populated_index: CodingGuidelinesIndex,
        tmp_index_path: Path,
    ):
        populated_index.save()
        with patch.dict(
            os.environ, {"EMS_CODING_GUIDELINES_INDEX": str(tmp_index_path)}
        ):
            from ems_pipeline.rag import retrieve_coding_guidelines
            result = retrieve_coding_guidelines(["VITAL_BP"])
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_result_keys_have_correct_format(
        self,
        populated_index: CodingGuidelinesIndex,
        tmp_index_path: Path,
    ):
        populated_index.save()
        with patch.dict(
            os.environ, {"EMS_CODING_GUIDELINES_INDEX": str(tmp_index_path)}
        ):
            from ems_pipeline.rag import retrieve_coding_guidelines
            result = retrieve_coding_guidelines(["VITAL_BP"])
        for key in result:
            assert "[ICD10]" in key or "[CPT]" in key

    def test_cardiovascular_entity_returns_cardiovascular_codes(
        self,
        populated_index: CodingGuidelinesIndex,
        tmp_index_path: Path,
    ):
        populated_index.save()
        with patch.dict(
            os.environ, {"EMS_CODING_GUIDELINES_INDEX": str(tmp_index_path)}
        ):
            from ems_pipeline.rag import retrieve_coding_guidelines
            result = retrieve_coding_guidelines(["VITAL_BP"])
        # Chest pain and CPR are cardiovascular; resp failure is not
        assert "R07.9 [ICD10]" in result
        assert "92950 [CPT]" in result
        assert "J96.00 [ICD10]" not in result

    def test_respiratory_entity_returns_respiratory_codes(
        self,
        populated_index: CodingGuidelinesIndex,
        tmp_index_path: Path,
    ):
        populated_index.save()
        with patch.dict(
            os.environ, {"EMS_CODING_GUIDELINES_INDEX": str(tmp_index_path)}
        ):
            from ems_pipeline.rag import retrieve_coding_guidelines
            result = retrieve_coding_guidelines(["VITAL_SPO2"])
        assert "J96.00 [ICD10]" in result
        assert "R07.9 [ICD10]" not in result


# ---------------------------------------------------------------------------
# ingest_coding_guidelines script helpers
# ---------------------------------------------------------------------------


class TestIngestScriptHelpers:
    """Import and test the helper functions from the ingest script directly."""

    @pytest.fixture(autouse=True)
    def import_helpers(self):
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import importlib
        self.mod = importlib.import_module("ingest_coding_guidelines")

    def test_parse_pipe_list_normal(self):
        result = self.mod._parse_pipe_list("onset|character|severity")
        assert result == ["onset", "character", "severity"]

    def test_parse_pipe_list_empty_string(self):
        assert self.mod._parse_pipe_list("") == []

    def test_parse_pipe_list_strips_whitespace(self):
        result = self.mod._parse_pipe_list(" onset | severity ")
        assert result == ["onset", "severity"]

    def test_parse_pipe_list_drops_blank_segments(self):
        result = self.mod._parse_pipe_list("a||b")
        assert result == ["a", "b"]

    def test_none_if_blank_empty(self):
        assert self.mod._none_if_blank("") is None
        assert self.mod._none_if_blank("   ") is None

    def test_none_if_blank_non_empty(self):
        assert self.mod._none_if_blank("emergency_medicine") == "emergency_medicine"
        assert self.mod._none_if_blank("  value  ") == "value"

    def test_load_json_rows(self, tmp_path: Path):
        data = [ENTRY_CHEST_PAIN, ENTRY_CPR]
        p = tmp_path / "input.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        rows = self.mod._load_json_rows(p)
        assert len(rows) == 2
        assert rows[0]["code"] == "R07.9"

    def test_load_json_rows_non_array_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"key": "val"}), encoding="utf-8")
        with pytest.raises(ValueError, match="top-level array"):
            self.mod._load_json_rows(p)

    def test_load_csv_rows(self, tmp_path: Path):
        csv_content = (
            "code,code_type,description,specialty,body_system,"
            "required_documentation,exclusion_codes,notes,source\n"
            "R07.9,ICD10,Chest pain,emergency_medicine,cardiovascular,"
            "onset|severity,R07.1,, ICD-10-CM 2024\n"
        )
        p = tmp_path / "input.csv"
        p.write_text(csv_content, encoding="utf-8")
        rows = self.mod._load_csv_rows(p)
        assert len(rows) == 1
        r = rows[0]
        assert r["code"] == "R07.9"
        assert r["required_documentation"] == ["onset", "severity"]
        assert r["exclusion_codes"] == ["R07.1"]
        assert r["notes"] is None
        assert r["source"] == "ICD-10-CM 2024"

    def test_main_json_ingest(self, tmp_path: Path):
        data = [ENTRY_CHEST_PAIN, ENTRY_RESP_FAILURE, ENTRY_CPR]
        input_file = tmp_path / "guidelines.json"
        input_file.write_text(json.dumps(data), encoding="utf-8")
        index_file = tmp_path / "index.json"

        rc = self.mod.main(["--index", str(index_file), str(input_file)])
        assert rc == 0
        assert index_file.exists()
        payload = json.loads(index_file.read_text())
        assert len(payload) == 3

    def test_main_csv_ingest(self, tmp_path: Path):
        csv_content = (
            "code,code_type,description,specialty,body_system,"
            "required_documentation,exclusion_codes,notes,source\n"
            "92950,CPT,Cardiopulmonary resuscitation,emergency_medicine,"
            "cardiovascular,duration|outcome,,, AMA CPT 2024\n"
        )
        input_file = tmp_path / "guidelines.csv"
        input_file.write_text(csv_content, encoding="utf-8")
        index_file = tmp_path / "index.json"

        rc = self.mod.main(["--index", str(index_file), str(input_file)])
        assert rc == 0
        payload = json.loads(index_file.read_text())
        assert len(payload) == 1
        assert payload[0]["code"] == "92950"

    def test_main_missing_input_returns_1(self, tmp_path: Path):
        rc = self.mod.main(["--index", str(tmp_path / "idx.json"),
                            str(tmp_path / "nonexistent.json")])
        assert rc == 1

    def test_main_unknown_format_returns_1(self, tmp_path: Path):
        f = tmp_path / "data.xyz"
        f.write_text("[]", encoding="utf-8")
        rc = self.mod.main(["--index", str(tmp_path / "idx.json"), str(f)])
        assert rc == 1

    def test_main_explicit_format_flag(self, tmp_path: Path):
        """--format json overrides extension inference."""
        data = [ENTRY_CHEST_PAIN]
        # Name it .txt but pass --format json
        input_file = tmp_path / "data.txt"
        input_file.write_text(json.dumps(data), encoding="utf-8")
        index_file = tmp_path / "index.json"
        rc = self.mod.main([
            "--format", "json",
            "--index", str(index_file),
            str(input_file),
        ])
        assert rc == 0
        payload = json.loads(index_file.read_text())
        assert len(payload) == 1

    def test_main_appends_to_existing_index(self, tmp_path: Path):
        index_file = tmp_path / "index.json"
        # First ingest
        d1 = tmp_path / "d1.json"
        d1.write_text(json.dumps([ENTRY_CHEST_PAIN]), encoding="utf-8")
        self.mod.main(["--index", str(index_file), str(d1)])

        # Second ingest — should append
        d2 = tmp_path / "d2.json"
        d2.write_text(json.dumps([ENTRY_CPR]), encoding="utf-8")
        self.mod.main(["--index", str(index_file), str(d2)])

        payload = json.loads(index_file.read_text())
        assert len(payload) == 2

    def test_main_bad_row_skipped_not_fatal(self, tmp_path: Path):
        data = [
            ENTRY_CHEST_PAIN,
            {"code": "BAD", "code_type": "INVALID_TYPE", "description": "bad",
             "source": "test"},
        ]
        input_file = tmp_path / "mixed.json"
        input_file.write_text(json.dumps(data), encoding="utf-8")
        index_file = tmp_path / "index.json"
        rc = self.mod.main(["--index", str(index_file), str(input_file)])
        assert rc == 0  # script exits 0 (skipped, not fatal)
        payload = json.loads(index_file.read_text())
        assert len(payload) == 1  # only valid row ingested
