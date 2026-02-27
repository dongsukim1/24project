# EMS Ambient Pipeline

Prototype for EMS ambient documentation and claims workflow.

Core pipeline:
1. `transcribe`: audio -> diarized transcript segments
2. `extract`: transcript -> structured EMS entities
3. `build-claim`: entities -> proto-claim JSON with provenance

The repository also includes a 4-agent orchestration flow (`agent1`..`agent4`, plus `run`) built around `SessionContext`.

## Setup

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## CLI Commands

Show all commands:

```bash
python -m ems_pipeline --help
```

Stage-by-stage:

```bash
python -m ems_pipeline transcribe ./sample.wav --out ./out/transcript.json
python -m ems_pipeline extract ./out/transcript.json --out ./out/entities.json
python -m ems_pipeline build-claim ./out/entities.json --out ./out/claim.json
```

Canonical claim (schema v1.0) and export:

```bash
# Build a canonical v1.0 claim (requires both entities + transcript)
python -m ems_pipeline build-claim ./out/entities.json \
    --transcript ./out/transcript.json \
    --schema-version 1.0 \
    --out ./out/canonical.json

# Export to NEMSIS v3.5 data-element dict
python -m ems_pipeline export-claim ./out/canonical.json --format nemsis --out ./out/nemsis.json

# Export to X12 837P structured representation
python -m ems_pipeline export-claim ./out/canonical.json --format x12 --out ./out/837p.json

# Export to FHIR R4 Bundle
python -m ems_pipeline export-claim ./out/canonical.json --format fhir --out ./out/fhir_bundle.json

# Cross-format coverage report (JSON, machine-readable)
python -m ems_pipeline export-claim ./out/canonical.json --format coverage --out ./out/coverage.json
```

Evaluation:

```bash
python -m ems_pipeline eval --gold ./data/gold.json --pred ./out/pred.json
```

Agent/orchestrator flow:

```bash
python -m ems_pipeline agent1 ./sample.wav --out-dir ./out/a1
python -m ems_pipeline agent2 ./out/a1/session.json --out-dir ./out/a2
python -m ems_pipeline agent3 ./out/a2/session.json --out-dir ./out/a3
python -m ems_pipeline agent4 ./out/a3/session.json --out-dir ./out/a4
python -m ems_pipeline run ./sample.wav --out-dir ./out/run
```

## Canonical Claim Schema (v1.0)

The v1.0 `CanonicalClaim` (in `ems_pipeline.claim.canonical`) provides fully-structured
sections for downstream export to NEMSIS, X12 837P, and FHIR R4.

### Sections

| Section | Model | Maps to |
|---|---|---|
| A — Patient | `PatientInfo` | NEMSIS ePatient, X12 2000C, FHIR Patient |
| B — Subscriber/payer | `SubscriberInfo` | X12 2000B, FHIR Coverage |
| C — Encounter/admin | `EncounterInfo` | NEMSIS eResponse, FHIR Encounter |
| D — Crew/unit | `UnitInfo`, `CrewMember` | NEMSIS eResponse.07, X12 NM1*82 |
| E — Timeline | `IncidentTimeline` | NEMSIS eTimes |
| F — Clinical | `ClinicalInfo`, `VitalSign`, `Medication`, `Procedure` | NEMSIS eSituation/eVitals/eProcedures/eMedications, FHIR Observation/Procedure/MedicationAdministration |
| G — Transport | `TransportInfo` | NEMSIS eDisposition, X12 CLM05 |
| H — Billing | `BillingInfo`, `ServiceLine` | X12 837P SV1/HI, FHIR Claim |
| J — Provenance | `CanonicalProvenance` | Traceability back to transcript segment IDs |

### Example canonical claim structure

```json
{
  "schema_version": "1.0",
  "claim_id": "<sha256-hash>",
  "patient": { "full_name": null, "sex_at_birth": "M", "age_years": 54 },
  "subscriber": { "payer_id": "PAYER01", "member_id": "M123" },
  "encounter": { "run_number": "RUN-001", "agency_name": "Metro EMS" },
  "unit": { "unit_id": "UNIT12" },
  "timeline": { "psap_call_datetime": null, "patient_contact_datetime": null },
  "clinical": {
    "chief_complaint": "chest pain",
    "vitals": [{ "name": "BP", "value": "150/90", "unit": "mmHg" }],
    "medications": [{ "drug": "aspirin 324 mg" }],
    "procedures": [{ "name": "oxygen" }]
  },
  "transport": { "disposition_status": "transported", "destination_facility_name": "Mercy Hospital" },
  "billing": {
    "claim_type": "professional",
    "place_of_service": "41",
    "icd10_diagnoses": [{ "code": "R07.9", "code_system": "ICD-10-CM" }],
    "service_lines": [{ "line_number": 1, "procedure_code": "A0427", "units": 1.0 }]
  },
  "provenance": [{ "segment_id": "seg_0002", "field_path": "patient.age_years" }]
}
```

See `tests/fixtures/canonical_claim_sample.json` for a complete annotated example.

### Migration from v0.1 proto-claim

| v0.1 `fields` key | v1.0 canonical field | Notes |
|---|---|---|
| `fields.patient.age_hint` | `patient.age_years` + `patient.age_hint_text` | Age parsed to integer |
| `fields.patient.sex_hint` | `patient.sex_at_birth` | Normalized to "M"/"F"/"U" |
| `fields.primary_impression` | `clinical.chief_complaint` + `clinical.primary_impression` | Impression gets ICD code stub |
| `fields.findings.vitals[]` | `clinical.vitals[]` | Typed `VitalSign` with name/value/unit |
| `fields.findings.symptoms[]` | `clinical.symptoms[]` | Plain string list |
| `fields.interventions[]` | `clinical.medications[]` + `clinical.procedures[]` | Split by entity type |
| `fields.disposition.status` | `transport.disposition_status` | Same vocabulary |
| `fields.disposition.destination_hint` | `transport.destination_facility_name` | |
| `fields.location_hint` | `transport.origin_address.street` | |
| `fields.dispatch.units[]` | `unit.unit_id` | First unit used |
| `provenance[]` | `provenance[]` | Promoted to `CanonicalProvenance` with `field_path` |

The v0.1 `Claim` model in `ems_pipeline.models` is **unchanged** — both versions
coexist.  `build-claim` with `--schema-version 0.1` (default) produces the legacy model;
`--schema-version 1.0` produces `CanonicalClaim`.

### Known limitations (v1.0 MVP)

- **ICD-10 codes are placeholders** — `primary_impression.code = "UNKNOWN"` until Agent 2
  provides a mapped code via `agent2_codes`.
- **Timestamps are all `null`** — NEMSIS eTimes fields require absolute datetime values.
  The current pipeline does not yet convert relative audio timestamps to wall-clock time.
- **NPI / Tax ID fields** are always `null` — no provider directory lookup implemented.
- **Subscriber identity** — only `payer_id` forwarded from Agent 3 options; member/group IDs
  not yet extracted from audio.
- **Race/ethnicity** — entity types not yet extracted; fields default to empty list / null.
- **FHIR profiles** — output uses base R4 resources; US Core / CARIN BlueButton profile
  constraints not yet validated.
- **X12 837 EDI string** — the exporter produces a structured dict, not a raw EDI segment
  string.  A separate EDI formatter layer is needed for submission.

## Runtime Notes

- `transcribe` requires a local Whisper backend:
  - `faster-whisper` (preferred) or
  - `openai-whisper`
- ASR model name comes from `EMS_ASR_MODEL` (`base`, `small`, `medium`, `large-v3`).
- Payer-rules retrieval reads from `EMS_PAYER_RULES_INDEX` when set.
- Coding-guidelines retrieval reads from `EMS_CODING_GUIDELINES_INDEX` when set.
- NLP extraction uses `gliner` (`Ihor/gliner-biomed-base-v1.0` by default).
- GLiNER weights download from Hugging Face on first model load.
- GLiNER runs on CPU by default; no GPU is required.
- GLiNER confidence threshold defaults to `0.5`.
- Negation/context handling is a v1 ConText-inspired rule set with known scope/syntax limitations.
- The v2 dependency-parse negation idea is documented only (no implementation yet).
- No `spacy` dependency is required for the current NLP/negation path.
- This repository is a prototype and is not production-complete.

### Diarization

Speaker diarization is optional and controlled by environment variables.  Without any
configuration the pipeline assigns every word to a single speaker (`spk0`).

| Env var | Values | Default | Description |
|---|---|---|---|
| `EMS_DIARIZE_BACKEND` | `pyannote` / unset | *(fallback)* | Diarization backend to use. |
| `EMS_PYANNOTE_AUTH_TOKEN` | HuggingFace token | *(none)* | Required when `EMS_DIARIZE_BACKEND=pyannote`. |
| `EMS_DIARIZE_OVERLAP_POLICY` | `reject` / `normalize` / `allow` | `normalize` | How to handle overlapping speaker turns from the backend. |

**Backends:**

- **Fallback (default)** — No extra dependency.  Assigns all audio to `spk0`.  Suitable
  for single-speaker recordings or when diarization is not needed.

- **pyannote** (`EMS_DIARIZE_BACKEND=pyannote`) — Uses
  [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) speaker-diarization-3.1.
  Requires:
  1. `pip install pyannote.audio`
  2. A HuggingFace account with access to `pyannote/speaker-diarization-3.1` (request at
     [hf.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)).
  3. `EMS_PYANNOTE_AUTH_TOKEN=<your-hf-token>` in the environment.

  If the package is missing or the token is not set, a `RuntimeWarning` is emitted and the
  pipeline falls back to the single-speaker baseline — no hard crash.

**Overlap policies** (apply to backend output before transcript segmentation):

- `reject` — Raises `ValueError` on any overlapping turn.  Use for curated / pre-validated
  diarization data.
- `normalize` *(default)* — Trims overlapping turns by moving the later turn's start time
  to the previous turn's end.  Turns that shrink to zero duration are dropped.
- `allow` — Passes overlapping turns through unchanged.  Downstream code must handle the
  overlap (e.g. the speaker-assignment logic uses max-overlap matching).

**Speaker assignment** — Each ASR word is assigned to the speaker turn with the greatest
time overlap with the word interval `[word.start, word.end]`.  Words outside all turn
boundaries fall back to a nearest-turn midpoint lookup.

**Transcript metadata** — `transcript.metadata["diarization"]` contains:

```json
{
  "backend": "pyannote",
  "policy": "normalize",
  "num_speakers": 2
}
```

## Data And Tests

- Gold eval dataset: `data/gold.json`
- Test suite:

```bash
python -m pytest -q
```

## Development Commands

Make targets:

- `make format`
- `make lint`
- `make test`

PowerShell scripts:

- `./scripts/format.ps1`
- `./scripts/lint.ps1`
- `./scripts/test.ps1`
