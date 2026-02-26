# EMS Ambient Pipeline

A Python 3.11+ prototype pipeline for EMS ambient documentation:

1. **Transcribe** audio into timestamped segments with speaker labels.  
2. **Extract** EMS entities from the transcript with lightweight normalization.  
3. **Build** a structured proto-claim JSON with provenance links.

This repository now includes substantial implementation for stages (1) and (2), plus a more complete claim-builder implementation in `ems_pipeline.claim.builder`. However, there are still integration and data gaps before a full end-to-end real-world demo.

## What currently exists

### Implemented modules

- **Data models**: Pydantic schemas for transcript, entities, and claim documents.
- **Audio preprocessing**:
  - WAV loading/resampling/chunking utilities.
  - Optional bandpass filter and spectral-gate denoise.
- **ASR adapter**:
  - Offline Whisper integration via `faster-whisper` (preferred) or `openai-whisper`.
  - Local-only model expectation (no runtime download logic in app flow).
- **Diarization**:
  - Baseline fallback diarizer that currently behaves as a simple/default strategy.
- **Entity extraction**:
  - Rule-based extractor for selected EMS terms (e.g., vitals, symptoms, procedures, ETA, unit IDs).
  - Lexicon-backed normalization.
- **Timeline + claim assembly**:
  - `ems_pipeline.claim.timeline` event extraction.
  - `ems_pipeline.claim.builder` proto-claim construction with evidence links.
- **Evaluation harness**:
  - PRF by entity type and key-term coverage over a gold dataset format.
- **Tests**:
  - Unit/integration-style tests spanning models, ASR adapters, extraction, timeline, claim builder, audio, and eval metrics.

## Important current limitation

The CLI command `build-claim` imports `build_claim` from `ems_pipeline.claim` (package `__init__`), and that function currently raises `NotImplementedError`. The fuller implementation lives in `ems_pipeline.claim.builder`.

So, **CLI-based end-to-end runs currently stop at claim-building unless this wiring is updated**.

## Running locally

### 1) Create environment

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
python -m pip install -e '.[dev]'
```

### 2) Configure optional ASR model selection

```bash
export EMS_ASR_MODEL=base
# or: small, medium, large-v3
```

### 3) Run tests

```bash
pytest
```

### 4) CLI usage

```bash
python -m ems_pipeline transcribe ./sample.wav --out ./out/transcript.json
python -m ems_pipeline extract ./out/transcript.json --out ./out/entities.json
python -m ems_pipeline build-claim ./out/entities.json --out ./out/claim.json
python -m ems_pipeline eval --gold ./data/gold.json --pred ./out/pred.json
```

## What is still missing for a full demo

Beyond "no Whisper/ASR model connected" and "no real audio/CAD files", these are the major gaps:

1. **Claim stage CLI wiring gap**
   - `build-claim` currently points at a `NotImplementedError` function path.
   - The implemented claim builder exists but is not wired into the stable CLI path.

2. **Runtime ASR dependencies + local model cache**
   - You need one backend installed (`faster-whisper` or `openai-whisper`).
   - You need local Whisper model weights available in cache/model dir.

3. **Dependency/bootstrap reliability in constrained environments**
   - The project uses modern Python deps (`pydantic`, `numpy`, etc.) and build backend tooling.
   - In restricted networks, dependency install may fail unless mirrors/proxy are configured.

4. **Real diarization quality layer**
   - Current diarization path is baseline/fallback and likely insufficient for realistic multi-speaker call audio.

5. **Extraction coverage depth**
   - Extraction is rule-based and intentionally narrow; no robust negation/uncertainty or advanced ontology coverage yet.

6. **Evaluation/prediction data contract in demo scripts**
   - `eval` expects specific JSON structures; demo assets/pred generation scripts need to output those shapes consistently.

7. **Operational/demo assets**
   - No sample end-to-end demo script/notebook that runs complete ingest -> output using packaged example inputs.
   - No packaged synthetic audio fixtures representative of real call conditions.

8. **Production hardening**
   - No service/API deployment layer, auth, observability, or persistent storage workflow.
   - No explicit PHI/PII handling policy implementation in code paths.

## Development commands

- Make:
  - `make format`
  - `make lint`
  - `make test`
- PowerShell scripts:
  - `./scripts/format.ps1`
  - `./scripts/lint.ps1`
  - `./scripts/test.ps1`
