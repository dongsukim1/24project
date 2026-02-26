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
