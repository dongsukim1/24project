# EMS Ambient Pipeline

Python 3.11+ prototype for EMS ambient documentation and claims workflow.

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
