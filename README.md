# EMS Ambient Pipeline (Skeleton)

Minimal Python 3.11+ project skeleton for an “EMS Ambient Pipeline” that turns an input audio file into:

1) diarized transcript with timestamps  
2) extracted EMS entities + context (negation/uncertainty)  
3) a structured proto-claim JSON

This repo provides schemas, CLI wiring, and tests. The actual ML/NLP implementations are intentionally left as TODOs.

## Layout

```
src/ems_pipeline/
  cli.py          # CLI entrypoints: transcribe/extract/build-claim
  models.py       # Pydantic schemas (Segment/Transcript/Entity/Claim/...)
  io_utils.py     # JSON read/write helpers
  transcribe.py   # TODO: audio -> Transcript
  extract.py      # TODO: Transcript -> EntitiesDocument
  claim.py        # TODO: EntitiesDocument -> Claim
tests/
  test_models_roundtrip.py
```

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Run tests:

```powershell
python -m pytest
```

## CLI usage (wiring only)

These commands exist, but the stage logic is not implemented yet (they will exit with code 2 until you implement the TODOs).

```powershell
python -m ems_pipeline transcribe .\sample.wav --out .\out\transcript.json
python -m ems_pipeline extract .\out\transcript.json --out .\out\entities.json
python -m ems_pipeline build-claim .\out\entities.json --out .\out\claim.json
```

## Dev commands

- With Make (if installed): `make format`, `make lint`, `make test`
- With PowerShell scripts: `.\scripts\format.ps1`, `.\scripts\lint.ps1`, `.\scripts\test.ps1`
