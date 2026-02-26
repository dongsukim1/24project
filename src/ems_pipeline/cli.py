"""Command-line interface for the EMS Ambient Pipeline.

This CLI wires together the pipeline stages and handles JSON I/O. The actual
transcription / extraction / claim-building logic is intentionally left as TODOs
in the stage modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ems_pipeline.agents.agent1 import Agent1Options
from ems_pipeline.agents.agent1 import run as agent1_run
from ems_pipeline.agents.agent2 import Agent2Options
from ems_pipeline.agents.agent2 import run as agent2_run
from ems_pipeline.claim import build_claim
from ems_pipeline.eval.harness import evaluate as eval_entities
from ems_pipeline.eval.harness import format_report as format_eval_report
from ems_pipeline.extract import extract_entities
from ems_pipeline.io_utils import read_model, write_model
from ems_pipeline.models import Claim, EntitiesDocument, Transcript
from ems_pipeline.session import SessionContext
from ems_pipeline.transcribe import transcribe_audio

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("transcribe")
def transcribe_cmd(
    audio_path: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(..., "--out", "-o", help="Output transcript JSON path."),
    bandpass: bool = typer.Option(
        False,
        "--bandpass",
        help="Apply a telephone-like bandpass filter (~200–3400 Hz).",
    ),
    denoise: bool = typer.Option(
        False,
        "--denoise",
        help="Apply lightweight spectral-gate noise reduction.",
    ),
) -> None:
    """Convert audio into a diarized transcript with timestamps."""

    try:
        transcript = transcribe_audio(audio_path, bandpass=bandpass, denoise=denoise)
    except NotImplementedError as exc:
        typer.echo(f"Not implemented: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    write_model(out, transcript)


@app.command("extract")
def extract_cmd(
    transcript_json: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(..., "--out", "-o", help="Output entities JSON path."),
) -> None:
    """Extract EMS entities + context (negation/uncertainty) from a transcript JSON."""

    transcript = read_model(transcript_json, Transcript)

    try:
        entities_doc = extract_entities(transcript)
    except NotImplementedError as exc:
        typer.echo(f"Not implemented: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    write_model(out, entities_doc)


@app.command("build-claim")
def build_claim_cmd(
    entities_json: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(..., "--out", "-o", help="Output proto-claim JSON path."),
) -> None:
    """Build a structured proto-claim JSON from extracted entities."""

    entities_doc = read_model(entities_json, EntitiesDocument)

    try:
        claim = build_claim(entities_doc)
    except NotImplementedError as exc:
        typer.echo(f"Not implemented: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    write_model(out, claim)


@app.command("eval")
def eval_cmd(
    gold: Path = typer.Option(..., "--gold", help="Gold dataset JSON path."),
    pred: Path = typer.Option(..., "--pred", help="Predicted entities JSON path."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Optional output report JSON path."),
) -> None:
    """Evaluate predicted entities against a gold dataset."""

    report = eval_entities(gold, pred)
    formatted = format_eval_report(report)

    if out is None:
        typer.echo(formatted)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(formatted + "\n", encoding="utf-8")


@app.command("agent1")
def agent1_cmd(
    audio_path: Path = typer.Argument(..., exists=True, readable=True),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for session.json."),
    bandpass: bool = typer.Option(False, "--bandpass", help="Apply a telephone-like bandpass filter (~200–3400 Hz)."),
    denoise: bool = typer.Option(False, "--denoise", help="Apply lightweight spectral-gate noise reduction."),
    model: str = typer.Option("base", "--model", help="ASR model name (base, small, medium, large-v3)."),
    confidence_threshold: float = typer.Option(
        0.7,
        "--confidence-threshold",
        help="Segments with ASR confidence below this threshold are flagged.",
        min=0.0,
        max=1.0,
    ),
) -> None:
    """Transcribe audio and extract entities; write a SessionContext JSON."""

    session = SessionContext.create()
    options = Agent1Options(
        bandpass=bandpass,
        denoise=denoise,
        asr_model=model,
        confidence_threshold=confidence_threshold,
    )

    try:
        session = agent1_run(audio_path, session, options)
    except NotImplementedError as exc:
        typer.echo(f"Not implemented: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    session_path = out_dir / "session.json"
    session.to_json(session_path)

    # Emit a JSON summary to stdout so the MCP tool can parse the result.
    summary = {
        "encounter_id": session.encounter_id,
        "session_path": str(session_path),
        "confidence_flags_count": len(session.confidence_flags or []),
        "ambiguities_count": len(session.ambiguities or []),
    }
    typer.echo(json.dumps(summary))


@app.command("agent2")
def agent2_cmd(
    session_json: Path = typer.Argument(..., exists=True, readable=True),
    out_dir: Path = typer.Option(..., "--out-dir", help="Output directory for updated session.json."),
    payer_id: str | None = typer.Option(None, "--payer-id", help="Payer ID for requirements retrieval."),
) -> None:
    """Generate a clinical report and code suggestions; write updated SessionContext JSON."""

    session = SessionContext.from_json(session_json)
    options = Agent2Options(payer_id=payer_id)

    try:
        session = agent2_run(session, options)
    except NotImplementedError as exc:
        typer.echo(f"Not implemented: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    session_path = out_dir / "session.json"
    session.to_json(session_path)

    # Emit a JSON summary to stdout so the MCP tool can parse the result.
    summary = {
        "encounter_id": session.encounter_id,
        "session_path": str(session_path),
        "report_length": len(session.report_draft or ""),
        "codes_suggested": len(session.code_suggestions or []),
    }
    typer.echo(json.dumps(summary))


def _typecheck_imports() -> None:
    """Keep imports used so static tools don't prune them."""

    _: type[Transcript] = Transcript
    _: type[EntitiesDocument] = EntitiesDocument
    _: type[Claim] = Claim
