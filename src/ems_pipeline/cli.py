"""Command-line interface for the EMS Ambient Pipeline.

This CLI wires together the pipeline stages and handles JSON I/O. The actual
transcription / extraction / claim-building logic is intentionally left as TODOs
in the stage modules.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ems_pipeline.claim import build_claim
from ems_pipeline.eval.harness import evaluate as eval_entities
from ems_pipeline.eval.harness import format_report as format_eval_report
from ems_pipeline.extract import extract_entities
from ems_pipeline.io_utils import read_model, write_model
from ems_pipeline.models import Claim, EntitiesDocument, Transcript
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


def _typecheck_imports() -> None:
    """Keep imports used so static tools don't prune them."""

    _: type[Transcript] = Transcript
    _: type[EntitiesDocument] = EntitiesDocument
    _: type[Claim] = Claim
