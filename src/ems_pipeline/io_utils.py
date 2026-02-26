"""JSON I/O helpers for pipeline documents.

These helpers are intentionally thin wrappers around Pydantic's JSON support.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    """Read a JSON file into a Pydantic model instance."""

    path = Path(path)
    data = path.read_text(encoding="utf-8")
    return model_type.model_validate_json(data)


def write_model(path: str | Path, model: BaseModel) -> None:
    """Write a Pydantic model instance to a JSON file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

