from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: integration test (requires optional dependencies and local model cache)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Default behavior: integration tests are allowed locally, but are skipped in CI
    # (or when explicitly disabled).
    if os.getenv("CI") or os.getenv("EMS_SKIP_INTEGRATION_TESTS"):
        skip = pytest.mark.skip(reason="Skipping integration tests in CI.")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)

