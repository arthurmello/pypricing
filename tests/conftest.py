"""Shared pytest fixtures and path-based markers."""

from __future__ import annotations

from pathlib import Path

import pytest

# Short MCMC defaults for integration smoke tests.
SMOKE_DRAWS = 40
SMOKE_TUNE = 40
SMOKE_CHAINS = 2

# Defaults for recovery suite (also hardcoded in recovery tests for clarity).
RECOVERY_DRAWS = 300
RECOVERY_TUNE = 300
RECOVERY_CHAINS = 2


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: fast tests without heavy MCMC")
    config.addinivalue_line(
        "markers", "integration: short MCMC smoke / end-to-end checks"
    )
    config.addinivalue_line(
        "markers", "recovery: statistical parameter recovery (slower)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark tests by directory under tests/."""
    root = Path(__file__).resolve().parent
    for item in items:
        try:
            rel = Path(item.path).resolve().relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if not parts:
            continue
        folder = parts[0]
        if folder == "unit":
            item.add_marker(pytest.mark.unit)
        elif folder == "integration":
            item.add_marker(pytest.mark.integration)
        elif folder == "recovery":
            item.add_marker(pytest.mark.recovery)
