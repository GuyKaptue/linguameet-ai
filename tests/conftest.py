"""Shared pytest fixtures.

Settings are isolated from the developer's real environment and any ``.env`` file:
every setting variable is removed from ``os.environ`` and the cwd is a temp dir.
"""

from __future__ import annotations

import pytest

from shared.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_settings_env(monkeypatch, tmp_path):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env is picked up
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
