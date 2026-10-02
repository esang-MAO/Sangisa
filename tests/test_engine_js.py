"""Runs the in-browser engine's parity tests (tests/engine/*.test.mjs) under Node."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js isn't installed")
def test_engine_matches_python(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "tests/engine/reference.py"), str(tmp_path)], check=True)
    result = subprocess.run(
        ["node", "--test", "--test-reporter=spec", str(ROOT / "tests/engine/engine.test.mjs")],
        env={**os.environ, "SANGISA_REF_DIR": str(tmp_path)},
        capture_output=True, text=True, timeout=900,
    )
    print(result.stdout[-6000:], result.stderr[-3000:])
    assert result.returncode == 0, "JavaScript engine tests failed (output above)"


def test_engine_config_matches_python_defaults():
    """site/engine/config.json is a copy of default_config.toml; regenerate it if this fails:
    uv run python -c "import json; from sangisa.config import default_config;
    json.dump(default_config(), open('site/engine/config.json', 'w'), indent=2)"
    """
    import json

    from sangisa.config import default_config

    assert json.loads((ROOT / "site/engine/config.json").read_text()) == default_config()
