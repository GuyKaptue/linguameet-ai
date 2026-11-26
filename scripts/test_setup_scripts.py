"""Tests for the real-stack helper scripts (no network, no models needed)."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_setup  # noqa: E402
import download_models  # noqa: E402
from shared.config import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SECRET = "secret_that_is_at_least_32_characters_long"


def settings(tmp_path, **kw):
    return Settings(_env_file=None, livekit_api_key="devkey", livekit_api_secret=SECRET,
                    nmt_model_dir=tmp_path / "nmt", tts_voices_dir=tmp_path / "tts", **kw)


def test_preflight_fails_clearly_when_models_and_server_are_missing(tmp_path):
    with socket.socket() as sock:  # a port nobody listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    res = {n: (ok, d) for n, ok, d in check_setup.run(settings(tmp_path, livekit_url=f"ws://127.0.0.1:{port}"))}
    assert res["LiveKit key/secret"][0] and res["Token minting"][0]
    assert not res["NLLB CTranslate2 model"][0] and not res["Piper voices"][0]
    assert not res["LiveKit server reachable"][0] and "docker compose up livekit" in res["LiveKit server reachable"][1]


def test_preflight_passes_with_models_and_a_listening_server(tmp_path):
    (tmp_path / "nmt").mkdir()
    (tmp_path / "nmt" / "model.bin").write_bytes(b"x")
    (tmp_path / "tts").mkdir()
    (tmp_path / "tts" / "en_US-lessac-medium.onnx").write_bytes(b"x")
    (tmp_path / "tts" / "en_US-lessac-medium.onnx.json").write_text("{}")
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    try:
        results = check_setup.run(settings(tmp_path, livekit_url=f"ws://127.0.0.1:{srv.getsockname()[1]}"))
    finally:
        srv.close()
    assert all(ok for _, ok, _ in results), results


def test_preflight_flags_mock_mode_and_missing_keys(tmp_path):
    s = Settings(_env_file=None, backend_mode="mock", nmt_model_dir=tmp_path, tts_voices_dir=tmp_path)
    res = {n: ok for n, ok, _ in check_setup.check_settings(s)}
    assert res == {"LiveKit key/secret": False, "BACKEND_MODE": False}


def test_voice_configs_are_required(tmp_path):
    (tmp_path / "tts").mkdir()
    (tmp_path / "tts" / "es_ES-davefx-medium.onnx").write_bytes(b"x")
    res = {n: ok for n, ok, _ in check_setup.check_models(settings(tmp_path))}
    assert res["Piper voices"] and not res["Piper voice configs (.onnx.json)"]


def test_download_plan_lists_every_step_and_validates_languages(tmp_path):
    steps = dict(download_models.plan(tmp_path, "small", ["en", "es"], set()))
    assert set(steps) == {"asr", "nmt", "nmt-tokenizer", "tts"}
    assert "es_ES-davefx-medium" in steps["tts"] and "piper.download_voices" in steps["tts"]
    assert "--download-dir" in steps["tts"] and str(tmp_path / "tts") in steps["tts"]
    assert steps["nmt"][-1] == "int8" and "facebook/nllb-200-distilled-600M" in steps["nmt"]
    assert "'small'" in steps["asr"][-1]
    assert list(dict(download_models.plan(tmp_path, "base", ["en"], {"asr"}))) == ["asr"]
    with pytest.raises(SystemExit):
        download_models.plan(tmp_path, "base", ["xx"], {"tts"})


def test_compose_and_livekit_config_are_consistent():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    lk = yaml.safe_load((ROOT / "deploy" / "livekit.yaml").read_text())
    assert set(compose["services"]) == {"livekit", "gateway", "web"}
    env = compose["services"]["gateway"]["environment"]
    key, secret = next(iter(lk["keys"].items()))
    assert env["LIVEKIT_API_KEY"] == key and env["LIVEKIT_API_SECRET"] == secret  # gateway can authenticate
    assert len(secret) >= 32  # our Settings validator and LiveKit both want a long secret
    assert env["LIVEKIT_URL"] == f"ws://127.0.0.1:{lk['port']}"
    assert lk["rtc"]["tcp_port"] == 7881 and lk["rtc"]["port_range_start"] < lk["rtc"]["port_range_end"]
    Settings(_env_file=None, livekit_api_key=key, livekit_api_secret=secret, livekit_url=env["LIVEKIT_URL"])  # accepted


MAKE_TARGETS = ["help", "install", "install-torch", "env", "models", "up", "down", "check", "setup", "gateway", "web", "stack",
                "tokens", "demo", "test", "test-py", "test-js", "bench", "screenshots", "clean"]


@pytest.mark.skipif(__import__("shutil").which("make") is None, reason="make not installed")
def test_makefile_targets_exist_and_are_documented():
    import subprocess

    help_text = subprocess.run(["make", "-C", str(ROOT), "help"], capture_output=True, text=True, check=True).stdout
    for target in MAKE_TARGETS:
        assert f"make {target} " in help_text, target  # every target has a one-line description
        dry = subprocess.run(["make", "-C", str(ROOT), "-n", target], capture_output=True, text=True)
        assert dry.returncode == 0, (target, dry.stderr)


@pytest.mark.skipif(__import__("shutil").which("make") is None, reason="make not installed")
def test_make_setup_runs_the_steps_in_order_and_passes_variables():
    import subprocess

    out = subprocess.run(["make", "-C", str(ROOT), "-n", "setup", "LANGS=en,fr", "ASR=small", "ROOM=r1"], capture_output=True, text=True, check=True).stdout
    order = ["pip install -e", "download.pytorch.org", "cp .env.example", "download_models.py --langs en,fr --asr-size small", "docker compose up -d livekit", "check_setup.py --room r1"]
    positions = [out.index(x) for x in order]
    assert positions == sorted(positions)


def test_env_example_matches_the_livekit_config():
    lk = yaml.safe_load((ROOT / "deploy" / "livekit.yaml").read_text())
    key, secret = next(iter(lk["keys"].items()))
    env = dict(l.split("=", 1) for l in (ROOT / ".env.example").read_text().splitlines() if "=" in l and not l.startswith("#"))
    assert env["LIVEKIT_API_KEY"] == key and env["LIVEKIT_API_SECRET"] == secret
    assert env["LIVEKIT_URL"] == f"ws://127.0.0.1:{lk['port']}"
