"""Module 1: configuration loading, validation and path handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.config import ConfigError, LatencyBudget, Settings, get_settings


def make(**overrides) -> Settings:
    """Build Settings ignoring any .env file."""
    return Settings(_env_file=None, **overrides)


# ----------------------------------------------------------------------- defaults
def test_defaults_match_documented_values():
    s = make()
    assert s.backend_mode == "real"
    assert s.audio_sample_rate == 16000
    assert s.vad_window_samples == 512  # 32 ms at 16 kHz (Silero requirement)
    assert s.asr_model_size == "base"
    assert s.asr_local_agreement_n == 2
    assert s.asr_buffer_max_seconds == 30.0
    assert s.nmt_max_input_tokens == 512  # NLLB training limit
    assert s.nmt_tokenizer == "facebook/nllb-200-distilled-600M"
    assert s.budget == LatencyBudget(e2e_ms=2000, asr_ms=1000, nmt_ms=500, tts_ms=500)
    assert not s.is_mock


def test_vad_window_for_8khz():
    assert make(audio_sample_rate=8000).vad_window_samples == 256


# ----------------------------------------------------------------- env variables
def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("ASR_MODEL_SIZE", "small")
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    monkeypatch.setenv("ASR_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("NMT_BEAM_SIZE", "4")
    monkeypatch.setenv("BACKEND_MODE", "mock")
    s = make()
    assert (s.asr_model_size, s.asr_device, s.asr_compute_type) == ("small", "cpu", "int8")
    assert s.nmt_beam_size == 4
    assert s.is_mock


def test_env_names_are_case_insensitive(monkeypatch):
    monkeypatch.setenv("log_level", "debug")
    assert make().log_level == "DEBUG"


def test_dotenv_file_is_loaded(tmp_path):
    env = tmp_path / "test.env"
    env.write_text("ASR_MODEL_SIZE=medium\nTTS_ENGINE=mock\nLIVEKIT_URL=wss://meet.example.com/\n")
    s = Settings(_env_file=env)
    assert s.asr_model_size == "medium"
    assert s.tts_engine == "mock"
    assert s.livekit_url == "wss://meet.example.com"  # trailing slash stripped


def test_real_environment_beats_dotenv(monkeypatch, tmp_path):
    env = tmp_path / "test.env"
    env.write_text("ASR_MODEL_SIZE=medium\n")
    monkeypatch.setenv("ASR_MODEL_SIZE", "tiny")
    assert Settings(_env_file=env).asr_model_size == "tiny"


def test_unknown_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("SOME_UNRELATED_VARIABLE", "x")
    make()  # must not raise


def test_env_example_is_valid(monkeypatch):
    """The shipped .env.example must itself be a loadable configuration."""
    example = Path(__file__).resolve().parents[1] / ".env.example"
    s = Settings(_env_file=example)
    assert s.asr_device == "auto"
    assert s.livekit_api_key == "devkey"
    s.require_livekit()  # credentials present


# -------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "field,value",
    [
        ("audio_sample_rate", 44100),
        ("vad_threshold", 0.0),
        ("vad_threshold", 1.0),
        ("asr_model_size", "gigantic"),
        ("asr_model_size", ""),
        ("asr_device", "tpu"),
        ("asr_compute_type", "int4"),
        ("asr_beam_size", 0),
        ("asr_local_agreement_n", 0),
        ("asr_buffer_max_seconds", 60),
        ("nmt_max_input_tokens", 1024),
        ("nmt_max_batch_size", 0),
        ("tts_engine", "festival"),
        ("log_level", "LOUD"),
        ("livekit_url", "ftp://example.com"),
        ("livekit_url", "localhost:7880"),
        ("budget_e2e_ms", 0),
        ("backend_mode", "maybe"),
    ],
)
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_invalid_env_value_is_rejected(monkeypatch):
    monkeypatch.setenv("AUDIO_SAMPLE_RATE", "12345")
    with pytest.raises(ValidationError, match="audio_sample_rate"):
        make()


@pytest.mark.parametrize("size", ["tiny", "large-v3", "turbo", "distil-large-v3", "org/my-ct2-model", "./local/ct2"])
def test_valid_asr_model_names(size):
    assert make(asr_model_size=size).asr_model_size == size


def test_stage_budgets_must_fit_in_e2e_budget():
    with pytest.raises(ValidationError, match="must not exceed budget_e2e_ms"):
        make(budget_e2e_ms=1000, budget_asr_ms=800, budget_nmt_ms=400)


def test_log_level_is_normalised():
    assert make(log_level=" warning ").log_level == "WARNING"


# ------------------------------------------------------------------ model paths
def test_model_paths_are_pathlib_and_expand_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = make(models_dir="~/m", nmt_model_dir="~/m/nllb", tts_voices_dir="~/v")
    assert s.models_dir == tmp_path / "m"
    assert s.nmt_model_dir == tmp_path / "m" / "nllb"
    assert s.tts_voices_dir == tmp_path / "v"
    assert isinstance(s.cache_dir, Path)


def test_ensure_directories_creates_them(tmp_path):
    s = make(models_dir=tmp_path / "a" / "models", cache_dir=tmp_path / "b" / "cache")
    s.ensure_directories()
    assert (tmp_path / "a" / "models").is_dir()
    assert (tmp_path / "b" / "cache").is_dir()
    s.ensure_directories()  # idempotent


# --------------------------------------------------------------- device / dtype
def test_explicit_device_is_not_overridden():
    s = make()
    assert s.resolve_device("cpu") == "cpu"
    assert s.resolve_device("cuda") == "cuda"


def test_auto_device_follows_cuda_availability(monkeypatch):
    import shared.config as cfg

    monkeypatch.setattr(cfg, "_cuda_available", lambda: True)
    assert make().resolve_device("auto") == "cuda"
    monkeypatch.setattr(cfg, "_cuda_available", lambda: False)
    assert make().resolve_device("auto") == "cpu"


def test_cuda_probe_returns_bool_without_raising():
    from shared.config import _cuda_available

    assert isinstance(_cuda_available(), bool)


def test_compute_type_auto_resolution():
    s = make()
    assert s.resolve_compute_type("auto", "cuda") == "float16"
    assert s.resolve_compute_type("auto", "cpu") == "int8"
    assert s.resolve_compute_type("float32", "cpu") == "float32"


# ----------------------------------------------------------------------- livekit
def test_require_livekit_reports_missing_credentials():
    s = make()
    with pytest.raises(ConfigError, match="LIVEKIT_API_KEY, LIVEKIT_API_SECRET"):
        s.require_livekit()
    with pytest.raises(ConfigError, match="LIVEKIT_API_SECRET"):
        make(livekit_api_key="k").require_livekit()
    make(livekit_api_key="k", livekit_api_secret="s").require_livekit()


# --------------------------------------------------------------------- singleton
def test_get_settings_is_cached_and_clearable(monkeypatch):
    a = get_settings()
    assert get_settings() is a
    monkeypatch.setenv("ASR_MODEL_SIZE", "tiny")
    assert get_settings().asr_model_size == "base"  # still cached
    get_settings.cache_clear()
    assert get_settings().asr_model_size == "tiny"
