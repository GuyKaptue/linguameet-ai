"""Central configuration for every LinguaMeet service.

All values come from environment variables (or a ``.env`` file) and are validated
at start-up so that a bad deployment fails immediately instead of mid-meeting.
Variable names are the upper-case field names, e.g. ``ASR_MODEL_SIZE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["auto", "cpu", "cuda"]
# Compute types accepted by CTranslate2 / faster-whisper that we expose.
ComputeType = Literal["auto", "int8", "int8_float16", "int16", "float16", "float32"]
BackendMode = Literal["real", "mock"]
TTSEngine = Literal["piper", "mock"]  # XTTS is not implemented (non-commercial model licence, untested here)

# Whisper model names accepted by faster-whisper (a local CT2 directory is also accepted).
WHISPER_SIZES = frozenset(
    {
        "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
        "large-v1", "large-v2", "large-v3", "large", "turbo", "large-v3-turbo",
        "distil-large-v3",
    }
)
VAD_SAMPLE_RATES = (8000, 16000)
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    """Raised when configuration is missing or inconsistent for the requested feature."""


@dataclass(frozen=True)
class LatencyBudget:
    """Per-stage and end-to-end latency budgets in milliseconds."""

    e2e_ms: int
    asr_ms: int
    nmt_ms: int
    tts_ms: int


class Settings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- mode -------------------------------------------------------------------
    backend_mode: BackendMode = "real"

    # --- audio / VAD ------------------------------------------------------------
    audio_sample_rate: int = 16000
    vad_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    vad_min_silence_ms: int = Field(default=300, ge=32, le=5000)
    vad_speech_pad_ms: int = Field(default=100, ge=0, le=1000)
    vad_min_speech_ms: int = Field(default=64, ge=32, le=1000)
    vad_max_segment_seconds: float = Field(default=15.0, gt=1.0, le=30.0)

    # --- ASR --------------------------------------------------------------------
    asr_model_size: str = "base"
    asr_device: Device = "auto"
    asr_compute_type: ComputeType = "auto"
    asr_beam_size: int = Field(default=1, ge=1, le=10)
    asr_min_chunk_seconds: float = Field(default=1.0, gt=0.1, le=5.0)
    asr_buffer_max_seconds: float = Field(default=30.0, ge=5.0, le=30.0)
    asr_local_agreement_n: int = Field(default=2, ge=1, le=4)

    # --- NMT --------------------------------------------------------------------
    nmt_model_dir: Path = Path("./models/nllb-200-distilled-600M-ct2")
    nmt_tokenizer: str = "facebook/nllb-200-distilled-600M"
    nmt_device: Device = "auto"
    nmt_compute_type: ComputeType = "auto"
    nmt_beam_size: int = Field(default=2, ge=1, le=10)
    nmt_max_batch_size: int = Field(default=16, ge=1, le=256)
    nmt_batch_window_ms: int = Field(default=20, ge=0, le=500)
    nmt_max_input_tokens: int = Field(default=512, ge=8, le=512)  # NLLB trained on <=512 tokens

    # --- TTS --------------------------------------------------------------------
    tts_engine: TTSEngine = "piper"
    tts_voices_dir: Path = Path("./models/tts")
    tts_output_sample_rate: int = Field(default=24000, ge=8000, le=48000)
    tts_chunk_ms: int = Field(default=100, ge=20, le=1000)

    # --- LiveKit ----------------------------------------------------------------
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    # --- latency budgets (ms) ---------------------------------------------------
    budget_e2e_ms: int = Field(default=2000, gt=0)
    budget_asr_ms: int = Field(default=1000, gt=0)
    budget_nmt_ms: int = Field(default=500, gt=0)
    budget_tts_ms: int = Field(default=500, gt=0)

    # --- misc -------------------------------------------------------------------
    log_level: str = "INFO"
    models_dir: Path = Path("./models")
    cache_dir: Path = Path("./cache")

    # ---------------------------------------------------------------- validators
    @field_validator("audio_sample_rate")
    @classmethod
    def _check_sample_rate(cls, v: int) -> int:
        if v not in VAD_SAMPLE_RATES:
            raise ValueError(f"audio_sample_rate must be one of {VAD_SAMPLE_RATES} (Silero VAD limit), got {v}")
        return v

    @field_validator("asr_model_size")
    @classmethod
    def _check_asr_model(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("asr_model_size must not be empty")
        # Either a known Whisper size, a Hugging Face repo id, or a local path.
        if v in WHISPER_SIZES or "/" in v or v.startswith((".", "~")):
            return v
        raise ValueError(
            f"asr_model_size '{v}' is not a known Whisper size {sorted(WHISPER_SIZES)} or a path/repo id"
        )

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in LOG_LEVELS:
            raise ValueError(f"log_level must be one of {LOG_LEVELS}, got {v}")
        return v

    @field_validator("livekit_url")
    @classmethod
    def _check_livekit_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("ws", "wss", "http", "https") or not parsed.netloc:
            raise ValueError(f"livekit_url must be ws(s):// or http(s):// URL, got '{v}'")
        return v.rstrip("/")

    @field_validator("models_dir", "cache_dir", "nmt_model_dir", "tts_voices_dir")
    @classmethod
    def _expand_path(cls, v: Path) -> Path:
        return v.expanduser()

    @model_validator(mode="after")
    def _check_budgets(self) -> "Settings":
        if self.budget_asr_ms + self.budget_nmt_ms > self.budget_e2e_ms:
            raise ValueError(
                "budget_asr_ms + budget_nmt_ms must not exceed budget_e2e_ms "
                f"({self.budget_asr_ms}+{self.budget_nmt_ms} > {self.budget_e2e_ms})"
            )
        return self

    # ------------------------------------------------------------------ helpers
    @property
    def budget(self) -> LatencyBudget:
        return LatencyBudget(
            e2e_ms=self.budget_e2e_ms,
            asr_ms=self.budget_asr_ms,
            nmt_ms=self.budget_nmt_ms,
            tts_ms=self.budget_tts_ms,
        )

    @property
    def is_mock(self) -> bool:
        return self.backend_mode == "mock"

    @property
    def vad_window_samples(self) -> int:
        """Silero VAD window: 512 samples at 16 kHz, 256 at 8 kHz (32 ms)."""
        return 512 if self.audio_sample_rate == 16000 else 256

    def resolve_device(self, requested: Device | None = None) -> str:
        """Return ``"cuda"`` or ``"cpu"``; ``"auto"`` picks CUDA when CTranslate2 sees a GPU."""
        device = requested or self.asr_device
        if device != "auto":
            return device
        return "cuda" if _cuda_available() else "cpu"

    def resolve_compute_type(self, requested: ComputeType, device: str) -> str:
        """``auto`` -> float16 on GPU, int8 on CPU; explicit values are returned unchanged."""
        if requested != "auto":
            return requested
        return "float16" if device == "cuda" else "int8"

    def require_livekit(self) -> None:
        """Fail fast when the gateway is started without LiveKit credentials."""
        missing = [
            name
            for name, value in (
                ("LIVEKIT_API_KEY", self.livekit_api_key),
                ("LIVEKIT_API_SECRET", self.livekit_api_secret),
            )
            if not value
        ]
        if missing:
            raise ConfigError(f"Missing required LiveKit settings: {', '.join(missing)}")

    def ensure_directories(self) -> None:
        for path in (self.models_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # library missing or driver problem -> CPU
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (clear with ``get_settings.cache_clear()`` in tests)."""
    return Settings()
