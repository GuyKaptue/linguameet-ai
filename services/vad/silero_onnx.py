"""Torch-free runner for the Silero VAD ONNX model.

The call convention mirrors ``OnnxWrapper`` in the ``silero-vad`` 6.x package
(github.com/snakers4/silero-vad): the model takes ``[context | window]`` where the
context is the last 64 samples (16 kHz) / 32 samples (8 kHz) of the previous window,
plus a recurrent ``state`` of shape (2, batch, 128) and the sampling rate. Output is
one speech probability per window (512 samples at 16 kHz, 256 at 8 kHz = 32 ms).

The model file ``models/silero_vad.onnx`` is taken verbatim from the silero-vad 6.2.3
wheel on PyPI (MIT license, see ``models/SILERO_LICENSE.txt``).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "silero_vad.onnx"
_WINDOW = {16000: 512, 8000: 256}
_CONTEXT = {16000: 64, 8000: 32}


class VADModel(Protocol):
    """Anything that scores one fixed-size window and keeps recurrent state."""

    sample_rate: int
    window_samples: int

    def __call__(self, window: np.ndarray) -> float: ...

    def reset_states(self) -> None: ...


_SESSIONS: dict[str, ort.InferenceSession] = {}
_SESSIONS_LOCK = threading.Lock()


def _get_session(path: Path) -> ort.InferenceSession:
    """One ONNX session per model file, shared by all streams (``run`` is thread-safe)."""
    key = str(path.resolve())
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(key)
        if session is None:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.log_severity_level = 3  # silence harmless "unused initializer" warnings
            session = ort.InferenceSession(key, sess_options=opts, providers=["CPUExecutionProvider"])
            _SESSIONS[key] = session
        return session


class SileroOnnxModel:
    """Stateful Silero VAD. One instance per audio stream (it carries the LSTM state);
    the underlying ONNX session is shared between instances."""

    def __init__(self, sample_rate: int = 16000, model_path: Path | str = DEFAULT_MODEL_PATH):
        if sample_rate not in _WINDOW:
            raise ValueError(f"Silero VAD supports {sorted(_WINDOW)} Hz, got {sample_rate}")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Silero VAD model not found: {path}")
        self._session = _get_session(path)
        self.sample_rate = sample_rate
        self.window_samples = _WINDOW[sample_rate]
        self._context_size = _CONTEXT[sample_rate]
        self._sr = np.array(sample_rate, dtype=np.int64)
        self.reset_states()

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_size), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        window = np.asarray(window, dtype=np.float32)
        if window.shape != (self.window_samples,):
            raise ValueError(
                f"expected a {self.window_samples}-sample 1-D window at {self.sample_rate} Hz, got shape {window.shape}"
            )
        x = np.concatenate([self._context, window[None, :]], axis=1)
        out, state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._state = state
        self._context = x[:, -self._context_size :]
        return float(out[0, 0])
