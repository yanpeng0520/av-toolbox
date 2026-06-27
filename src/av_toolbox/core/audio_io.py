"""Audio loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    path: str
    duration: float
    sample_rate: int
    channels: int
    samples: int

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "path": self.path,
            "duration": round(self.duration, 6),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "samples": self.samples,
        }


def _imports() -> tuple[Any, Any]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Audio helpers require librosa and NumPy. Install with: pip install -e '.[audio]'"
        ) from exc
    return librosa, np


def load_audio_samples(
    path: str | Path,
    *,
    sample_rate: int = 22050,
    max_seconds: float | None = None,
    normalize: bool = True,
) -> tuple[Any, int, AudioMetadata]:
    """Load mono audio from WAV/video containers and return samples plus metadata."""
    librosa, np = _imports()
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    load_kwargs: dict[str, Any] = {"sr": sample_rate, "mono": True}
    if max_seconds is not None:
        load_kwargs["duration"] = float(max_seconds)
    y, sr = librosa.load(str(audio_path), **load_kwargs)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise RuntimeError(f"No audio decoded from {audio_path}")
    if normalize:
        peak = float(np.max(np.abs(y)))
        if peak > 1e-8:
            y = y / peak

    metadata = AudioMetadata(
        path=str(audio_path),
        duration=float(y.size / sr),
        sample_rate=int(sr),
        channels=1,
        samples=int(y.size),
    )
    return y, int(sr), metadata
