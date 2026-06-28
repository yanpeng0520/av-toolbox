from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

import av_toolbox


def _write_preview_audio(path: Path, *, duration: float = 3.0, sr: int = 22050) -> Path:
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    for t0 in np.arange(0.0, duration, 0.5):
        i0 = int(t0 * sr)
        size = min(n - i0, int(0.12 * sr))
        if size <= 0:
            continue
        t = np.arange(size, dtype=np.float32) / sr
        y[i0:i0 + size] += 0.82 * np.sin(2 * np.pi * 70 * t) * np.exp(-t * 18)
        y[i0:i0 + size] += 0.18 * np.sin(2 * np.pi * 880 * t) * np.exp(-t * 36)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = 0.85 * y / peak
    pcm = np.clip(y * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return path


def _ffprobe_stream(path: Path, stream: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            stream,
            "-show_entries",
            "stream=codec_name,pix_fmt",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout)["streams"]
    assert streams, f"no {stream} stream in {path}"
    return streams[0]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for output codec validation",
)
def test_audio_overlay_mp4_uses_browser_friendly_codec_and_pixel_format(tmp_path) -> None:
    audio = _write_preview_audio(tmp_path / "beat.wav")

    result = av_toolbox.run_tool(
        "audio.beat_detection",
        input_path=audio,
        output_dir=tmp_path / "out",
        max_seconds=2.5,
        sample_rate=22050,
        overlay_fps=5.0,
        window_sec=2.0,
        overlay_width=480,
        overlay_height=270,
        device="cpu",
    )

    assert result.overlay_path is not None
    overlay = Path(result.overlay_path)
    assert overlay.exists()

    video_stream = _ffprobe_stream(overlay, "v:0")
    audio_stream = _ffprobe_stream(overlay, "a:0")

    assert video_stream["codec_name"] == "h264"
    assert video_stream["pix_fmt"] == "yuv420p"
    assert audio_stream["codec_name"] == "aac"
