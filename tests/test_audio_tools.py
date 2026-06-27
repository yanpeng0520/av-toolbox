from __future__ import annotations

import csv
import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

import av_toolbox


def _write_beat_track(path: Path, *, duration: float = 10.0, sr: int = 22050, bpm: float = 96.0) -> Path:
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    beat = 60.0 / bpm
    rng = np.random.default_rng(3)
    for idx, t0 in enumerate(np.arange(0.0, duration, beat)):
        i0 = int(t0 * sr)
        kick_len = min(n - i0, int(0.18 * sr))
        if kick_len <= 0:
            continue
        t = np.arange(kick_len, dtype=np.float32) / sr
        freq = 48 + 42 * np.exp(-t * 24)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        y[i0:i0 + kick_len] += 0.9 * np.sin(phase) * np.exp(-t * 10)
        if idx % 2 == 1:
            snare_len = min(n - i0, int(0.11 * sr))
            noise = rng.standard_normal(snare_len).astype(np.float32)
            y[i0:i0 + snare_len] += 0.35 * noise * np.exp(-np.arange(snare_len) / sr * 30)
    for t0 in np.arange(0.0, duration, beat / 2):
        i0 = int(t0 * sr)
        hat_len = min(n - i0, int(0.035 * sr))
        noise = rng.standard_normal(hat_len).astype(np.float32)
        y[i0:i0 + hat_len] += 0.12 * np.diff(noise, prepend=noise[0]) * np.exp(-np.arange(hat_len) / sr * 75)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = 0.85 * y / peak
    pcm = np.clip(y * 32767, -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return path


def _write_event_track(path: Path, *, duration: float = 10.0, sr: int = 22050) -> Path:
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(9)

    def add_tone(start: float, end: float, freq: float, amp: float) -> None:
        i0 = int(start * sr)
        i1 = min(n, int(end * sr))
        t = np.arange(max(0, i1 - i0), dtype=np.float32) / sr
        env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, np.maximum(0.0, (end - start) - t) / 0.05)
        y[i0:i1] += amp * np.sin(2 * np.pi * freq * t) * env

    def add_impact(start: float, amp: float = 1.0) -> None:
        i0 = int(start * sr)
        size = min(n - i0, int(0.16 * sr))
        if size <= 0:
            return
        t = np.arange(size, dtype=np.float32) / sr
        noise = rng.standard_normal(size).astype(np.float32)
        y[i0:i0 + size] += amp * (0.6 * noise + 0.4 * np.sin(2 * np.pi * 90 * t)) * np.exp(-t * 18)

    add_tone(1.1, 3.3, 180.0, 0.35)
    add_tone(3.4, 4.7, 520.0, 0.22)
    add_tone(6.7, 8.8, 260.0, 0.42)
    add_tone(8.8, 9.8, 390.0, 0.26)
    for t0 in (1.1, 2.4, 4.0, 6.7):
        add_impact(t0)
    y += 0.004 * rng.standard_normal(n).astype(np.float32)
    # Keep a clear low-energy region in the middle.
    y[int(5.0 * sr):int(6.3 * sr)] *= 0.01
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = 0.9 * y / peak
    pcm = np.clip(y * 32767, -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return path



def test_audio_tools_registered() -> None:
    names = [tool["name"] for tool in av_toolbox.list_tools()]
    assert "audio.beat_detection" in names
    assert "audio.event_detection" in names
    assert "audio.music_phase" in names


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for overlay muxing")
def test_beat_detection_creates_declared_artifacts_and_overlay(tmp_path) -> None:
    audio = _write_beat_track(tmp_path / "beat_track.wav")

    result = av_toolbox.run_tool(
        "audio.beat_detection",
        input_path=audio,
        output_dir=tmp_path / "out",
        max_seconds=6.0,
        sample_rate=22050,
        overlay_fps=5.0,
        window_sec=3.0,
        overlay_width=640,
        overlay_height=360,
        device="cpu",
    )

    assert result.overlay_path is not None
    assert result.timeline_json is not None
    assert result.csv_path is not None
    assert result.config_path is not None
    assert result.log_path is not None
    assert Path(result.overlay_path).exists()
    assert Path(result.timeline_json).exists()
    assert Path(result.csv_path).exists()

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "audio.beat_detection"
    assert payload["summary"]["beat_count"] > 0
    assert payload["summary"]["onset_count"] > 0

    with Path(result.csv_path).open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "tempo_bpm" in rows[0]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for overlay muxing")
def test_music_phase_creates_declared_artifacts_and_overlay(tmp_path) -> None:
    audio = _write_beat_track(tmp_path / "phase_track.wav", duration=12.0)

    result = av_toolbox.run_tool(
        "audio.music_phase",
        input_path=audio,
        output_dir=tmp_path / "out",
        max_seconds=10.0,
        sample_rate=22050,
        overlay_fps=5.0,
        window_sec=4.0,
        overlay_width=640,
        overlay_height=380,
        device="cpu",
    )

    assert result.overlay_path is not None
    assert result.timeline_json is not None
    assert result.csv_path is not None
    assert result.config_path is not None
    assert result.log_path is not None
    assert Path(result.overlay_path).exists()
    assert Path(result.timeline_json).exists()
    assert Path(result.csv_path).exists()

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "audio.music_phase"
    assert payload["summary"]["phase_count"] > 0
    assert payload["segments"]

    with Path(result.csv_path).open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "phase_label" in rows[0]



@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for overlay muxing")
def test_event_detection_creates_declared_artifacts_and_overlay(tmp_path) -> None:
    audio = _write_event_track(tmp_path / "event_track.wav")

    result = av_toolbox.run_tool(
        "audio.event_detection",
        input_path=audio,
        output_dir=tmp_path / "out",
        max_seconds=8.0,
        sample_rate=22050,
        overlay_fps=5.0,
        window_sec=3.0,
        overlay_width=640,
        overlay_height=380,
        impact_delta=0.2,
        spectral_delta=0.2,
        tonal_delta=0.2,
        silence_threshold=0.1,
        high_energy_quantile=0.72,
        min_region_seconds=0.2,
        device="cpu",
    )

    assert result.overlay_path is not None
    assert result.timeline_json is not None
    assert result.csv_path is not None
    assert result.config_path is not None
    assert result.log_path is not None
    assert Path(result.overlay_path).exists()
    assert Path(result.timeline_json).exists()
    assert Path(result.csv_path).exists()

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "audio.event_detection"
    assert payload["summary"]["event_count"] > 0
    assert payload["summary"]["impact_count"] > 0
    assert payload["summary"]["low_energy_count"] > 0
    assert payload["summary"]["high_energy_count"] > 0
    assert payload["energy_regions"]

    with Path(result.csv_path).open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "event_type" in rows[0]
