from __future__ import annotations

import csv
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

import av_toolbox


def _write_wav(path: Path, *, duration: float = 5.0, sr: int = 22050, events: tuple[float, ...] = (0.8, 1.6, 2.4, 3.2, 4.0)) -> Path:
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    for event_t in events:
        i0 = int(event_t * sr)
        size = min(n - i0, int(0.18 * sr))
        if size <= 0:
            continue
        t = np.arange(size, dtype=np.float32) / sr
        y[i0:i0 + size] += 0.95 * np.sin(2 * np.pi * 80 * t) * np.exp(-t * 16)
        y[i0:i0 + size] += 0.18 * np.sin(2 * np.pi * 900 * t) * np.exp(-t * 26)
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = 0.86 * y / peak
    pcm = np.clip(y * 32767, -32768, 32767).astype('<i2')
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return path


def _write_sync_clip(path: Path, *, duration: float = 5.0, fps: float = 10.0) -> Path:
    try:
        import cv2
    except ImportError as exc:
        pytest.skip(f"OpenCV unavailable: {exc}")
    wav_path = path.with_suffix('.wav')
    video_only = path.with_name(path.stem + '_video_only.mp4')
    events = (0.8, 1.6, 2.4, 3.2, 4.0)
    _write_wav(wav_path, duration=duration, events=events)

    size = (320, 240)
    writer = cv2.VideoWriter(str(video_only), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {video_only}")
    try:
        n_frames = int(duration * fps)
        for idx in range(n_frames):
            t = idx / fps
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            frame[:] = (18, 22, 26)
            x = int(40 + (idx % 24) * 8)
            cv2.circle(frame, (x, 120), 22, (70, 90, 130), -1)
            if any(abs(t - event_t) <= 0.055 for event_t in events):
                frame[:] = (210, 214, 220)
                cv2.rectangle(frame, (72, 64), (248, 176), (20, 80, 235), -1)
            cv2.putText(frame, f"t={t:.2f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()

    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(video_only),
        '-i', str(wav_path),
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '128k', '-shortest',
        str(path),
    ], check=True)
    return path


def test_sync_correspondence_registered() -> None:
    assert "av.sync_correspondence" in [tool["name"] for tool in av_toolbox.list_tools()]


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg is required for AV test clip and overlay muxing')
def test_sync_correspondence_creates_declared_artifacts_and_overlay(tmp_path) -> None:
    clip = _write_sync_clip(tmp_path / 'sync_clip.mp4')

    result = av_toolbox.run_tool(
        'av.sync_correspondence',
        input_path=clip,
        output_dir=tmp_path / 'out',
        max_seconds=4.6,
        sample_fps=10.0,
        sample_rate=22050,
        sync_window=0.2,
        overlay_fps=5.0,
        window_sec=2.0,
        overlay_width=640,
        overlay_height=420,
        device='cpu',
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
    assert payload['tool_name'] == 'av.sync_correspondence'
    assert payload['summary']['audio_event_count'] > 0
    assert payload['summary']['motion_peak_count'] > 0
    assert payload['summary']['sync_match_count'] > 0

    with Path(result.csv_path).open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert 'offset_seconds' in rows[0]
