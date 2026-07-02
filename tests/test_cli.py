from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from av_toolbox.cli import main


def _write_cli_audio(path: Path, duration: float = 6.0, sr: int = 22050) -> Path:
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float32)
    beat = 60.0 / 96.0
    for t0 in np.arange(0.0, duration, beat):
        i0 = int(t0 * sr)
        size = min(n - i0, int(0.12 * sr))
        t = np.arange(size, dtype=np.float32) / sr
        y[i0:i0 + size] += 0.85 * np.sin(2 * np.pi * 70 * t) * np.exp(-t * 18)
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


def _write_cli_sync_clip(path: Path, duration: float = 4.0, fps: float = 10.0) -> Path:
    try:
        import cv2
    except ImportError as exc:
        pytest.skip(f"OpenCV unavailable: {exc}")
    audio_path = _write_cli_audio(path.with_suffix('.wav'), duration=duration)
    video_only = path.with_name(path.stem + '_video_only.mp4')
    writer = cv2.VideoWriter(str(video_only), cv2.VideoWriter_fourcc(*'mp4v'), fps, (240, 180))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {video_only}")
    events = (0.625, 1.25, 1.875, 2.5, 3.125)
    try:
        for idx in range(int(duration * fps)):
            t = idx / fps
            frame = np.zeros((180, 240, 3), dtype=np.uint8)
            frame[:] = (18, 22, 26)
            if any(abs(t - event_t) <= 0.055 for event_t in events):
                frame[:] = (230, 230, 230)
                cv2.rectangle(frame, (55, 44), (185, 136), (40, 80, 220), -1)
            writer.write(frame)
    finally:
        writer.release()
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(video_only),
        '-i', str(audio_path),
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-c:a', 'aac', '-b:a', '128k', '-shortest',
        str(path),
    ], check=True)
    return path



def test_cli_list_tools_outputs_json(capsys) -> None:
    assert main(["list-tools"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip().startswith("[")
    assert "audio.beat_detection" in captured.out
    assert "audio.event_detection" in captured.out
    assert "audio.music_phase" in captured.out
    assert "av.denseav" in captured.out
    assert "av.sync_correspondence" in captured.out
    assert "video.image_quality" in captured.out
    assert "video.motion" in captured.out


def test_cli_video_image_quality_smoke(tmp_path, capsys, demo_video_path: Path) -> None:
    assert main([
        "video",
        "image-quality",
        str(demo_video_path),
        "--output",
        str(tmp_path),
        "--sample-fps",
        "2",
        "--max-seconds",
        "0.5",
        "--device",
        "cpu",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["tool_name"] == "video.image_quality"
    assert Path(payload["timeline_json"]).exists()


def test_cli_video_motion_smoke(tmp_path, capsys, demo_video_path: Path) -> None:
    assert main([
        "video",
        "motion",
        str(demo_video_path),
        "--output",
        str(tmp_path),
        "--sample-fps",
        "4",
        "--max-seconds",
        "1",
        "--device",
        "cpu",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["tool_name"] == "video.motion"
    assert Path(payload["timeline_json"]).exists()


def test_cli_video_cut_detection_smoke(tmp_path, capsys, demo_video_path: Path) -> None:
    assert main([
        "video",
        "cut-detection",
        str(demo_video_path),
        "--output",
        str(tmp_path),
        "--max-seconds",
        "1",
        "--backend",
        "lightweight",
        "--device",
        "cpu",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["tool_name"] == "video.cut_detection"
    assert Path(payload["timeline_json"]).exists()


def test_cli_audio_beat_detection_smoke(tmp_path, capsys) -> None:
    audio = _write_cli_audio(tmp_path / "cli_audio.wav")

    assert main([
        "audio",
        "beat-detection",
        str(audio),
        "--output",
        str(tmp_path / "out"),
        "--max-seconds",
        "4",
        "--sample-rate",
        "22050",
        "--device",
        "cpu",
        "--no-overlay",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["tool_name"] == "audio.beat_detection"
    assert Path(payload["timeline_json"]).exists()
    assert payload["overlay_path"] is None



def test_cli_audio_event_detection_smoke(tmp_path, capsys) -> None:
    audio = _write_cli_audio(tmp_path / "cli_event_audio.wav")

    assert main([
        "audio",
        "event-detection",
        str(audio),
        "--output",
        str(tmp_path / "out_events"),
        "--max-seconds",
        "4",
        "--sample-rate",
        "22050",
        "--device",
        "cpu",
        "--no-overlay",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["tool_name"] == "audio.event_detection"
    assert Path(payload["timeline_json"]).exists()
    assert payload["overlay_path"] is None



@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg is required for AV CLI smoke clip')
def test_cli_av_sync_correspondence_smoke(tmp_path, capsys) -> None:
    clip = _write_cli_sync_clip(tmp_path / 'cli_sync.mp4')

    assert main([
        'av',
        'sync-correspondence',
        str(clip),
        '--output',
        str(tmp_path / 'out_sync'),
        '--max-seconds',
        '3.5',
        '--sample-fps',
        '10',
        '--sample-rate',
        '22050',
        '--device',
        'cpu',
        '--no-overlay',
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload['tool_name'] == 'av.sync_correspondence'
    assert Path(payload['timeline_json']).exists()
    assert payload['overlay_path'] is None
