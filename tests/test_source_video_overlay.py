from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from av_toolbox.video.source_overlay import render_source_video_overlay


def _write_video(path: Path, *, duration: float = 1.0, fps: float = 8.0) -> Path:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    if not writer.isOpened():
        pytest.skip(f"OpenCV cannot write {path}")
    try:
        for idx in range(int(duration * fps)):
            frame = np.full((120, 160, 3), (24, 28, 32), dtype=np.uint8)
            offset = idx * 8
            cv2.rectangle(frame, (18 + offset, 34), (58 + offset, 76), (220, 90, 40), -1)
            cv2.circle(frame, (118 - offset // 2, 36 + offset // 2), 10, (40, 180, 220), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _write_audio(path: Path, *, duration: float = 1.0, sr: int = 22050) -> Path:
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    y = 0.45 * np.sin(2 * np.pi * 330.0 * t)
    pcm = np.clip(y * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return path


def _write_video_with_audio(path: Path, *, duration: float = 1.25, fps: float = 8.0) -> Path:
    video_only = _write_video(path.with_name(path.stem + "_video_only.mp4"), duration=duration, fps=fps)
    audio = _write_audio(path.with_suffix(".wav"), duration=duration)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_only),
            "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k",
            str(path),
        ],
        check=True,
    )
    return path


def _ffprobe_json(path: Path, *args: str) -> dict:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", *args, "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _duration(path: Path) -> float:
    payload = _ffprobe_json(path, "-show_entries", "format=duration")
    return float(payload["format"]["duration"])


def _has_audio_stream(path: Path) -> bool:
    payload = _ffprobe_json(path, "-select_streams", "a", "-show_entries", "stream=codec_type")
    return bool(payload.get("streams"))


def _assert_decodable(path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        pytest.fail(f"Cannot open overlay: {path}")
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
    finally:
        cap.release()
    assert frames > 0
    assert width >= 320
    assert height >= 240
    assert ok
    assert frame is not None


def _video_size(path: Path) -> tuple[int, int]:
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        pytest.fail(f"Cannot open overlay: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return width, height


@pytest.mark.parametrize(
    ("mode", "rows", "events", "kwargs"),
    [
        (
            "metric",
            [
                {"timestamp": 0.125, "active_pct": 20.0, "is_motion": True},
                {"timestamp": 0.375, "active_pct": 55.0, "is_motion": True},
            ],
            [{"start": 0.125, "end": 0.5, "label": "motion"}],
            {
                "metric_key": "active_pct",
                "metric_label": "active",
                "flow_mask": True,
                "flow_threshold_px": 0.2,
                "flow_scale_px": 2.0,
            },
        ),
        (
            "cuts",
            [{"timestamp": 0.5, "score": 0.9}],
            [{"start": 0.5, "end": 0.5, "label": "cut"}],
            {"metric_key": "score", "metric_label": "score"},
        ),
        (
            "boxes",
            [{"timestamp": 0.25, "label": "box", "confidence": 0.8, "x1": 20, "y1": 20, "x2": 90, "y2": 80}],
            [{"start": 0.25, "end": 0.25, "label": "box"}],
            {},
        ),
        (
            "segments",
            [{"timestamp": 0.25, "label": "mask", "confidence": 0.8, "x1": 20, "y1": 20, "x2": 90, "y2": 80}],
            [{"start": 0.25, "end": 0.25, "label": "mask"}],
            {},
        ),
        (
            "pose",
            [
                {"timestamp": 0.25, "landmark_idx": 11, "x": 0.35, "y": 0.35, "visibility": 0.9},
                {"timestamp": 0.25, "landmark_idx": 12, "x": 0.55, "y": 0.35, "visibility": 0.9},
                {"timestamp": 0.25, "landmark_idx": 13, "x": 0.30, "y": 0.55, "visibility": 0.9},
                {"timestamp": 0.25, "landmark_idx": 14, "x": 0.60, "y": 0.55, "visibility": 0.9},
            ],
            [{"start": 0.25, "end": 0.25, "label": "pose_tracking"}],
            {},
        ),
        (
            "labels",
            [{"timestamp": 0.25, "label": "medium shot", "confidence": 0.72}],
            [{"start": 0.25, "end": 0.25, "label": "medium shot"}],
            {"metric_key": "confidence", "metric_label": "confidence"},
        ),
    ],
)
def test_source_video_overlay_modes_decode(tmp_path: Path, mode: str, rows: list[dict], events: list[dict], kwargs: dict) -> None:
    source = _write_video(tmp_path / f"source_{mode}.mp4")
    output = tmp_path / f"overlay_{mode}.mp4"

    result = render_source_video_overlay(
        input_path=source,
        output_path=output,
        rows=rows,
        events=events,
        duration=1.0,
        workspace=tmp_path / "workspace",
        tool_label=mode,
        fps=6.0,
        width=320,
        mode=mode,
        **kwargs,
    )

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0
    _assert_decodable(output)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for audio mux validation",
)
def test_source_video_overlay_preserves_source_audio_and_duration(tmp_path: Path) -> None:
    source = _write_video_with_audio(tmp_path / "source_with_audio.mp4")
    output = tmp_path / "overlay_with_audio.mp4"

    render_source_video_overlay(
        input_path=source,
        output_path=output,
        rows=[{"timestamp": 0.25, "active_pct": 35.0}],
        events=[{"start": 0.25, "end": 0.5, "label": "motion"}],
        duration=1.25,
        workspace=tmp_path / "workspace_audio",
        tool_label="video.motion",
        fps=6.0,
        width=320,
        mode="metric",
        metric_key="active_pct",
        metric_label="active",
    )

    assert _has_audio_stream(output)
    assert abs(_duration(output) - _duration(source)) <= 0.25
    _assert_decodable(output)

def test_source_video_overlay_compact_and_transparent_timeline_layouts(tmp_path: Path) -> None:
    source = _write_video(tmp_path / "layout_source.mp4")
    compact = tmp_path / "compact_boxes.mp4"
    transparent = tmp_path / "transparent_labels.mp4"
    panel = tmp_path / "panel_boxes.mp4"

    common = {
        "input_path": source,
        "rows": [{"timestamp": 0.25, "label": "cat", "confidence": 0.8, "x1": 20, "y1": 20, "x2": 90, "y2": 80}],
        "events": [{"start": 0.25, "end": 0.25, "label": "cat"}],
        "duration": 1.0,
        "workspace": tmp_path / "workspace_layout",
        "tool_label": "layout",
        "fps": 6.0,
        "width": 320,
    }

    render_source_video_overlay(output_path=panel, mode="boxes", **common)
    render_source_video_overlay(output_path=compact, mode="boxes", timeline_style="none", **common)
    render_source_video_overlay(
        output_path=transparent,
        mode="labels",
        metric_key="confidence",
        metric_label="confidence",
        timeline_style="transparent",
        **common,
    )

    _assert_decodable(compact)
    _assert_decodable(transparent)
    assert _video_size(compact)[1] < _video_size(panel)[1]
    assert _video_size(transparent) == _video_size(compact)
