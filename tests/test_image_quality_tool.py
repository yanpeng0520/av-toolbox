from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import av_toolbox
from av_toolbox.video.image_quality import _image_quality_row_at_time


def _write_uniform_video(path: Path, *, duration: float = 0.8, fps: float = 5.0) -> Path:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (80, 60))
    if not writer.isOpened():
        pytest.skip(f"OpenCV cannot write {path}")
    try:
        for _ in range(int(duration * fps)):
            writer.write(np.full((60, 80, 3), 90, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_image_quality_registered() -> None:
    assert "video.image_quality" in [tool["name"] for tool in av_toolbox.list_tools()]


def test_image_quality_creates_declared_artifacts(tmp_path, demo_video_path: Path) -> None:
    result = av_toolbox.run_tool(
        "video.image_quality",
        input_path=demo_video_path,
        output_dir=tmp_path,
        sample_fps=2.0,
        max_seconds=0.5,
        device="cpu",
    )

    assert result.timeline_json is not None
    assert result.csv_path is not None
    assert result.overlay_path is not None
    assert result.config_path is not None
    assert result.log_path is not None
    assert Path(result.timeline_json).exists()
    assert Path(result.csv_path).exists()
    assert Path(result.overlay_path).exists()
    assert Path(result.overlay_path).stat().st_size > 0
    assert Path(result.config_path).exists()
    assert Path(result.log_path).exists()

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "video.image_quality"
    assert payload["media"]["duration"] > 0
    assert payload["summary"]["sample_count"] > 0

    with Path(result.csv_path).open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "laplacian_variance" in rows[0]
    assert "is_obstructed" in rows[0]


def test_image_quality_can_disable_overlay(tmp_path, demo_video_path: Path) -> None:
    result = av_toolbox.run_tool(
        "video.image_quality",
        input_path=demo_video_path,
        output_dir=tmp_path,
        sample_fps=2.0,
        max_seconds=0.5,
        device="cpu",
        export_overlay=False,
    )

    assert result.overlay_path is None


def test_image_quality_detects_obstruction(tmp_path: Path) -> None:
    video = _write_uniform_video(tmp_path / "uniform.mp4")

    result = av_toolbox.run_tool(
        "video.image_quality",
        input_path=video,
        output_dir=tmp_path / "out",
        sample_fps=5,
        max_seconds=0.5,
        device="cpu",
    )

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "video.image_quality"
    assert payload["summary"]["obstructed_count"] > 0


def test_image_quality_row_uses_nearest_sample() -> None:
    rows = [
        {"timestamp": 0.2, "laplacian_variance": 12.0, "mean_luminance": 44.0, "std_luminance": 30.0},
        {"timestamp": 0.5, "laplacian_variance": 80.0, "mean_luminance": 120.0, "std_luminance": 40.0},
        {"timestamp": 0.9, "laplacian_variance": 21.0, "mean_luminance": 240.0, "std_luminance": 5.0},
    ]
    times = np.asarray([row["timestamp"] for row in rows], dtype=float)

    assert _image_quality_row_at_time(rows, times, 0.48) is rows[1]
    assert _image_quality_row_at_time(rows, times, 0.1) is rows[0]
    assert _image_quality_row_at_time([], np.asarray([], dtype=float), 0.1)["mean_luminance"] == 0.0
