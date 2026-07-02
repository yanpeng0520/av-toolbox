from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import av_toolbox
from av_toolbox.video.optical_flow import _optical_flow_row_at_time


def _write_motion_video(path: Path, *, duration: float = 1.2, fps: float = 10.0) -> Path:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    if not writer.isOpened():
        pytest.skip(f"OpenCV cannot write {path}")
    try:
        for idx in range(int(duration * fps)):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[:] = (20, 22, 24)
            offset = int((idx % 6) * 8)
            cv2.rectangle(frame, (20 + offset, 35), (70 + offset, 85), (220, 80, 40), -1)
            cv2.circle(frame, (120 - offset // 2, 35 + offset), 12, (40, 180, 220), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_optical_flow_registered() -> None:
    assert "video.optical_flow" in [tool["name"] for tool in av_toolbox.list_tools()]


def test_optical_flow_creates_declared_artifacts_and_overlay(tmp_path: Path) -> None:
    video = _write_motion_video(tmp_path / "motion.mp4")

    result = av_toolbox.run_tool(
        "video.optical_flow",
        input_path=video,
        output_dir=tmp_path / "out",
        sample_fps=5,
        max_seconds=1.0,
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
    assert payload["tool_name"] == "video.optical_flow"
    assert payload["summary"]["sample_count"] > 0


def test_optical_flow_can_disable_overlay(tmp_path: Path) -> None:
    video = _write_motion_video(tmp_path / "motion.mp4")

    result = av_toolbox.run_tool(
        "video.optical_flow",
        input_path=video,
        output_dir=tmp_path / "out",
        sample_fps=5,
        max_seconds=1.0,
        device="cpu",
        export_overlay=False,
    )

    assert result.overlay_path is None


def test_optical_flow_overlay_value_uses_nearest_sample() -> None:
    times = np.asarray([0.0, 0.5, 1.0], dtype=float)
    rows = [
        {"timestamp": 0.0, "active_pct": 1.0, "p95_magnitude": 0.1},
        {"timestamp": 0.5, "active_pct": 35.0, "p95_magnitude": 2.1},
        {"timestamp": 1.0, "active_pct": 4.0, "p95_magnitude": 0.4},
    ]

    assert _optical_flow_row_at_time(rows, times, 0.46) is rows[1]
    assert _optical_flow_row_at_time(rows, times, 0.1) is rows[0]
    assert _optical_flow_row_at_time([], np.asarray([], dtype=float), 0.2)["active_pct"] == 0.0
