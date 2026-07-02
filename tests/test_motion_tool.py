from __future__ import annotations

import csv
import json
from pathlib import Path

import av_toolbox
from av_toolbox.video.motion import _motion_row_at_time


def test_motion_registered() -> None:
    assert "video.motion" in [tool["name"] for tool in av_toolbox.list_tools()]


def test_motion_creates_declared_artifacts(tmp_path, demo_video_path: Path) -> None:
    result = av_toolbox.run_tool(
        "video.motion",
        input_path=demo_video_path,
        output_dir=tmp_path,
        sample_fps=4.0,
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
    assert payload["tool_name"] == "video.motion"
    assert payload["media"]["duration"] > 0
    assert payload["summary"]["sample_count"] > 0

    with Path(result.csv_path).open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "active_pct" in rows[0]


def test_motion_can_disable_overlay(tmp_path, demo_video_path: Path) -> None:
    result = av_toolbox.run_tool(
        "video.motion",
        input_path=demo_video_path,
        output_dir=tmp_path,
        sample_fps=4.0,
        max_seconds=1.0,
        device="cpu",
        export_overlay=False,
    )

    assert result.overlay_path is None


def test_motion_overlay_row_uses_nearest_sample() -> None:
    np = __import__("numpy")

    rows = [
        {"timestamp": 0.2, "active_pct": 3.0, "is_motion": False},
        {"timestamp": 0.5, "active_pct": 42.5, "is_motion": True},
        {"timestamp": 0.9, "active_pct": 9.0, "is_motion": True},
    ]
    times = np.asarray([row["timestamp"] for row in rows], dtype=float)

    assert _motion_row_at_time(rows, times, 0.46) is rows[1]
    assert _motion_row_at_time(rows, times, 0.1) is rows[0]
    assert _motion_row_at_time([], np.asarray([], dtype=float), 0.1)["active_pct"] == 0.0
