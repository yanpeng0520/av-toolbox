from __future__ import annotations

import csv
import json
from pathlib import Path

import av_toolbox


def test_shot_boundary_registered() -> None:
    assert "video.shot_boundary" in [tool["name"] for tool in av_toolbox.list_tools()]


def test_shot_boundary_creates_declared_artifacts(tmp_path, demo_video_path: Path) -> None:
    result = av_toolbox.run_tool(
        "video.shot_boundary",
        input_path=demo_video_path,
        output_dir=tmp_path,
        sample_fps=4.0,
        max_seconds=1.0,
        device="cpu",
    )

    assert result.timeline_json is not None
    assert result.csv_path is not None
    assert result.config_path is not None
    assert result.log_path is not None
    assert Path(result.timeline_json).exists()
    assert Path(result.csv_path).exists()
    assert Path(result.config_path).exists()
    assert Path(result.log_path).exists()

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "video.shot_boundary"
    assert payload["media"]["duration"] > 0
    assert payload["summary"]["sample_count"] > 0
    assert payload["segments"]

    with Path(result.csv_path).open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "hist_distance" in rows[0]
