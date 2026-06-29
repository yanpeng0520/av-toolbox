from __future__ import annotations

from pathlib import Path


def test_generated_demo_video_exists(demo_video_path: Path) -> None:
    assert demo_video_path.exists()
    assert demo_video_path.stat().st_size > 0
