from __future__ import annotations

from pathlib import Path

from av_toolbox.core.media_io import read_video_metadata


def test_demo_video_metadata(demo_video_path: Path) -> None:
    metadata = read_video_metadata(demo_video_path)

    assert metadata.duration > 0
    assert metadata.fps > 0
    assert metadata.width > 0
    assert metadata.height > 0
    assert metadata.frame_count > 0
