from __future__ import annotations

from pathlib import Path

from av_toolbox.core.media_io import read_video_metadata


def test_demo_video_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    demo = root / "data_segments" / "Clever_Cat_Outsmarts_Warrior_square.mp4"

    metadata = read_video_metadata(demo)

    assert metadata.duration > 0
    assert metadata.fps > 0
    assert metadata.width > 0
    assert metadata.height > 0
    assert metadata.frame_count > 0
