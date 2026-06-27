from __future__ import annotations

from pathlib import Path


def test_default_demo_video_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    demo = root / "data_segments" / "Clever_Cat_Outsmarts_Warrior_square.mp4"

    assert demo.exists()
    assert demo.stat().st_size > 0

