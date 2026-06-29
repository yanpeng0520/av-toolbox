from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def demo_media_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")

    from av_toolbox.demo_media import generate_synthetic_hiphop

    output_dir = tmp_path_factory.mktemp("demo_media")
    try:
        payload = generate_synthetic_hiphop(
            output_dir=output_dir,
            duration=3.0,
            sample_rate=22050,
            stem="pytest_demo",
            fps=12.0,
            width=640,
            height=360,
        )
    except Exception as exc:
        pytest.skip(f"Could not generate test demo media: {exc}")
    return {
        "video": Path(payload["mp4_path"]),
        "audio": Path(payload["wav_path"]),
        "metadata": Path(payload["meta_path"]),
    }


@pytest.fixture(scope="session")
def demo_video_path(demo_media_paths: dict[str, Path]) -> Path:
    return demo_media_paths["video"]


@pytest.fixture(scope="session")
def demo_audio_path(demo_media_paths: dict[str, Path]) -> Path:
    return demo_media_paths["audio"]
