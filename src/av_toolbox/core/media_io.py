"""Media inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "path": self.path,
            "duration": round(self.duration, 6),
            "fps": round(self.fps, 6),
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
        }


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Video helpers require OpenCV. Install with: pip install -e '.[video]'"
        ) from exc
    return cv2


def read_video_metadata(path: str | Path) -> VideoMetadata:
    """Read basic video metadata with OpenCV."""
    cv2 = _cv2()
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(
        path=str(video_path),
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
    )


def iter_sampled_frames(
    path: str | Path,
    *,
    sample_fps: float | None = None,
    max_seconds: float | None = None,
) -> Iterator[tuple[int, float, object]]:
    """Yield ``(frame_idx, timestamp, frame)`` sampled from a video."""
    cv2 = _cv2()
    metadata = read_video_metadata(path)
    if metadata.fps <= 0:
        raise RuntimeError(f"Cannot determine video FPS for {path}")

    requested_fps = sample_fps or metadata.fps
    if requested_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    effective_fps = min(requested_fps, metadata.fps)
    step = max(1, round(metadata.fps / effective_fps))
    max_frames = (
        min(metadata.frame_count, int(max_seconds * metadata.fps))
        if max_seconds is not None
        else metadata.frame_count
    )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        idx = 0
        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield idx, idx / metadata.fps, frame
            idx += 1
    finally:
        cap.release()

