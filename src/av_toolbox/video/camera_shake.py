"""Camera-shake detection via tracked background translations."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import resize_to_width, write_standard_artifacts


class CameraShakeTool(BaseTool):
    """Detect high-frequency frame jitter with sparse optical flow."""

    name = "video.camera_shake"
    category = "video"
    description = "Detect camera shake from detrended sparse optical-flow translations."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 25.0,
        max_seconds: float | None = None,
        downscale_width: int = 512,
        window: int = 25,
        threshold: float = 0.5,
        min_features: int = 20,
        max_features: int = 300,
        redetect_interval: int = 30,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.camera_shake requires input_path")
        cv2, np = _imports()
        metadata = read_video_metadata(input_path)

        rows = []
        events = []
        prev_gray = None
        prev_points = None
        translations: deque[tuple[float, float]] = deque(maxlen=max(3, window))
        sample_interval = 1.0 / max(sample_fps, 1e-9)

        for sample_idx, (frame_idx, timestamp, frame) in enumerate(
            iter_sampled_frames(input_path, sample_fps=sample_fps, max_seconds=max_seconds)
        ):
            resized = resize_to_width(frame, downscale_width, cv2)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = gray
                prev_points = _detect_features(gray, cv2, max_features)
                continue

            dx = dy = 0.0
            tracked_features = 0
            curr_points = None
            if prev_points is not None and len(prev_points) > 0:
                curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)
                if curr_points is not None and status is not None:
                    good_prev = prev_points[status.reshape(-1) == 1]
                    good_curr = curr_points[status.reshape(-1) == 1]
                    tracked_features = int(len(good_curr))
                    if tracked_features >= 3:
                        matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
                        if matrix is not None:
                            dx = float(matrix[0, 2])
                            dy = float(matrix[1, 2])
                        else:
                            delta = good_curr.reshape(-1, 2) - good_prev.reshape(-1, 2)
                            dx = float(np.median(delta[:, 0]))
                            dy = float(np.median(delta[:, 1]))
            translations.append((dx, dy))
            shake_score = _shake_score(translations, np)
            is_shaking = shake_score >= threshold and len(translations) >= min(5, window)
            row = {
                "timestamp": round(timestamp, 6),
                "frame_idx": int(frame_idx),
                "translation_x": round(dx, 6),
                "translation_y": round(dy, 6),
                "shake_score": round(shake_score, 6),
                "tracked_features": tracked_features,
                "is_shaking": bool(is_shaking),
            }
            rows.append(row)
            if is_shaking:
                events.append({
                    "start": row["timestamp"],
                    "end": round(timestamp + sample_interval, 6),
                    "label": "camera_shake",
                    "data": row,
                })

            should_redetect = (
                tracked_features < min_features
                or redetect_interval > 0 and sample_idx % redetect_interval == 0
            )
            prev_gray = gray
            prev_points = _detect_features(gray, cv2, max_features) if should_redetect else curr_points

        summary = {
            "sample_count": len(rows),
            "shaking_count": sum(1 for row in rows if row["is_shaking"]),
            "peak_shake_score": round(max((float(row["shake_score"]) for row in rows), default=0.0), 6),
            "mean_shake_score": round(
                sum(float(row["shake_score"]) for row in rows) / len(rows),
                6,
            ) if rows else 0.0,
            "detector": "sparse-lk-detrended-translation",
        }
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "samples": rows,
        }
        config = {
            "tool_name": self.name,
            "sample_fps": sample_fps,
            "max_seconds": max_seconds,
            "downscale_width": downscale_width,
            "window": window,
            "threshold": threshold,
            "min_features": min_features,
            "max_features": max_features,
            "redetect_interval": redetect_interval,
        }
        _, result = write_standard_artifacts(
            tool_name=self.name,
            input_path=input_path,
            context=context,
            config=config,
            timeline_payload=timeline_payload,
            rows=rows,
            csv_fields=[
                "timestamp",
                "frame_idx",
                "translation_x",
                "translation_y",
                "shake_score",
                "tracked_features",
                "is_shaking",
            ],
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            log_lines=[
                f"tool={self.name}",
                f"input={input_path}",
                f"frames_analyzed={len(rows)}",
                f"events={len(events)}",
            ],
        )
        return result


def _imports() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.camera_shake requires OpenCV and NumPy. Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _detect_features(gray: Any, cv2: Any, max_features: int) -> Any:
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )


def _shake_score(translations: deque[tuple[float, float]], np: Any) -> float:
    if len(translations) < 3:
        return 0.0
    values = np.asarray(translations, dtype=np.float32)
    x = np.arange(len(values), dtype=np.float32)
    residual_energy = 0.0
    for axis in (0, 1):
        coeff = np.polyfit(x, values[:, axis], deg=1)
        trend = coeff[0] * x + coeff[1]
        residual = values[:, axis] - trend
        residual_energy += float(np.mean(residual * residual))
    return float(np.sqrt(residual_energy))
