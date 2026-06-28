"""Dense optical-flow motion analysis tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import resize_to_width, write_standard_artifacts


class OpticalFlowTool(BaseTool):
    """Measure dense optical-flow magnitude between sampled frames."""

    name = "video.optical_flow"
    category = "video"
    description = "Estimate dense optical-flow magnitude and pixel-motion events."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 5.0,
        max_seconds: float | None = None,
        downscale_width: int = 512,
        active_threshold_px: float = 1.5,
        event_threshold_px: float = 2.5,
        active_pct_threshold: float = 3.0,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.optical_flow requires input_path")
        cv2, np = _imports()
        metadata = read_video_metadata(input_path)

        rows = []
        events = []
        prev_gray = None
        sample_interval = 1.0 / max(sample_fps, 1e-9)
        for frame_idx, timestamp, frame in iter_sampled_frames(
            input_path,
            sample_fps=sample_fps,
            max_seconds=max_seconds,
        ):
            resized = resize_to_width(frame, downscale_width, cv2)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = gray
                continue

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            active = mag >= active_threshold_px
            active_pct = float(np.mean(active) * 100.0) if active.size else 0.0
            mean_mag = float(np.mean(mag))
            p95_mag = float(np.percentile(mag, 95)) if mag.size else 0.0
            max_mag = float(np.max(mag)) if mag.size else 0.0
            mean_angle = float(np.mean(ang[active])) if bool(np.any(active)) else 0.0
            is_motion = p95_mag >= event_threshold_px or active_pct >= active_pct_threshold
            row = {
                "timestamp": round(timestamp, 6),
                "frame_idx": int(frame_idx),
                "mean_magnitude": round(mean_mag, 6),
                "p95_magnitude": round(p95_mag, 6),
                "max_magnitude": round(max_mag, 6),
                "active_pct": round(active_pct, 4),
                "mean_angle_rad": round(mean_angle, 6),
                "is_motion": bool(is_motion),
            }
            rows.append(row)
            if is_motion:
                events.append({
                    "start": row["timestamp"],
                    "end": round(timestamp + sample_interval, 6),
                    "label": "pixel_motion",
                    "data": row,
                })
            prev_gray = gray

        summary = _summary(rows)
        summary["detector"] = "farneback-dense-flow"
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
            "active_threshold_px": active_threshold_px,
            "event_threshold_px": event_threshold_px,
            "active_pct_threshold": active_pct_threshold,
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
                "mean_magnitude",
                "p95_magnitude",
                "max_magnitude",
                "active_pct",
                "mean_angle_rad",
                "is_motion",
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
            "video.optical_flow requires OpenCV and NumPy. Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "motion_count": 0,
            "peak_magnitude": 0.0,
            "mean_magnitude": 0.0,
        }
    return {
        "sample_count": len(rows),
        "motion_count": sum(1 for row in rows if row["is_motion"]),
        "peak_magnitude": round(max(float(row["max_magnitude"]) for row in rows), 6),
        "mean_magnitude": round(
            sum(float(row["mean_magnitude"]) for row in rows) / len(rows),
            6,
        ),
        "mean_active_pct": round(
            sum(float(row["active_pct"]) for row in rows) / len(rows),
            4,
        ),
    }
