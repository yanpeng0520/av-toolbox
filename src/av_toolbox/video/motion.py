"""Frame-to-frame motion analysis tool."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.outputs import make_artifact_paths
from av_toolbox.core.result import AVResult


class MotionTool(BaseTool):
    """Classical motion intensity tool based on sampled frame differences."""

    name = "video.motion"
    category = "video"
    description = "Estimate frame-to-frame motion intensity."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 5.0,
        max_seconds: float | None = None,
        threshold: float = 15.0,
        active_pct_threshold: float = 5.0,
        downscale_width: int = 512,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.motion requires input_path")

        cv2, np = _imports()
        metadata = read_video_metadata(input_path)
        artifacts = make_artifact_paths(
            input_path=input_path,
            output_dir=context.output_dir,
            tool_name=self.name,
        )

        config = {
            "tool_name": self.name,
            "sample_fps": sample_fps,
            "max_seconds": max_seconds,
            "threshold": threshold,
            "active_pct_threshold": active_pct_threshold,
            "downscale_width": downscale_width,
        }
        artifacts.config_path.write_text(yaml.safe_dump(config, sort_keys=True))

        rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        prev_gray = None
        sample_interval = 1.0 / max(sample_fps, 1e-9)

        for frame_idx, timestamp, frame in iter_sampled_frames(
            input_path,
            sample_fps=sample_fps,
            max_seconds=max_seconds,
        ):
            resized = _resize(frame, downscale_width, cv2)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = gray
                continue

            diff = cv2.absdiff(prev_gray, gray)
            _, active = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            active_pixels = int(cv2.countNonZero(active))
            total_pixels = int(active.size)
            active_pct = (active_pixels / total_pixels * 100.0) if total_pixels else 0.0
            mean_diff = float(np.mean(diff))
            max_diff = float(np.max(diff))
            is_motion = active_pct >= active_pct_threshold

            row = {
                "timestamp": round(timestamp, 6),
                "frame_idx": int(frame_idx),
                "mean_diff": round(mean_diff, 4),
                "max_diff": round(max_diff, 4),
                "active_pixels": active_pixels,
                "total_pixels": total_pixels,
                "active_pct": round(active_pct, 4),
                "is_motion": is_motion,
            }
            rows.append(row)
            if is_motion:
                events.append({
                    "start": round(timestamp, 6),
                    "end": round(timestamp + sample_interval, 6),
                    "label": "motion",
                    "data": row,
                })
            prev_gray = gray

        summary = _summary(rows)
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "samples": rows,
        }

        if export_json:
            artifacts.timeline_json.write_text(json.dumps(timeline_payload, indent=2))
        if export_csv:
            _write_csv(artifacts.csv_path, rows)
        if export_report:
            artifacts.report_html.write_text(_html_report(timeline_payload))

        artifacts.log_path.write_text(
            "\n".join([
                f"tool={self.name}",
                f"input={input_path}",
                f"frames_analyzed={len(rows)}",
                f"events={len(events)}",
            ])
            + "\n"
        )

        return AVResult(
            tool_name=self.name,
            input_path=input_path,
            output_dir=context.output_dir,
            timeline_json=artifacts.timeline_json if export_json else None,
            csv_path=artifacts.csv_path if export_csv else None,
            report_html=artifacts.report_html if export_report else None,
            config_path=artifacts.config_path,
            log_path=artifacts.log_path,
            metadata={
                "media": metadata.to_dict(),
                "summary": summary,
                "frames_analyzed": len(rows),
                "events": len(events),
            },
        )


def _imports():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.motion requires OpenCV and NumPy. "
            "Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _resize(frame: Any, downscale_width: int, cv2: Any) -> Any:
    if downscale_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= downscale_width:
        return frame
    scale = downscale_width / float(width)
    size = (downscale_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "motion_count": 0,
            "peak_active_pct": 0.0,
            "mean_active_pct": 0.0,
            "mean_diff": 0.0,
        }
    return {
        "sample_count": len(rows),
        "motion_count": sum(1 for row in rows if row["is_motion"]),
        "peak_active_pct": round(max(float(row["active_pct"]) for row in rows), 4),
        "mean_active_pct": round(
            sum(float(row["active_pct"]) for row in rows) / len(rows),
            4,
        ),
        "mean_diff": round(sum(float(row["mean_diff"]) for row in rows) / len(rows), 4),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "timestamp",
        "frame_idx",
        "mean_diff",
        "max_diff",
        "active_pixels",
        "total_pixels",
        "active_pct",
        "is_motion",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    title = "video.motion report"
    return "\n".join([
        "<!doctype html>",
        "<html><head>",
        f"<title>{title}</title>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>Input: {payload['input_path']}</p>",
        "<ul>",
        f"<li>Samples: {summary.get('sample_count', 0)}</li>",
        f"<li>Motion samples: {summary.get('motion_count', 0)}</li>",
        f"<li>Peak active area: {summary.get('peak_active_pct', 0.0)}%</li>",
        f"<li>Mean active area: {summary.get('mean_active_pct', 0.0)}%</li>",
        "</ul>",
        "</body></html>",
    ])
