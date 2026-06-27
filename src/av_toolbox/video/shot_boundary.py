"""Shot boundary analysis tool."""

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


class ShotBoundaryTool(BaseTool):
    """Detect likely hard cuts with sampled color-histogram differences."""

    name = "video.shot_boundary"
    category = "video"
    description = "Detect likely shot boundaries and scene segments."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 5.0,
        max_seconds: float | None = None,
        threshold: float = 0.45,
        min_scene_seconds: float = 0.5,
        downscale_width: int = 512,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.shot_boundary requires input_path")

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
            "min_scene_seconds": min_scene_seconds,
            "downscale_width": downscale_width,
        }
        artifacts.config_path.write_text(yaml.safe_dump(config, sort_keys=True))

        rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        prev_hist = None
        prev_gray = None
        last_boundary_at = 0.0

        for frame_idx, timestamp, frame in iter_sampled_frames(
            input_path,
            sample_fps=sample_fps,
            max_seconds=max_seconds,
        ):
            resized = _resize(frame, downscale_width, cv2)
            hist = _histogram(resized, cv2)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if prev_hist is None or prev_gray is None:
                prev_hist = hist
                prev_gray = gray
                continue

            hist_distance = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            mean_absdiff = float(np.mean(cv2.absdiff(prev_gray, gray)))
            enough_gap = (timestamp - last_boundary_at) >= min_scene_seconds
            is_boundary = hist_distance >= threshold and enough_gap

            row = {
                "timestamp": round(timestamp, 6),
                "frame_idx": int(frame_idx),
                "hist_distance": round(hist_distance, 6),
                "mean_absdiff": round(mean_absdiff, 4),
                "is_boundary": is_boundary,
            }
            rows.append(row)
            if is_boundary:
                event = {
                    "start": round(timestamp, 6),
                    "end": round(timestamp, 6),
                    "label": "shot_boundary",
                    "data": row,
                }
                events.append(event)
                last_boundary_at = timestamp

            prev_hist = hist
            prev_gray = gray

        analyzed_duration = metadata.duration
        if max_seconds is not None:
            analyzed_duration = min(metadata.duration, max_seconds)
        segments = _segments(events, analyzed_duration)
        summary = _summary(rows, segments)
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "segments": segments,
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
                f"boundaries={len(events)}",
                f"segments={len(segments)}",
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
                "segments": len(segments),
            },
        )


def _imports():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.shot_boundary requires OpenCV and NumPy. "
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


def _histogram(frame: Any, cv2: Any) -> Any:
    hist = cv2.calcHist(
        [frame],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256],
    )
    cv2.normalize(hist, hist)
    return hist


def _segments(events: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    boundaries = [float(event["start"]) for event in events]
    starts = [0.0] + boundaries
    ends = boundaries + [max(0.0, duration)]
    segments: list[dict[str, Any]] = []
    for idx, (start, end) in enumerate(zip(starts, ends)):
        if end < start:
            end = start
        segments.append({
            "index": idx,
            "start": round(start, 6),
            "end": round(end, 6),
            "duration": round(max(0.0, end - start), 6),
        })
    return segments


def _summary(rows: list[dict[str, Any]], segments: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "boundary_count": 0,
            "segment_count": len(segments),
            "peak_hist_distance": 0.0,
            "mean_hist_distance": 0.0,
        }
    return {
        "sample_count": len(rows),
        "boundary_count": sum(1 for row in rows if row["is_boundary"]),
        "segment_count": len(segments),
        "peak_hist_distance": round(max(float(row["hist_distance"]) for row in rows), 6),
        "mean_hist_distance": round(
            sum(float(row["hist_distance"]) for row in rows) / len(rows),
            6,
        ),
        "mean_absdiff": round(sum(float(row["mean_absdiff"]) for row in rows) / len(rows), 4),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "timestamp",
        "frame_idx",
        "hist_distance",
        "mean_absdiff",
        "is_boundary",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    title = "video.shot_boundary report"
    return "\n".join([
        "<!doctype html>",
        "<html><head>",
        f"<title>{title}</title>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>Input: {payload['input_path']}</p>",
        "<ul>",
        f"<li>Samples: {summary.get('sample_count', 0)}</li>",
        f"<li>Boundaries: {summary.get('boundary_count', 0)}</li>",
        f"<li>Segments: {summary.get('segment_count', 0)}</li>",
        f"<li>Peak histogram distance: {summary.get('peak_hist_distance', 0.0)}</li>",
        "</ul>",
        "</body></html>",
    ])
