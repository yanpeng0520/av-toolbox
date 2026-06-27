"""Blur and exposure analysis tool."""

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


class BlurExposureTool(BaseTool):
    """Classical video quality tool for blur and luminance/exposure flags."""

    name = "video.blur_exposure"
    category = "video"
    description = "Detect blur, dark frames, and overexposed frames."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 5.0,
        max_seconds: float | None = None,
        blur_threshold: float = 10.0,
        dark_threshold: float = 50.0,
        super_dark_threshold: float = 10.0,
        overexposed_threshold: float = 230.0,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.blur_exposure requires input_path")

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
            "blur_threshold": blur_threshold,
            "dark_threshold": dark_threshold,
            "super_dark_threshold": super_dark_threshold,
            "overexposed_threshold": overexposed_threshold,
        }
        artifacts.config_path.write_text(yaml.safe_dump(config, sort_keys=True))

        rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for frame_idx, timestamp, frame in iter_sampled_frames(
            input_path,
            sample_fps=sample_fps,
            max_seconds=max_seconds,
        ):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_lum = float(np.mean(gray))
            std_lum = float(np.std(gray))
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            is_blurry = lap_var < blur_threshold
            is_super_dark = mean_lum < super_dark_threshold
            is_dark = mean_lum < dark_threshold
            is_overexposed = mean_lum >= overexposed_threshold

            row = {
                "timestamp": round(timestamp, 6),
                "frame_idx": int(frame_idx),
                "mean_luminance": round(mean_lum, 4),
                "std_luminance": round(std_lum, 4),
                "laplacian_variance": round(lap_var, 4),
                "is_blurry": is_blurry,
                "is_dark": is_dark,
                "is_super_dark": is_super_dark,
                "is_overexposed": is_overexposed,
            }
            rows.append(row)
            if is_blurry or is_dark or is_overexposed:
                labels = []
                if is_blurry:
                    labels.append("blur")
                if is_dark:
                    labels.append("dark")
                if is_overexposed:
                    labels.append("overexposed")
                events.append({
                    "start": round(timestamp, 6),
                    "end": round(timestamp + (1.0 / max(sample_fps, 1e-9)), 6),
                    "label": "+".join(labels),
                    "data": row,
                })

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
            "video.blur_exposure requires OpenCV and NumPy. "
            "Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "blurry_count": 0,
            "dark_count": 0,
            "overexposed_count": 0,
        }
    return {
        "sample_count": len(rows),
        "blurry_count": sum(1 for row in rows if row["is_blurry"]),
        "dark_count": sum(1 for row in rows if row["is_dark"]),
        "super_dark_count": sum(1 for row in rows if row["is_super_dark"]),
        "overexposed_count": sum(1 for row in rows if row["is_overexposed"]),
        "mean_luminance": round(sum(float(row["mean_luminance"]) for row in rows) / len(rows), 4),
        "mean_laplacian_variance": round(
            sum(float(row["laplacian_variance"]) for row in rows) / len(rows),
            4,
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "timestamp",
        "frame_idx",
        "mean_luminance",
        "std_luminance",
        "laplacian_variance",
        "is_blurry",
        "is_dark",
        "is_super_dark",
        "is_overexposed",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    title = "video.blur_exposure report"
    return "\n".join([
        "<!doctype html>",
        "<html><head>",
        f"<title>{title}</title>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>Input: {payload['input_path']}</p>",
        "<ul>",
        f"<li>Samples: {summary.get('sample_count', 0)}</li>",
        f"<li>Blurry: {summary.get('blurry_count', 0)}</li>",
        f"<li>Dark: {summary.get('dark_count', 0)}</li>",
        f"<li>Overexposed: {summary.get('overexposed_count', 0)}</li>",
        "</ul>",
        "</body></html>",
    ])

