"""Lens obstruction detection tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts


class ObstructionTool(BaseTool):
    """Detect near-uniform frames that indicate a blocked lens."""

    name = "video.obstruction"
    category = "video"
    description = "Detect likely lens obstruction from low image variance."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 5.0,
        max_seconds: float | None = None,
        threshold: float = 10.0,
        min_luminance: float = 10.0,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.obstruction requires input_path")
        cv2, np = _imports()
        metadata = read_video_metadata(input_path)

        rows = []
        events = []
        sample_interval = 1.0 / max(sample_fps, 1e-9)
        for frame_idx, timestamp, frame in iter_sampled_frames(
            input_path,
            sample_fps=sample_fps,
            max_seconds=max_seconds,
        ):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_luminance = float(np.mean(gray))
            std_dev = float(np.std(gray))
            is_obstructed = std_dev < threshold and mean_luminance >= min_luminance
            row = {
                "timestamp": round(timestamp, 6),
                "frame_idx": int(frame_idx),
                "mean_luminance": round(mean_luminance, 4),
                "std_dev": round(std_dev, 4),
                "is_obstructed": bool(is_obstructed),
            }
            rows.append(row)
            if is_obstructed:
                events.append({
                    "start": row["timestamp"],
                    "end": round(timestamp + sample_interval, 6),
                    "label": "obstructed",
                    "data": row,
                })

        summary = {
            "sample_count": len(rows),
            "obstructed_count": sum(1 for row in rows if row["is_obstructed"]),
            "mean_std_dev": round(
                sum(float(row["std_dev"]) for row in rows) / len(rows),
                4,
            ) if rows else 0.0,
            "threshold": threshold,
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
            "threshold": threshold,
            "min_luminance": min_luminance,
        }
        _, result = write_standard_artifacts(
            tool_name=self.name,
            input_path=input_path,
            context=context,
            config=config,
            timeline_payload=timeline_payload,
            rows=rows,
            csv_fields=["timestamp", "frame_idx", "mean_luminance", "std_dev", "is_obstructed"],
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
            "video.obstruction requires OpenCV and NumPy. Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np
