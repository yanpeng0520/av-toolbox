"""MediaPipe pose landmark tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts


class PoseTool(BaseTool):
    """Extract pose landmarks from sampled video frames with MediaPipe."""

    name = "video.pose"
    category = "video"
    description = "Extract human pose landmarks from sampled video frames with MediaPipe."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 5.0,
        max_seconds: float | None = None,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.pose requires input_path")
        cv2, mp = _imports()
        metadata = read_video_metadata(input_path)

        rows = []
        events = []
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        try:
            for frame_idx, timestamp, frame in iter_sampled_frames(
                input_path,
                sample_fps=sample_fps,
                max_seconds=max_seconds,
            ):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if not result.pose_landmarks:
                    continue
                frame_rows = []
                for landmark_idx, landmark in enumerate(result.pose_landmarks.landmark):
                    row = {
                        "timestamp": round(timestamp, 6),
                        "frame_idx": int(frame_idx),
                        "landmark_idx": landmark_idx,
                        "x": round(float(landmark.x), 6),
                        "y": round(float(landmark.y), 6),
                        "z": round(float(landmark.z), 6),
                        "visibility": round(float(landmark.visibility), 6),
                    }
                    rows.append(row)
                    frame_rows.append(row)
                events.append({
                    "start": round(timestamp, 6),
                    "end": round(timestamp, 6),
                    "label": "pose_tracking",
                    "landmark_count": len(frame_rows),
                    "data": {"frame_idx": int(frame_idx), "landmarks": frame_rows},
                })
        finally:
            pose.close()

        summary = {
            "pose_frame_count": len(events),
            "landmark_count": len(rows),
            "model": "mediapipe-pose",
        }
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "landmarks": rows,
        }
        config = {
            "tool_name": self.name,
            "sample_fps": sample_fps,
            "max_seconds": max_seconds,
            "model_complexity": model_complexity,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        _, result = write_standard_artifacts(
            tool_name=self.name,
            input_path=input_path,
            context=context,
            config=config,
            timeline_payload=timeline_payload,
            rows=rows,
            csv_fields=["timestamp", "frame_idx", "landmark_idx", "x", "y", "z", "visibility"],
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            log_lines=[
                f"tool={self.name}",
                f"input={input_path}",
                f"pose_frames={len(events)}",
                f"landmarks={len(rows)}",
            ],
        )
        return result


def _imports() -> tuple[Any, Any]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "video.pose requires OpenCV and MediaPipe. Install with: "
            "pip install -e '.[pose]'"
        ) from exc
    return cv2, mp
