"""YOLO instance segmentation tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts


class SegmentationTool(BaseTool):
    """Segment objects in sampled video frames using Ultralytics YOLO."""

    name = "video.segmentation"
    category = "video"
    description = "Segment object instances in sampled video frames with YOLO."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 2.0,
        max_seconds: float | None = None,
        model_name: str | None = "yolov8n-seg.pt",
        confidence: float = 0.25,
        image_size: int | None = None,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.segmentation requires input_path")
        YOLO = _import_yolo()
        np = _import_numpy()
        metadata = read_video_metadata(input_path)
        model = YOLO(model_name or "yolov8n-seg.pt")

        rows = []
        events = []
        device_arg = context.hardware.resolved_device()
        if not device_arg.startswith("cuda"):
            device_arg = "cpu"
        for frame_idx, timestamp, frame in iter_sampled_frames(
            input_path,
            sample_fps=sample_fps,
            max_seconds=max_seconds,
        ):
            kwargs: dict[str, Any] = {
                "conf": confidence,
                "device": device_arg,
                "verbose": False,
            }
            if image_size:
                kwargs["imgsz"] = image_size
            result = model.predict(frame, **kwargs)[0]
            names = getattr(result, "names", getattr(model, "names", {}))
            if getattr(result, "boxes", None) is None:
                continue
            boxes = result.boxes
            xyxy = _to_numpy(boxes.xyxy)
            confs = _to_numpy(boxes.conf)
            classes = _to_numpy(boxes.cls).astype(int)
            mask_areas = [0.0] * len(classes)
            if getattr(result, "masks", None) is not None:
                masks = _to_numpy(result.masks.data)
                mask_areas = [float(np.mean(mask > 0.5)) for mask in masks]

            for det_idx, (box, score, cls_id, mask_area) in enumerate(zip(xyxy, confs, classes, mask_areas)):
                label = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else str(int(cls_id))
                row = {
                    "timestamp": round(timestamp, 6),
                    "frame_idx": int(frame_idx),
                    "segment_idx": det_idx,
                    "label": label,
                    "class_id": int(cls_id),
                    "confidence": round(float(score), 6),
                    "mask_area_pct": round(mask_area * 100.0, 6),
                    "x1": round(float(box[0]), 3),
                    "y1": round(float(box[1]), 3),
                    "x2": round(float(box[2]), 3),
                    "y2": round(float(box[3]), 3),
                }
                rows.append(row)
                events.append({
                    "start": row["timestamp"],
                    "end": row["timestamp"],
                    "label": label,
                    "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
                    "data": row,
                })

        summary = _label_summary(rows)
        summary.update({
            "segment_count": len(rows),
            "model": model_name or "yolov8n-seg.pt",
        })
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "segments": rows,
        }
        config = {
            "tool_name": self.name,
            "sample_fps": sample_fps,
            "max_seconds": max_seconds,
            "model_name": model_name or "yolov8n-seg.pt",
            "confidence": confidence,
            "image_size": image_size,
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
                "segment_idx",
                "label",
                "class_id",
                "confidence",
                "mask_area_pct",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            log_lines=[
                f"tool={self.name}",
                f"input={input_path}",
                f"segments={len(rows)}",
            ],
        )
        return result


def _import_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "video.segmentation requires ultralytics. Install with: "
            "pip install -e '.[vision-models]'"
        ) from exc
    return YOLO


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.segmentation requires NumPy. Install with: pip install -e '.[video]'"
        ) from exc
    return np


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _label_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return {"label_counts": counts}
