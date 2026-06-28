"""Foreground-biased dense optical-flow motion analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import resize_to_width, write_standard_artifacts


class ForegroundMotionTool(BaseTool):
    """Measure optical flow inside foreground masks when available."""

    name = "video.foreground_motion"
    category = "video"
    description = "Estimate foreground motion with dense optical flow and optional YOLO masks."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 5.0,
        max_seconds: float | None = None,
        downscale_width: int = 512,
        mask_mode: str = "none",
        model_name: str | None = "yolov8n-seg.pt",
        confidence: float = 0.25,
        active_threshold_px: float = 1.5,
        event_threshold_px: float = 2.5,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.foreground_motion requires input_path")
        cv2, np = _imports()
        metadata = read_video_metadata(input_path)
        masker = _Masker(
            mask_mode=mask_mode,
            model_name=model_name,
            confidence=confidence,
            device=context.hardware.resolved_device(),
        )

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
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask = masker.mask(resized, cv2, np)
            masked_mag = mag * mask
            foreground_pixels = int(np.count_nonzero(mask))
            active_pixels = int(np.count_nonzero(masked_mag >= active_threshold_px))
            active_pct = (active_pixels / foreground_pixels * 100.0) if foreground_pixels else 0.0
            foreground_mean = float(np.sum(masked_mag) / foreground_pixels) if foreground_pixels else 0.0
            p95 = float(np.percentile(masked_mag[mask > 0], 95)) if foreground_pixels else 0.0
            max_mag = float(np.max(masked_mag)) if foreground_pixels else 0.0
            is_motion = p95 >= event_threshold_px
            row = {
                "timestamp": round(timestamp, 6),
                "frame_idx": int(frame_idx),
                "mask_mode": masker.effective_mode,
                "foreground_pixels": foreground_pixels,
                "mean_magnitude": round(foreground_mean, 6),
                "p95_magnitude": round(p95, 6),
                "max_magnitude": round(max_mag, 6),
                "active_pct": round(active_pct, 4),
                "is_motion": bool(is_motion),
            }
            rows.append(row)
            if is_motion:
                events.append({
                    "start": row["timestamp"],
                    "end": round(timestamp + sample_interval, 6),
                    "label": "foreground_motion",
                    "data": row,
                })
            prev_gray = gray

        summary = {
            "sample_count": len(rows),
            "motion_count": sum(1 for row in rows if row["is_motion"]),
            "peak_magnitude": round(max((float(row["max_magnitude"]) for row in rows), default=0.0), 6),
            "mean_active_pct": round(
                sum(float(row["active_pct"]) for row in rows) / len(rows),
                4,
            ) if rows else 0.0,
            "mask_mode": masker.effective_mode,
            "detector": "farneback-foreground-flow",
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
            "mask_mode": mask_mode,
            "model_name": model_name,
            "confidence": confidence,
            "active_threshold_px": active_threshold_px,
            "event_threshold_px": event_threshold_px,
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
                "mask_mode",
                "foreground_pixels",
                "mean_magnitude",
                "p95_magnitude",
                "max_magnitude",
                "active_pct",
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
                f"mask_mode={masker.effective_mode}",
            ],
        )
        return result


class _Masker:
    def __init__(self, *, mask_mode: str, model_name: str | None, confidence: float, device: str) -> None:
        self.requested_mode = mask_mode
        self.effective_mode = "none"
        self.model = None
        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        if mask_mode in ("yolo", "yolo_seg"):
            self.model = _load_yolo(model_name or "yolov8n-seg.pt")
            self.effective_mode = mask_mode

    def mask(self, frame: Any, cv2: Any, np: Any) -> Any:
        if self.model is None:
            return np.ones(frame.shape[:2], dtype=np.float32)

        device_arg = self.device if self.device.startswith("cuda") else "cpu"
        result = self.model.predict(
            frame,
            conf=self.confidence,
            device=device_arg,
            verbose=False,
        )[0]
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        if self.requested_mode == "yolo_seg" and getattr(result, "masks", None) is not None:
            data = result.masks.data
            masks = data.detach().cpu().numpy() if hasattr(data, "detach") else data
            for item in masks:
                resized = cv2.resize(item.astype("float32"), (frame.shape[1], frame.shape[0]))
                mask = np.maximum(mask, (resized > 0.5).astype("float32"))
        elif getattr(result, "boxes", None) is not None:
            xyxy = result.boxes.xyxy
            boxes = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else xyxy
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(
                    mask,
                    (max(0, int(x1)), max(0, int(y1))),
                    (min(frame.shape[1] - 1, int(x2)), min(frame.shape[0] - 1, int(y2))),
                    1.0,
                    -1,
                )
        return mask


def _imports() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.foreground_motion requires OpenCV and NumPy. "
            "Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _load_yolo(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "video.foreground_motion with mask_mode='yolo' or 'yolo_seg' requires ultralytics. "
            "Install with: pip install -e '.[vision-models]'"
        ) from exc
    return YOLO(model_name)
