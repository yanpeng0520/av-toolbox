"""Foreground-biased dense optical-flow motion analysis."""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.cache import ModelCache
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import resize_to_width, write_standard_artifacts
from av_toolbox.video.source_overlay import mux_video_overlay_with_source_audio
from av_toolbox.video.yolo_utils import (
    resolve_yolo_device,
    resolve_yolo_model_path,
    yolo_predict_with_auto_cpu_retry,
)


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
        export_overlay: bool = True,
        overlay_fps: float | None = 15.0,
        overlay_width: int = 960,
        overlay_height: int | None = None,
        overlay_alpha: float = 0.58,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.foreground_motion requires input_path")
        cv2, np = _imports()
        metadata = read_video_metadata(input_path)
        masker = _Masker(
            mask_mode=mask_mode,
            model_name=model_name,
            cache=context.cache,
            confidence=confidence,
            device=resolve_yolo_device(context.hardware) if mask_mode in ("yolo", "yolo_seg") else "cpu",
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
            "model_path": masker.model_path,
            "confidence": confidence,
            "active_threshold_px": active_threshold_px,
            "event_threshold_px": event_threshold_px,
            "export_overlay": export_overlay,
            "overlay_fps": overlay_fps,
            "overlay_width": overlay_width,
            "overlay_height": overlay_height,
            "overlay_alpha": overlay_alpha,
        }
        artifacts, result = write_standard_artifacts(
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
        if export_overlay:
            result.overlay_path = _render_foreground_motion_overlay(
                input_path=input_path,
                output_path=artifacts.overlay_path,
                rows=rows,
                duration=metadata.duration,
                workspace=context.workspace,
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
                active_threshold_px=active_threshold_px,
                event_threshold_px=event_threshold_px,
                overlay_alpha=overlay_alpha,
            )
        return result


class _Masker:
    def __init__(
        self,
        *,
        mask_mode: str,
        model_name: str | None,
        cache: ModelCache,
        confidence: float,
        device: str,
    ) -> None:
        self.requested_mode = mask_mode
        self.effective_mode = "none"
        self.model = None
        self.model_name = model_name
        self.model_path: str | None = None
        self.confidence = confidence
        self.device = device
        if mask_mode in ("yolo", "yolo_seg"):
            model_label = model_name or "yolov8n-seg.pt"
            self.model_path = resolve_yolo_model_path(model_label, cache)
            self.model = _load_yolo(self.model_path)
            self.effective_mode = mask_mode

    def mask(self, frame: Any, cv2: Any, np: Any) -> Any:
        if self.model is None:
            return np.ones(frame.shape[:2], dtype=np.float32)

        device_arg = self.device if self.device.startswith("cuda") else "cpu"
        result = yolo_predict_with_auto_cpu_retry(
            self.model,
            frame,
            {"conf": self.confidence, "device": device_arg, "verbose": False},
        )
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


def _render_foreground_motion_overlay(
    *,
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    duration: float,
    workspace: Path,
    fps: float,
    width: int,
    height: int | None,
    active_threshold_px: float,
    event_threshold_px: float,
    overlay_alpha: float,
) -> Path:
    """Dark-slate overlay: Farneback flow heatmap on the video + active-motion line."""
    cv2, np = _imports()
    if fps <= 0:
        raise ValueError("overlay_fps must be greater than zero")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for foreground-motion overlay: {input_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions: {input_path}")

    duration = max(0.1, float(duration or 0.1))
    out_w = _even(max(480, int(width or 960)))
    video_h = _even(max(240, int(round(out_w * src_h / max(1, src_w)))))
    panel_h = 150
    out_h = _even(max(video_h + panel_h, int(height or 0)))
    if height:
        video_h = _even(max(180, out_h - panel_h))

    tmp_path = workspace / f"{output_path.stem}_{uuid.uuid4().hex}_video_only.mp4"
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {tmp_path}")

    times = np.asarray([float(r.get("timestamp", 0.0)) for r in rows], dtype=float)
    active = np.asarray([float(r.get("active_pct", 0.0)) for r in rows], dtype=float)
    motion = np.asarray([bool(r.get("is_motion", False)) for r in rows], dtype=bool)

    # Cool dark-slate theme (BGR), shared with video.camera_shake.
    letterbox = (20, 18, 16)
    panel_bg = (40, 33, 28)
    text_hi = (243, 240, 238)
    text_lo = (170, 158, 150)
    axis_col = (86, 74, 66)
    ok_col = (150, 200, 96)        # teal-green : active
    dim_col = (120, 128, 132)      # grey       : idle
    playhead_col = (90, 205, 255)  # amber

    margin_l = 66
    margin_r = 22
    plot_x0 = margin_l
    plot_x1 = out_w - margin_r
    plot_y0 = video_h + 54
    plot_y1 = out_h - 22
    plot_w = max(1, plot_x1 - plot_x0)
    plot_h = max(1, plot_y1 - plot_y0)
    v_max = max(float(active.max()) if active.size else 1.0, 1.0)

    active_threshold_px = max(0.0, float(active_threshold_px))
    norm_scale = max(float(event_threshold_px) * 2.0, active_threshold_px * 2.0, 1.0)
    alpha = min(max(float(overlay_alpha), 0.0), 1.0)
    n_frames = max(1, int(math.ceil(duration * fps)) + 1)
    prev_gray = None

    try:
        for frame_idx in range(n_frames):
            t = min(frame_idx / fps, duration)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
            ok, frame = cap.read()
            canvas = np.full((out_h, out_w, 3), letterbox, dtype=np.uint8)
            if ok:
                resized = cv2.resize(frame, (out_w, video_h), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mask = mag >= active_threshold_px
                    heat = np.clip((mag / norm_scale) * 255.0, 0, 255).astype(np.uint8)
                    color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
                    blended = cv2.addWeighted(resized, 1.0 - alpha, color, alpha, 0.0)
                    overlay = resized.copy()
                    overlay[mask] = blended[mask]
                    resized = overlay
                prev_gray = gray
                canvas[:video_h, :out_w] = resized

            current = _fg_row_at_time(rows, times, t)
            cur_active = float(current.get("active_pct", 0.0))
            cur_p95 = float(current.get("p95_magnitude", 0.0))
            cur_state = bool(current.get("is_motion", False))

            _draw_fg_badge(
                canvas=canvas, cv2=cv2, np=np, active=cur_active, p95=cur_p95,
                is_motion=cur_state, ok_col=ok_col, dim_col=dim_col,
                text_hi=text_hi, text_lo=text_lo,
            )

            cv2.rectangle(canvas, (0, video_h), (out_w, out_h), panel_bg, cv2.FILLED)
            cv2.putText(canvas, "Foreground motion", (24, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_hi, 1, cv2.LINE_AA)
            cv2.putText(canvas, "active flow %", (236, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_lo, 1, cv2.LINE_AA)
            time_txt = f"{t:5.2f} / {duration:.2f}s"
            (tw, _th), _ = cv2.getTextSize(time_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, time_txt, (out_w - margin_r - tw, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_lo, 1, cv2.LINE_AA)

            playhead_x = int(round(plot_x0 + min(max(t / duration, 0.0), 1.0) * plot_w))
            _draw_fg_timeline(
                canvas=canvas, cv2=cv2, times=times, active=active, motion=motion,
                duration=duration, v_max=v_max, plot_x0=plot_x0, plot_x1=plot_x1,
                plot_y0=plot_y0, plot_y1=plot_y1, plot_w=plot_w, plot_h=plot_h,
                axis_col=axis_col, text_lo=text_lo, text_hi=text_hi, ok_col=ok_col,
                dim_col=dim_col, playhead_col=playhead_col, playhead_x=playhead_x,
                current_active=cur_active, current_state=cur_state,
            )
            writer.write(canvas)
    finally:
        writer.release()
        cap.release()

    try:
        mux_video_overlay_with_source_audio(
            video_only_path=tmp_path, source_path=input_path,
            output_path=output_path, duration=duration,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return output_path


def _fg_row_at_time(rows: list[dict[str, Any]], times: Any, timestamp: float) -> dict[str, Any]:
    if not rows or getattr(times, "size", 0) == 0:
        return {"active_pct": 0.0, "p95_magnitude": 0.0, "is_motion": False}
    index = int(abs(times - float(timestamp)).argmin())
    return rows[index]


def _draw_fg_badge(
    *,
    canvas: Any,
    cv2: Any,
    np: Any,
    active: float,
    p95: float,
    is_motion: bool,
    ok_col: tuple[int, int, int],
    dim_col: tuple[int, int, int],
    text_hi: tuple[int, int, int],
    text_lo: tuple[int, int, int],
) -> None:
    """Translucent card with the current active-flow % and p95 magnitude."""
    state_col = ok_col if is_motion else dim_col
    x0, y0, w, h = 16, 16, 208, 86
    x1, y1 = x0 + w, y0 + h
    roi = canvas[y0:y1, x0:x1]
    card = np.empty_like(roi)
    card[:] = (26, 22, 18)
    cv2.addWeighted(card, 0.6, roi, 0.4, 0.0, roi)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 60, 52), 1, cv2.LINE_AA)
    cv2.putText(canvas, "FOREGROUND MOTION", (x0 + 14, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{active:.1f}%", (x0 + 14, y0 + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, state_col, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"p95 flow {p95:.1f}px", (x0 + 14, y0 + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_hi, 1, cv2.LINE_AA)


def _draw_fg_timeline(
    *,
    canvas: Any,
    cv2: Any,
    times: Any,
    active: Any,
    motion: Any,
    duration: float,
    v_max: float,
    plot_x0: int,
    plot_x1: int,
    plot_y0: int,
    plot_y1: int,
    plot_w: int,
    plot_h: int,
    axis_col: tuple[int, int, int],
    text_lo: tuple[int, int, int],
    text_hi: tuple[int, int, int],
    ok_col: tuple[int, int, int],
    dim_col: tuple[int, int, int],
    playhead_col: tuple[int, int, int],
    playhead_x: int,
    current_active: float,
    current_state: bool,
) -> None:
    """Active-flow line (green when a motion event, grey when idle); no gridlines."""
    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), axis_col, 1, cv2.LINE_AA)
    top_label = f"{v_max:.0f}%" if v_max >= 10 else f"{v_max:.1f}%"
    cv2.putText(canvas, "0", (plot_x0 - 22, plot_y1 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)
    cv2.putText(canvas, top_label, (max(6, plot_x0 - 6 - 8 * len(top_label)), plot_y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)

    if getattr(times, "size", 0) == 0 or getattr(active, "size", 0) == 0:
        return

    def _pt(row_t: float, value: float) -> tuple[int, int]:
        x = int(round(plot_x0 + min(max(float(row_t) / duration, 0.0), 1.0) * plot_w))
        y = int(round(plot_y1 - min(max(float(value) / v_max, 0.0), 1.0) * plot_h))
        return x, y

    points = [_pt(times[i], active[i]) for i in range(len(active))]
    for i in range(1, len(points)):
        seg_col = ok_col if (bool(motion[i]) or bool(motion[i - 1])) else dim_col
        cv2.line(canvas, points[i - 1], points[i], seg_col, 2, cv2.LINE_AA)

    cv2.line(canvas, (playhead_x, plot_y0 - 6), (playhead_x, plot_y1 + 4), playhead_col, 2, cv2.LINE_AA)
    cy = int(round(plot_y1 - min(max(float(current_active) / v_max, 0.0), 1.0) * plot_h))
    cur_col = ok_col if current_state else dim_col
    cv2.circle(canvas, (playhead_x, cy), 4, cur_col, -1, cv2.LINE_AA)
    cv2.circle(canvas, (playhead_x, cy), 4, text_hi, 1, cv2.LINE_AA)

    val_txt = f"{current_active:.1f}%"
    (vw, _vh), _ = cv2.getTextSize(val_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    tx = playhead_x + 8
    if tx + vw > plot_x1:
        tx = playhead_x - 8 - vw
    ty = min(max(cy - 8, plot_y0 + 12), plot_y1 - 4)
    cv2.putText(canvas, val_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_hi, 1, cv2.LINE_AA)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1
