"""Image-quality analysis tool: sharpness (blur), luma (exposure), contrast, obstruction."""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts
from av_toolbox.video.source_overlay import mux_video_overlay_with_source_audio


class ImageQualityTool(BaseTool):
    """Per-frame image quality: sharpness/blur, luma/exposure, contrast, and lens obstruction."""

    name = "video.image_quality"
    category = "video"
    description = (
        "Assess per-frame image quality: sharpness (blur), luma (dark/overexposed), "
        "contrast, and lens obstruction."
    )

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
        obstruction_threshold: float = 10.0,
        obstruction_min_luminance: float = 10.0,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        export_overlay: bool = True,
        overlay_fps: float | None = 15.0,
        overlay_width: int = 960,
        overlay_height: int | None = None,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("video.image_quality requires input_path")

        cv2, np = _imports()
        metadata = read_video_metadata(input_path)

        rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        sample_interval = 1.0 / max(sample_fps, 1e-9)
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
            is_obstructed = std_lum < obstruction_threshold and mean_lum >= obstruction_min_luminance

            row = {
                "timestamp": round(timestamp, 6),
                "frame_idx": int(frame_idx),
                "mean_luminance": round(mean_lum, 4),
                "std_luminance": round(std_lum, 4),
                "laplacian_variance": round(lap_var, 4),
                "is_blurry": bool(is_blurry),
                "is_dark": bool(is_dark),
                "is_super_dark": bool(is_super_dark),
                "is_overexposed": bool(is_overexposed),
                "is_obstructed": bool(is_obstructed),
            }
            rows.append(row)
            labels = []
            if is_blurry:
                labels.append("blur")
            if is_dark:
                labels.append("dark")
            if is_overexposed:
                labels.append("overexposed")
            if is_obstructed:
                labels.append("obstructed")
            if labels:
                events.append({
                    "start": round(timestamp, 6),
                    "end": round(timestamp + sample_interval, 6),
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
        config = {
            "tool_name": self.name,
            "sample_fps": sample_fps,
            "max_seconds": max_seconds,
            "blur_threshold": blur_threshold,
            "dark_threshold": dark_threshold,
            "super_dark_threshold": super_dark_threshold,
            "overexposed_threshold": overexposed_threshold,
            "obstruction_threshold": obstruction_threshold,
            "obstruction_min_luminance": obstruction_min_luminance,
            "export_overlay": export_overlay,
            "overlay_fps": overlay_fps,
            "overlay_width": overlay_width,
            "overlay_height": overlay_height,
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
                "mean_luminance",
                "std_luminance",
                "laplacian_variance",
                "is_blurry",
                "is_dark",
                "is_super_dark",
                "is_overexposed",
                "is_obstructed",
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
        if export_overlay:
            result.overlay_path = _render_image_quality_overlay(
                input_path=input_path,
                output_path=artifacts.overlay_path,
                rows=rows,
                duration=metadata.duration,
                workspace=context.workspace,
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
                blur_threshold=blur_threshold,
                dark_threshold=dark_threshold,
                overexposed_threshold=overexposed_threshold,
                obstruction_threshold=obstruction_threshold,
            )
        return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "blurry_count": 0,
            "dark_count": 0,
            "super_dark_count": 0,
            "overexposed_count": 0,
            "obstructed_count": 0,
        }
    return {
        "sample_count": len(rows),
        "blurry_count": sum(1 for row in rows if row["is_blurry"]),
        "dark_count": sum(1 for row in rows if row["is_dark"]),
        "super_dark_count": sum(1 for row in rows if row["is_super_dark"]),
        "overexposed_count": sum(1 for row in rows if row["is_overexposed"]),
        "obstructed_count": sum(1 for row in rows if row["is_obstructed"]),
        "mean_luminance": round(sum(float(row["mean_luminance"]) for row in rows) / len(rows), 4),
        "mean_std_luminance": round(sum(float(row["std_luminance"]) for row in rows) / len(rows), 4),
        "mean_laplacian_variance": round(
            sum(float(row["laplacian_variance"]) for row in rows) / len(rows), 4
        ),
    }


def _imports() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.image_quality requires OpenCV and NumPy. Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _image_quality_row_at_time(
    rows: list[dict[str, Any]],
    times: Any,
    timestamp: float,
) -> dict[str, Any]:
    if not rows or getattr(times, "size", 0) == 0:
        return {
            "laplacian_variance": 0.0,
            "mean_luminance": 0.0,
            "std_luminance": 0.0,
            "is_blurry": False,
            "is_dark": False,
            "is_overexposed": False,
            "is_obstructed": False,
        }
    index = int(abs(times - float(timestamp)).argmin())
    return rows[index]


def _render_image_quality_overlay(
    *,
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    duration: float,
    workspace: Path,
    fps: float,
    width: int,
    height: int | None,
    blur_threshold: float,
    dark_threshold: float,
    overexposed_threshold: float,
    obstruction_threshold: float,
) -> Path:
    """Dark-slate overlay matching the video.camera_shake timeline style."""
    cv2, np = _imports()
    if fps <= 0:
        raise ValueError("overlay_fps must be greater than zero")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for image-quality overlay: {input_path}")
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

    times = np.asarray([float(row.get("timestamp", 0.0)) for row in rows], dtype=float)
    sharp = np.asarray([float(row.get("laplacian_variance", 0.0)) for row in rows], dtype=float)
    luma = np.asarray([float(row.get("mean_luminance", 0.0)) for row in rows], dtype=float)
    contrast = np.asarray([float(row.get("std_luminance", 0.0)) for row in rows], dtype=float)

    # Cool dark-slate theme (BGR), shared with video.camera_shake.
    letterbox = (20, 18, 16)
    panel_bg = (40, 33, 28)
    text_hi = (243, 240, 238)
    text_lo = (170, 158, 150)
    axis_col = (86, 74, 66)
    alert_col = (92, 96, 240)      # coral-red : failing
    sharp_col = (150, 200, 96)     # teal-green
    luma_col = (70, 185, 240)      # amber
    contrast_col = (240, 170, 120)  # periwinkle-blue
    playhead_col = (90, 205, 255)  # amber accent

    metrics = [
        {"key": "sharp", "label": "Sharp", "values": sharp, "color": sharp_col},
        {"key": "luma", "label": "Luma", "values": luma, "color": luma_col},
        {"key": "contrast", "label": "Contrast", "values": contrast, "color": contrast_col},
    ]

    margin_l = 66
    margin_r = 22
    plot_x0 = margin_l
    plot_x1 = out_w - margin_r
    plot_y0 = video_h + 54
    plot_y1 = out_h - 22
    plot_w = max(1, plot_x1 - plot_x0)
    plot_h = max(1, plot_y1 - plot_y0)

    n_frames = max(1, int(math.ceil(duration * fps)) + 1)
    try:
        for frame_idx in range(n_frames):
            t = min(frame_idx / fps, duration)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
            ok, frame = cap.read()
            canvas = np.full((out_h, out_w, 3), letterbox, dtype=np.uint8)
            if ok:
                resized = cv2.resize(frame, (out_w, video_h), interpolation=cv2.INTER_AREA)
                canvas[:video_h, :out_w] = resized

            current = _image_quality_row_at_time(rows, times, t)

            # Top-left card: Sharp / Luma / Contrast bars (red when failing).
            _draw_iq_card(
                canvas=canvas, cv2=cv2, np=np, current=current,
                blur_threshold=blur_threshold, dark_threshold=dark_threshold,
                overexposed_threshold=overexposed_threshold,
                obstruction_threshold=obstruction_threshold,
                sharp_col=sharp_col, luma_col=luma_col, contrast_col=contrast_col,
                alert_col=alert_col, text_hi=text_hi, text_lo=text_lo,
            )

            # Timeline panel under the video.
            cv2.rectangle(canvas, (0, video_h), (out_w, out_h), panel_bg, cv2.FILLED)
            cv2.putText(canvas, "Image quality", (24, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_hi, 1, cv2.LINE_AA)
            _draw_iq_legend(canvas, cv2, metrics, x=196, y=video_h + 28)
            time_txt = f"{t:5.2f} / {duration:.2f}s"
            (tw, _th), _ = cv2.getTextSize(time_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, time_txt, (out_w - margin_r - tw, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_lo, 1, cv2.LINE_AA)

            playhead_x = int(round(plot_x0 + min(max(t / duration, 0.0), 1.0) * plot_w))
            _draw_iq_timeline(
                canvas=canvas, cv2=cv2, times=times, metrics=metrics, duration=duration,
                plot_x0=plot_x0, plot_x1=plot_x1, plot_y0=plot_y0, plot_y1=plot_y1,
                plot_w=plot_w, plot_h=plot_h, axis_col=axis_col, playhead_col=playhead_col,
                playhead_x=playhead_x,
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


def _draw_iq_legend(canvas: Any, cv2: Any, metrics: list[dict[str, Any]], *, x: int, y: int) -> None:
    for spec in metrics:
        label = str(spec["label"]).lower()
        cv2.line(canvas, (x, y - 4), (x + 16, y - 4), spec["color"], 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 196, 200), 1, cv2.LINE_AA)
        (lw, _lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x += 22 + lw + 22


def _draw_iq_card(
    *,
    canvas: Any,
    cv2: Any,
    np: Any,
    current: dict[str, Any],
    blur_threshold: float,
    dark_threshold: float,
    overexposed_threshold: float,
    obstruction_threshold: float,
    sharp_col: tuple[int, int, int],
    luma_col: tuple[int, int, int],
    contrast_col: tuple[int, int, int],
    alert_col: tuple[int, int, int],
    text_hi: tuple[int, int, int],
    text_lo: tuple[int, int, int],
) -> None:
    """Translucent card with Sharp / Luma / Contrast bars; each turns red when failing."""
    sharp = float(current.get("laplacian_variance", 0.0))
    luma = float(current.get("mean_luminance", 0.0))
    contrast = float(current.get("std_luminance", 0.0))
    is_obstructed = bool(current.get("is_obstructed", False))

    sharp_fail = sharp < blur_threshold
    luma_fail = luma < dark_threshold or luma >= overexposed_threshold
    contrast_fail = is_obstructed or contrast < obstruction_threshold

    specs = [
        {"label": "Sharp", "color": sharp_col, "fail": sharp_fail,
         "fill": sharp / max(blur_threshold * 6.0, 1.0), "text": f"{sharp:.0f}"},
        {"label": "Luma", "color": luma_col, "fail": luma_fail,
         "fill": luma / 255.0, "text": f"{luma:.0f}"},
        {"label": "Contrast", "color": contrast_col, "fail": contrast_fail,
         "fill": contrast / 100.0, "text": f"{contrast:.0f}"},
    ]

    x0, y0, w, h = 16, 16, 252, 112
    x1, y1 = x0 + w, y0 + h
    roi = canvas[y0:y1, x0:x1]
    card = np.empty_like(roi)
    card[:] = (26, 22, 18)
    cv2.addWeighted(card, 0.6, roi, 0.4, 0.0, roi)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 60, 52), 1, cv2.LINE_AA)
    cv2.putText(canvas, "IMAGE QUALITY", (x0 + 14, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_lo, 1, cv2.LINE_AA)

    row_y = y0 + 40
    row_h = 24
    bar_x = x0 + 84
    bar_w, bar_h = 108, 9
    for spec in specs:
        col = alert_col if spec["fail"] else spec["color"]
        cv2.putText(canvas, str(spec["label"]), (x0 + 14, row_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        bar_y = row_y + 1
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (58, 52, 46), cv2.FILLED)
        fill_w = int(round(bar_w * min(max(float(spec["fill"]), 0.0), 1.0)))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), col, cv2.FILLED)
        cv2.putText(canvas, str(spec["text"]), (bar_x + bar_w + 8, row_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_hi, 1, cv2.LINE_AA)
        row_y += row_h


def _draw_iq_timeline(
    *,
    canvas: Any,
    cv2: Any,
    times: Any,
    metrics: list[dict[str, Any]],
    duration: float,
    plot_x0: int,
    plot_x1: int,
    plot_y0: int,
    plot_y1: int,
    plot_w: int,
    plot_h: int,
    axis_col: tuple[int, int, int],
    playhead_col: tuple[int, int, int],
    playhead_x: int,
) -> None:
    """Three colored metric lines (each self-normalized); no gridlines."""
    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), axis_col, 1, cv2.LINE_AA)

    if getattr(times, "size", 0) != 0:
        for spec in metrics:
            values = spec["values"]
            if getattr(values, "size", 0) == 0:
                continue
            vmin = float(values.min())
            vmax = float(values.max())
            if vmax - vmin <= 1e-9:
                vmin -= 1.0
                vmax += 1.0
            rng = vmax - vmin
            points: list[tuple[int, int]] = []
            for row_t, value in zip(times, values):
                x = int(round(plot_x0 + min(max(float(row_t) / duration, 0.0), 1.0) * plot_w))
                norm = min(max((float(value) - vmin) / rng, 0.0), 1.0)
                y = int(round(plot_y1 - norm * plot_h))
                points.append((x, y))
            for previous, point in zip(points, points[1:]):
                cv2.line(canvas, previous, point, spec["color"], 2, cv2.LINE_AA)

    # Playhead stays inside the panel.
    cv2.line(canvas, (playhead_x, plot_y0 - 6), (playhead_x, plot_y1 + 4), playhead_col, 2, cv2.LINE_AA)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1
