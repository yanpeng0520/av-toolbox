"""Camera-shake detection via tracked background translations."""

from __future__ import annotations

from collections import deque
import math
import uuid
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import resize_to_width, write_standard_artifacts
from av_toolbox.video.source_overlay import mux_video_overlay_with_source_audio


class CameraShakeTool(BaseTool):
    """Detect high-frequency frame jitter with sparse optical flow."""

    name = "video.camera_shake"
    category = "video"
    description = "Detect camera shake from detrended sparse optical-flow translations."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 25.0,
        max_seconds: float | None = None,
        downscale_width: int = 512,
        window: int = 25,
        threshold: float = 0.5,
        min_features: int = 20,
        max_features: int = 300,
        redetect_interval: int = 30,
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
            raise ValueError("video.camera_shake requires input_path")
        cv2, np = _imports()
        metadata = read_video_metadata(input_path)

        rows = []
        events = []
        prev_gray = None
        prev_points = None
        translations: deque[tuple[float, float]] = deque(maxlen=max(3, window))
        sample_interval = 1.0 / max(sample_fps, 1e-9)

        for sample_idx, (frame_idx, timestamp, frame) in enumerate(
            iter_sampled_frames(input_path, sample_fps=sample_fps, max_seconds=max_seconds)
        ):
            resized = resize_to_width(frame, downscale_width, cv2)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = gray
                prev_points = _detect_features(gray, cv2, max_features)
                continue

            dx = dy = 0.0
            tracked_features = 0
            curr_points = None
            if prev_points is not None and len(prev_points) > 0:
                curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)
                if curr_points is not None and status is not None:
                    good_prev = prev_points[status.reshape(-1) == 1]
                    good_curr = curr_points[status.reshape(-1) == 1]
                    tracked_features = int(len(good_curr))
                    if tracked_features >= 3:
                        matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
                        if matrix is not None:
                            dx = float(matrix[0, 2])
                            dy = float(matrix[1, 2])
                        else:
                            delta = good_curr.reshape(-1, 2) - good_prev.reshape(-1, 2)
                            dx = float(np.median(delta[:, 0]))
                            dy = float(np.median(delta[:, 1]))
            translations.append((dx, dy))
            shake_score = _shake_score(translations, np)
            is_shaking = shake_score >= threshold and len(translations) >= min(5, window)
            row = {
                "timestamp": round(timestamp, 6),
                "frame_idx": int(frame_idx),
                "translation_x": round(dx, 6),
                "translation_y": round(dy, 6),
                "shake_score": round(shake_score, 6),
                "tracked_features": tracked_features,
                "is_shaking": bool(is_shaking),
            }
            rows.append(row)
            if is_shaking:
                events.append({
                    "start": row["timestamp"],
                    "end": round(timestamp + sample_interval, 6),
                    "label": "camera_shake",
                    "data": row,
                })

            should_redetect = (
                tracked_features < min_features
                or redetect_interval > 0 and sample_idx % redetect_interval == 0
            )
            prev_gray = gray
            prev_points = _detect_features(gray, cv2, max_features) if should_redetect else curr_points

        summary = {
            "sample_count": len(rows),
            "shaking_count": sum(1 for row in rows if row["is_shaking"]),
            "peak_shake_score": round(max((float(row["shake_score"]) for row in rows), default=0.0), 6),
            "mean_shake_score": round(
                sum(float(row["shake_score"]) for row in rows) / len(rows),
                6,
            ) if rows else 0.0,
            "detector": "sparse-lk-detrended-translation",
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
            "window": window,
            "threshold": threshold,
            "min_features": min_features,
            "max_features": max_features,
            "redetect_interval": redetect_interval,
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
                "translation_x",
                "translation_y",
                "shake_score",
                "tracked_features",
                "is_shaking",
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
            result.overlay_path = _render_camera_shake_overlay(
                input_path=input_path,
                output_path=artifacts.overlay_path,
                rows=rows,
                duration=(min(metadata.duration, float(max_seconds)) if max_seconds is not None else metadata.duration),
                workspace=context.workspace,
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
                threshold=threshold,
            )
        return result


def _imports() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.camera_shake requires OpenCV and NumPy. Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _detect_features(gray: Any, cv2: Any, max_features: int) -> Any:
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )


def _shake_score(translations: deque[tuple[float, float]], np: Any) -> float:
    if len(translations) < 3:
        return 0.0
    values = np.asarray(translations, dtype=np.float32)
    x = np.arange(len(values), dtype=np.float32)
    residual_energy = 0.0
    for axis in (0, 1):
        coeff = np.polyfit(x, values[:, axis], deg=1)
        trend = coeff[0] * x + coeff[1]
        residual = values[:, axis] - trend
        residual_energy += float(np.mean(residual * residual))
    return float(np.sqrt(residual_energy))


def _render_camera_shake_overlay(
    *,
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    duration: float,
    workspace: Path,
    fps: float,
    width: int,
    height: int | None,
    threshold: float,
) -> Path:
    cv2, np = _imports()
    if fps <= 0:
        raise ValueError("overlay_fps must be greater than zero")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for camera-shake overlay: {input_path}")
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
    scores = np.asarray([float(row.get("shake_score", 0.0)) for row in rows], dtype=float)
    shaking = np.asarray([bool(row.get("is_shaking", False)) for row in rows], dtype=bool)

    # Cool dark-slate theme (no yellow cast). All colors are BGR for OpenCV.
    letterbox = (20, 18, 16)
    panel_bg = (40, 33, 28)
    text_hi = (243, 240, 238)
    text_lo = (170, 158, 150)
    axis_col = (86, 74, 66)
    ok_col = (150, 200, 96)        # teal-green : stable
    alert_col = (92, 96, 240)      # coral-red  : shaking
    playhead_col = (90, 205, 255)  # amber

    margin_l = 66
    margin_r = 22
    plot_x0 = margin_l
    plot_x1 = out_w - margin_r
    plot_y0 = video_h + 54
    plot_y1 = out_h - 22
    plot_w = max(1, plot_x1 - plot_x0)
    plot_h = max(1, plot_y1 - plot_y0)

    # Baseline pinned at 0 so calm stretches sit on the floor and shake bursts
    # rise; keep the threshold visible even when the whole clip is steady.
    v_max = max(float(scores.max()) if scores.size else 1.0, float(threshold) * 2.0, 1e-6)

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

            current = _shake_row_at_time(rows, times, t)
            current_score = float(current.get("shake_score", 0.0))
            current_state = bool(current.get("is_shaking", False))

            # Single status badge, kept only in the top-left corner.
            _draw_shake_badge(
                canvas=canvas,
                cv2=cv2,
                score=current_score,
                threshold=threshold,
                is_shaking=current_state,
                ok_col=ok_col,
                alert_col=alert_col,
                text_hi=text_hi,
            )

            # Timeline panel under the video.
            cv2.rectangle(canvas, (0, video_h), (out_w, out_h), panel_bg, cv2.FILLED)
            cv2.putText(canvas, "Camera shake", (24, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_hi, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"threshold {threshold:.2f}px", (190, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_lo, 1, cv2.LINE_AA)
            time_txt = f"{t:5.2f} / {duration:.2f}s"
            (tw, _th), _ = cv2.getTextSize(time_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, time_txt, (out_w - margin_r - tw, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_lo, 1, cv2.LINE_AA)

            playhead_x = int(round(plot_x0 + min(max(t / duration, 0.0), 1.0) * plot_w))
            _draw_shake_timeline(
                canvas=canvas,
                cv2=cv2,
                times=times,
                scores=scores,
                shaking=shaking,
                duration=duration,
                v_max=v_max,
                threshold=threshold,
                plot_x0=plot_x0,
                plot_x1=plot_x1,
                plot_y0=plot_y0,
                plot_y1=plot_y1,
                plot_w=plot_w,
                plot_h=plot_h,
                axis_col=axis_col,
                text_lo=text_lo,
                text_hi=text_hi,
                ok_col=ok_col,
                alert_col=alert_col,
                playhead_col=playhead_col,
                playhead_x=playhead_x,
                current_score=current_score,
                current_state=current_state,
            )
            writer.write(canvas)
    finally:
        writer.release()
        cap.release()

    try:
        mux_video_overlay_with_source_audio(
            video_only_path=tmp_path,
            source_path=input_path,
            output_path=output_path,
            duration=duration,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return output_path


def _shake_row_at_time(rows: list[dict[str, Any]], times: Any, timestamp: float) -> dict[str, Any]:
    if not rows or getattr(times, "size", 0) == 0:
        return {}
    index = int(abs(times - float(timestamp)).argmin())
    return rows[index]


def _draw_shake_badge(
    *,
    canvas: Any,
    cv2: Any,
    score: float,
    threshold: float,
    is_shaking: bool,
    ok_col: tuple[int, int, int],
    alert_col: tuple[int, int, int],
    text_hi: tuple[int, int, int],
) -> None:
    """Simple top-left gauge: a 'Shake' label over a green/red fill bar."""
    state_col = alert_col if is_shaking else ok_col
    x, y = 18, 34
    label = f"Shake: {score:.1f}px"
    # Dark outline + light fill so the label stays legible over any footage.
    cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (18, 18, 18), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_hi, 1, cv2.LINE_AA)

    bar_x, bar_y = x, y + 10
    bar_w, bar_h = 208, 16
    fill = min(max(float(score) / max(float(threshold) * 2.0, 1e-6), 0.0), 1.0)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (38, 38, 38), cv2.FILLED)
    fill_w = int(round(bar_w * fill))
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), state_col, cv2.FILLED)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (170, 170, 170), 1, cv2.LINE_AA)


def _draw_shake_timeline(
    *,
    canvas: Any,
    cv2: Any,
    times: Any,
    scores: Any,
    shaking: Any,
    duration: float,
    v_max: float,
    threshold: float,
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
    alert_col: tuple[int, int, int],
    playhead_col: tuple[int, int, int],
    playhead_x: int,
    current_score: float,
    current_state: bool,
) -> None:
    """Color-coded line chart of shake score over time; no gridlines."""
    # Single subtle baseline (x-axis) - the only horizontal rule in the panel.
    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), axis_col, 1, cv2.LINE_AA)

    # y-scale labels kept in the left margin, clear of the plotted line.
    top_label = f"{v_max:.0f}" if v_max >= 10 else f"{v_max:.1f}"
    cv2.putText(canvas, "0", (plot_x0 - 22, plot_y1 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)
    cv2.putText(canvas, top_label, (max(6, plot_x0 - 6 - 9 * len(top_label)), plot_y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)

    # Threshold shown as a short axis tick, not a full-width gridline.
    thr_y = int(round(plot_y1 - min(max(float(threshold) / v_max, 0.0), 1.0) * plot_h))
    cv2.line(canvas, (plot_x0 - 6, thr_y), (plot_x0 + 8, thr_y), (120, 128, 150), 1, cv2.LINE_AA)

    if getattr(times, "size", 0) == 0 or getattr(scores, "size", 0) == 0:
        return

    def _pt(row_t: float, value: float) -> tuple[int, int]:
        x = int(round(plot_x0 + min(max(float(row_t) / duration, 0.0), 1.0) * plot_w))
        y = int(round(plot_y1 - min(max(float(value) / v_max, 0.0), 1.0) * plot_h))
        return x, y

    points = [_pt(times[i], scores[i]) for i in range(len(scores))]
    for i in range(1, len(points)):
        seg_col = alert_col if (bool(shaking[i]) or bool(shaking[i - 1])) else ok_col
        cv2.line(canvas, points[i - 1], points[i], seg_col, 2, cv2.LINE_AA)

    # Playhead stays inside the panel and carries the current reading.
    cv2.line(canvas, (playhead_x, plot_y0 - 6), (playhead_x, plot_y1 + 4), playhead_col, 2, cv2.LINE_AA)
    cur_col = alert_col if current_state else ok_col
    cy = int(round(plot_y1 - min(max(float(current_score) / v_max, 0.0), 1.0) * plot_h))
    cv2.circle(canvas, (playhead_x, cy), 4, cur_col, -1, cv2.LINE_AA)
    cv2.circle(canvas, (playhead_x, cy), 4, text_hi, 1, cv2.LINE_AA)

    val_txt = f"{current_score:.2f}px"
    (vw, _vh), _ = cv2.getTextSize(val_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    tx = playhead_x + 8
    if tx + vw > plot_x1:
        tx = playhead_x - 8 - vw
    ty = min(max(cy - 8, plot_y0 + 12), plot_y1 - 4)
    cv2.putText(canvas, val_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_hi, 1, cv2.LINE_AA)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1
