"""Reusable source-video overlay rendering for video analysis tools."""

from __future__ import annotations

import hashlib
import math
import subprocess
import uuid
from pathlib import Path
from typing import Any


POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


def render_source_video_overlay(
    *,
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    duration: float,
    workspace: Path,
    tool_label: str,
    fps: float = 15.0,
    width: int = 960,
    height: int | None = None,
    mode: str = "metric",
    metric_key: str | None = None,
    metric_label: str | None = None,
    state_key: str | None = None,
    flow_mask: bool = False,
    flow_threshold_px: float = 1.5,
    flow_scale_px: float = 5.0,
    overlay_alpha: float = 0.58,
    timeline_style: str = "panel",
    show_timeline_events: bool = True,
) -> Path:
    """Render a browser-friendly MP4 with analysis marks on top of the source video."""
    cv2, np = _imports()
    if fps <= 0:
        raise ValueError("overlay_fps must be greater than zero")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for overlay: {input_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions: {input_path}")

    duration = max(0.1, float(duration or 0.1))
    if timeline_style not in {"panel", "none", "transparent"}:
        raise ValueError("timeline_style must be one of: panel, none, transparent")
    out_w = _even(max(480, int(width or 960)))
    video_h = _even(max(240, int(round(out_w * src_h / max(1, src_w)))))
    panel_h = 96 if timeline_style == "panel" else 0
    out_h = _even(max(video_h + panel_h, int(height or 0)))
    if height and timeline_style == "panel":
        video_h = _even(max(180, out_h - panel_h))

    tmp_path = workspace / f"{output_path.stem}_{uuid.uuid4().hex}_video_only.mp4"
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {tmp_path}")

    times = _row_times(rows, np)
    metric_values = _metric_values(rows, metric_key, np)
    metric_peak = max(float(metric_values.max()) if metric_values.size else 0.0, 1.0)
    event_ranges = [(float(event.get("start", 0.0)), float(event.get("end", event.get("start", 0.0)))) for event in events]

    bg = (18, 22, 24)
    panel = (238, 248, 245)
    ink = (28, 38, 35)
    muted = (98, 113, 108)
    axis = (176, 196, 190)
    accent = (94, 128, 44)
    event_color = (42, 68, 220)
    playhead = (45, 50, 195)
    warning = (52, 72, 230)
    margin = 54
    plot_x0 = margin
    plot_x1 = out_w - margin
    if timeline_style == "transparent":
        plot_y0 = max(24, video_h - 100)
        plot_y1 = max(plot_y0 + 24, video_h - 24)
    else:
        plot_y0 = video_h + 18
        plot_y1 = out_h - 18
    plot_w = max(1, plot_x1 - plot_x0)
    plot_h = max(1, plot_y1 - plot_y0)
    n_frames = max(1, int(math.ceil(duration * fps)) + 1)
    prev_gray = None
    alpha = min(max(float(overlay_alpha), 0.0), 1.0)
    flow_scale = max(float(flow_scale_px), max(float(flow_threshold_px), 0.0) * 2.0, 1.0)

    try:
        for frame_idx in range(n_frames):
            t = min(frame_idx / fps, duration)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
            ok, frame = cap.read()
            canvas = np.full((out_h, out_w, 3), bg, dtype=np.uint8)
            if ok:
                resized = cv2.resize(frame, (out_w, video_h), interpolation=cv2.INTER_AREA)
                if flow_mask:
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
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
                        mask = mag >= max(float(flow_threshold_px), 0.0)
                        heat = np.clip((mag / flow_scale) * 255.0, 0, 255).astype(np.uint8)
                        color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
                        blended = cv2.addWeighted(resized, 1.0 - alpha, color, alpha, 0.0)
                        overlay = resized.copy()
                        overlay[mask] = blended[mask]
                        resized = overlay
                    prev_gray = gray
                canvas[:video_h, :out_w] = resized

            current_rows = _rows_at_time(rows, times, t)
            current = current_rows[0] if current_rows else {}
            value = _nearest_metric_value(times, metric_values, t)
            if mode in {"boxes", "segments"}:
                _draw_boxes(canvas, current_rows, src_w, src_h, out_w, video_h, cv2, filled=(mode == "segments"))
            elif mode == "pose":
                current_rows = _rows_at_or_before_time(rows, times, t)
                _draw_pose(canvas, current_rows, out_w, video_h, cv2)
            elif mode == "cuts":
                near_cut = any(abs(t - start) <= max(0.16, 1.0 / fps) for start, _ in event_ranges)
                if near_cut:
                    _badge(canvas, "CUT", "boundary", cv2, tone=warning)
            elif mode == "labels":
                label = str(current.get("label", ""))
                confidence = _float(current.get("confidence", 0.0))
                _badge(canvas, _compact_label(label) or tool_label, f"{confidence * 100.0:05.1f}%", cv2)
            else:
                label = metric_label or metric_key or "score"
                state = str(current.get(state_key, "")).lower() if state_key else ""
                state_text = state.replace("_", " ") if state and state != "false" else ""
                if metric_key:
                    base_value = float(metric_values[0]) if metric_values.size else value
                    metric_text = f"{label}: {value:05.2f}  base {base_value:05.2f}"
                    _badge(canvas, tool_label, metric_text, cv2)
                else:
                    _badge(canvas, tool_label, state_text, cv2)

            ph_x = int(round(plot_x0 + min(max(t / duration, 0.0), 1.0) * plot_w))
            if timeline_style == "panel":
                cv2.rectangle(canvas, (0, video_h), (out_w, out_h), panel, cv2.FILLED)
                cv2.putText(canvas, _compact_label(tool_label), (18, video_h + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, ink, 2, cv2.LINE_AA)
                cv2.putText(
                    canvas,
                    f"t={t:05.2f}/{duration:.2f}s",
                    (out_w - 178, video_h + 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    muted,
                    1,
                    cv2.LINE_AA,
                )
                _draw_timeline(
                    canvas,
                    cv2,
                    times,
                    metric_values,
                    event_ranges,
                    duration,
                    plot_x0,
                    plot_w,
                    plot_y0,
                    plot_y1,
                    plot_h,
                    metric_peak,
                    axis,
                    event_color,
                    accent,
                    show_timeline_events=show_timeline_events,
                    value_current=value,
                )
                cv2.line(canvas, (ph_x, 0), (ph_x, video_h), playhead, 2, cv2.LINE_AA)
                cv2.line(canvas, (ph_x, plot_y0 - 5), (ph_x, plot_y1 + 5), playhead, 2, cv2.LINE_AA)
            elif timeline_style == "transparent":
                _draw_transparent_timeline(
                    canvas,
                    cv2,
                    times,
                    metric_values,
                    event_ranges,
                    duration,
                    plot_x0,
                    plot_w,
                    plot_y0,
                    plot_y1,
                    plot_h,
                    metric_peak,
                    playhead_x=ph_x,
                    label=metric_label or metric_key or tool_label,
                    show_timeline_events=show_timeline_events,
                    value_current=value,
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


def _draw_timeline(
    canvas: Any,
    cv2: Any,
    times: Any,
    metric_values: Any,
    event_ranges: list[tuple[float, float]],
    duration: float,
    plot_x0: int,
    plot_w: int,
    plot_y0: int,
    plot_y1: int,
    plot_h: int,
    metric_peak: float,
    axis: tuple[int, int, int],
    event_color: tuple[int, int, int],
    accent: tuple[int, int, int],
    *,
    show_timeline_events: bool,
    value_current: float,
) -> None:
    plot_x1 = plot_x0 + plot_w
    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), axis, 1, cv2.LINE_AA)
    if show_timeline_events and event_ranges:
        for start, end in event_ranges:
            x = int(round(plot_x0 + min(max(start / duration, 0.0), 1.0) * plot_w))
            if abs(end - start) < 1e-6:
                cv2.line(canvas, (x, plot_y0), (x, plot_y1), event_color, 2, cv2.LINE_AA)
            else:
                x2 = int(round(plot_x0 + min(max(end / duration, 0.0), 1.0) * plot_w))
                cv2.rectangle(canvas, (x, plot_y0), (max(x + 2, x2), plot_y1), event_color, cv2.FILLED)

    if times.size and metric_values.size:
        metric_min = float(metric_values.min())
        metric_max = float(metric_values.max())
        if metric_max - metric_min <= 0:
            metric_min = metric_min - 1.0
            metric_max = metric_max + 1.0
        range_den = metric_max - metric_min

        def _normalize_value(value: float) -> float:
            return min(max((value - metric_min) / range_den, 0.0), 1.0)

        bar_top = max(plot_y1 - 16, plot_y0)
        bar_bottom = plot_y1 - 4
        cv2.rectangle(canvas, (plot_x0, bar_top), (plot_x1, bar_bottom), (34, 50, 44), cv2.FILLED)
        fill_right = int(plot_x0 + plot_w * _normalize_value(value_current))
        cv2.rectangle(canvas, (plot_x0, bar_top), (fill_right, bar_bottom), (106, 160, 76), cv2.FILLED)
        cv2.line(canvas, (plot_x0, bar_top), (plot_x1, bar_top), axis, 1, cv2.LINE_AA)
        cv2.line(canvas, (plot_x0, bar_bottom), (plot_x1, bar_bottom), axis, 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"min {metric_min:0.2f}",
            (plot_x0, bar_top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            axis,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"max {metric_max:0.2f}",
            (plot_x0 + 130, bar_top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            axis,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"now {value_current:0.2f}",
            (plot_x1 - 130, bar_top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            axis,
            1,
            cv2.LINE_AA,
        )

        points: list[tuple[int, int]] = []
        for row_t, value in zip(times, metric_values):
            x = int(round(plot_x0 + min(max(float(row_t) / duration, 0.0), 1.0) * plot_w))
            y = int(round(plot_y1 - min(_normalize_value(float(value)), 1.0) * plot_h))
            points.append((x, y))

        for index, point in enumerate(points):
            cv2.circle(canvas, point, 2, (245, 248, 246), -1, cv2.LINE_AA)
            if index == 0:
                continue
            previous = points[index - 1]
            left = min(previous[0], point[0])
            right = max(previous[0], point[0])
            if left == right:
                right = left + 2
            cv2.rectangle(
                canvas,
                (left, point[1]),
                (right, plot_y1),
                (104, 152, 66),
                -1,
            )
            cv2.line(canvas, previous, point, accent, 2, cv2.LINE_AA)


def _draw_transparent_timeline(
    canvas: Any,
    cv2: Any,
    times: Any,
    metric_values: Any,
    event_ranges: list[tuple[float, float]],
    duration: float,
    plot_x0: int,
    plot_w: int,
    plot_y0: int,
    plot_y1: int,
    plot_h: int,
    metric_peak: float,
    *,
    playhead_x: int,
    label: str,
    show_timeline_events: bool,
    value_current: float,
) -> None:
    overlay = canvas.copy()
    plot_x1 = plot_x0 + plot_w
    cv2.rectangle(overlay, (plot_x0 - 14, plot_y0 - 24), (plot_x1 + 14, plot_y1 + 18), (8, 12, 14), cv2.FILLED)
    canvas[:] = cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0.0)
    cv2.putText(canvas, _compact_label(label), (plot_x0, plot_y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (238, 246, 243), 1, cv2.LINE_AA)
    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), (198, 214, 209), 1, cv2.LINE_AA)

    if show_timeline_events:
        for start, end in event_ranges:
            x = int(round(plot_x0 + min(max(start / duration, 0.0), 1.0) * plot_w))
            x2 = int(round(plot_x0 + min(max(end / duration, 0.0), 1.0) * plot_w))
            cv2.rectangle(canvas, (x, plot_y0), (max(x + 2, x2), plot_y1), (38, 74, 190), 1, cv2.LINE_AA)

    if times.size and metric_values.size:
        metric_min = float(metric_values.min())
        metric_max = float(metric_values.max())
        if metric_max - metric_min <= 0:
            metric_min = metric_min - 1.0
            metric_max = metric_max + 1.0
        range_den = metric_max - metric_min

        def _normalize_value(value: float) -> float:
            return min(max((value - metric_min) / range_den, 0.0), 1.0)

        bar_top = max(plot_y1 - 16, plot_y0)
        bar_bottom = plot_y1 - 4
        cv2.rectangle(canvas, (plot_x0, bar_top), (plot_x1, bar_bottom), (24, 40, 34), cv2.FILLED)
        fill_right = int(plot_x0 + plot_w * _normalize_value(value_current))
        cv2.rectangle(canvas, (plot_x0, bar_top), (fill_right, bar_bottom), (64, 114, 188), cv2.FILLED)
        cv2.line(canvas, (plot_x0, bar_top), (plot_x1, bar_top), (198, 214, 209), 1, cv2.LINE_AA)
        cv2.line(canvas, (plot_x0, bar_bottom), (plot_x1, bar_bottom), (198, 214, 209), 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"min {metric_min:0.2f}",
            (plot_x0, bar_top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (220, 238, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"max {metric_max:0.2f}",
            (plot_x0 + 130, bar_top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (220, 238, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"now {value_current:0.2f}",
            (plot_x1 - 130, bar_top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (220, 238, 230),
            1,
            cv2.LINE_AA,
        )

        points: list[tuple[int, int]] = []
        for row_t, value in zip(times, metric_values):
            x = int(round(plot_x0 + min(max(float(row_t) / duration, 0.0), 1.0) * plot_w))
            y = int(round(plot_y1 - min(_normalize_value(float(value)) * 1.0, 1.0) * plot_h))
            points.append((x, y))

        for index, point in enumerate(points):
            cv2.circle(canvas, point, 2, (245, 248, 246), -1, cv2.LINE_AA)
            if index == 0:
                continue
            previous = points[index - 1]
            left = min(previous[0], point[0])
            right = max(previous[0], point[0])
            if left == right:
                right = left + 2
            cv2.rectangle(
                canvas,
                (left, point[1]),
                (right, plot_y1),
                (80, 224, 198),
                -1,
            )
            cv2.line(canvas, previous, point, (80, 224, 198), 2, cv2.LINE_AA)

    cv2.line(canvas, (playhead_x, plot_y0 - 8), (playhead_x, plot_y1 + 8), (45, 50, 215), 2, cv2.LINE_AA)

def mux_video_overlay_with_source_audio(
    *,
    video_only_path: Path,
    source_path: Path,
    output_path: Path,
    duration: float | None = None,
    audio_bitrate: str = "192k",
) -> Path:
    """Encode an overlay MP4 and preserve source audio when the source has it."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_only_path),
        "-i", str(source_path),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", audio_bitrate,
    ]
    source_duration = _source_video_duration(source_path)
    if (
        duration is not None
        and float(duration) > 0
        and source_duration is not None
        and float(duration) < source_duration - 0.03
    ):
        cmd.extend(["-to", f"{float(duration):.6f}"])
    cmd.append("-shortest")
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)
    return output_path


def _draw_boxes(canvas: Any, rows: list[dict[str, Any]], src_w: int, src_h: int, out_w: int, video_h: int, cv2: Any, *, filled: bool) -> None:
    for row in rows[:24]:
        x1 = int(round(_float(row.get("x1", 0.0)) / max(1, src_w) * out_w))
        y1 = int(round(_float(row.get("y1", 0.0)) / max(1, src_h) * video_h))
        x2 = int(round(_float(row.get("x2", 0.0)) / max(1, src_w) * out_w))
        y2 = int(round(_float(row.get("y2", 0.0)) / max(1, src_h) * video_h))
        color = _color_for_label(str(row.get("label", "object")))
        if filled:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
            canvas[:] = cv2.addWeighted(canvas, 0.78, overlay, 0.22, 0.0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = _compact_label(str(row.get("label", "object")))
        confidence = _float(row.get("confidence", 0.0))
        text = f"{label} {confidence * 100.0:.0f}%" if confidence else label
        cv2.putText(canvas, text, (max(4, x1 + 4), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def _draw_pose(canvas: Any, rows: list[dict[str, Any]], out_w: int, video_h: int, cv2: Any) -> None:
    points = {
        int(row["landmark_idx"]): (
            int(round(_float(row.get("x", 0.0)) * out_w)),
            int(round(_float(row.get("y", 0.0)) * video_h)),
            _float(row.get("visibility", 0.0)),
        )
        for row in rows
    }
    for a, b in POSE_CONNECTIONS:
        if a in points and b in points and points[a][2] >= 0.25 and points[b][2] >= 0.25:
            cv2.line(canvas, points[a][:2], points[b][:2], (42, 210, 230), 3, cv2.LINE_AA)
    for x, y, visibility in points.values():
        if visibility >= 0.25:
            cv2.circle(canvas, (x, y), 4, (245, 248, 246), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 5, (42, 210, 230), 1, cv2.LINE_AA)
    if points:
        _badge(canvas, "POSE", f"{len(points)} points", cv2)


def _badge(canvas: Any, title: str, detail: str, cv2: Any, *, tone: tuple[int, int, int] = (245, 248, 246)) -> None:
    title = _compact_label(title).upper()
    detail = detail[:22]
    cv2.rectangle(canvas, (14, 14), (218, 72), (12, 16, 18), cv2.FILLED)
    cv2.rectangle(canvas, (14, 14), (218, 72), (72, 84, 82), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.56, tone, 2, cv2.LINE_AA)
    if detail:
        cv2.putText(canvas, detail, (28, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (218, 229, 225), 1, cv2.LINE_AA)


def _row_times(rows: list[dict[str, Any]], np: Any) -> Any:
    return np.asarray([_float(row.get("timestamp", 0.0)) for row in rows], dtype=float)


def _metric_values(rows: list[dict[str, Any]], metric_key: str | None, np: Any) -> Any:
    if not metric_key:
        return np.asarray([], dtype=float)
    return np.asarray([_float(row.get(metric_key, 0.0)) for row in rows], dtype=float)




def _rows_at_or_before_time(rows: list[dict[str, Any]], times: Any, timestamp: float) -> list[dict[str, Any]]:
    if not rows or getattr(times, "size", 0) == 0:
        return []
    t = float(timestamp)
    index = int((times <= t).sum()) - 1
    if index < 0:
        return []
    target = float(times[index])
    return [row for row in rows if abs(_float(row.get("timestamp", 0.0)) - target) < 1e-6]


def _rows_at_time(rows: list[dict[str, Any]], times: Any, timestamp: float) -> list[dict[str, Any]]:
    if not rows or getattr(times, "size", 0) == 0:
        return []
    index = int(abs(times - float(timestamp)).argmin())
    target = float(times[index])
    return [row for row in rows if abs(_float(row.get("timestamp", 0.0)) - target) < 1e-6]


def _nearest_metric_value(times: Any, values: Any, timestamp: float) -> float:
    if getattr(times, "size", 0) == 0 or getattr(values, "size", 0) == 0:
        return 0.0
    index = int(abs(times - float(timestamp)).argmin())
    return float(values[index])




def _source_video_duration(path: Path) -> float | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0.0 or frame_count <= 0:
        return None
    return frame_count / fps

def _color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    return (80 + digest[0] % 150, 80 + digest[1] % 150, 80 + digest[2] % 150)


def _compact_label(value: str) -> str:
    text = value.replace("_", " ").replace("-", " ").strip()
    return text[:20] or "analysis"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _imports() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "source-video overlays require OpenCV and NumPy. Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np
