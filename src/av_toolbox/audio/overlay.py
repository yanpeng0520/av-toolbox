"""Audio timeline overlay rendering."""

from __future__ import annotations

import math
import subprocess
import uuid
from pathlib import Path
from typing import Any


Lane = tuple[str, Any, tuple[int, int, int]]

PHASE_COLORS = {
    "intro": (74, 68, 46),
    "verse": (42, 70, 58),
    "hook": (54, 58, 96),
    "chorus": (54, 58, 96),
    "bridge": (72, 54, 82),
    "breakdown": (72, 54, 82),
    "outro": (64, 58, 48),
    "high_energy": (48, 72, 88),
    "low_energy": (44, 46, 50),
}


def render_timeline_overlay(
    *,
    audio_path: str | Path,
    output_path: str | Path,
    y: Any,
    sr: int,
    duration: float,
    lanes: list[Lane],
    title: str,
    workspace: str | Path,
    segments: list[dict[str, Any]] | None = None,
    fps: float = 15.0,
    width: int = 1280,
    height: int = 540,
    window_sec: float = 8.0,
    playhead_frac: float = 0.35,
) -> Path:
    """Render a rolling waveform/timeline MP4 and mux the source audio."""
    cv2, np = _imports()
    if not lanes:
        raise ValueError("At least one overlay lane is required")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if window_sec <= 0:
        raise ValueError("window_sec must be greater than zero")

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    tmp_path = workspace_path / f"{output_path.stem}_{uuid.uuid4().hex}_video_only.mp4"

    width = _even(max(360, int(width)))
    height = _even(max(260, int(height)))
    y = np.asarray(y, dtype=np.float32)
    wf_peak = float(np.max(np.abs(y))) if y.size else 1.0
    if wf_peak <= 1e-8:
        wf_peak = 1.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {tmp_path}")

    pad = 18
    header_h = 42
    label_w = 128
    plot_x0 = pad + label_w
    plot_x1 = width - pad
    plot_w = max(1, plot_x1 - plot_x0)
    lane_gap = 8
    tick_h = 24
    lane_h = 46 if len(lanes) >= 3 else 56
    lanes_total_h = len(lanes) * lane_h + (len(lanes) - 1) * lane_gap
    wave_y0 = header_h + pad
    available = height - pad - tick_h - 2 - lanes_total_h - lane_gap - wave_y0
    wave_h = max(88, available)
    wave_y1 = wave_y0 + wave_h
    lane_ranges: list[tuple[int, int]] = []
    cursor = wave_y1 + lane_gap
    for _ in lanes:
        lane_ranges.append((cursor, cursor + lane_h))
        cursor += lane_h + lane_gap
    lanes_bottom = lane_ranges[-1][1]
    tick_y0 = lanes_bottom + 2

    bg = (22, 24, 27)
    panel = (31, 35, 38)
    axis = (105, 116, 120)
    grid = (49, 56, 60)
    wave = (212, 226, 220)
    text = (224, 230, 226)
    dim = (150, 164, 160)
    playhead = (58, 62, 238)

    n_frames = max(1, int(math.ceil(duration * fps)))
    segments = segments or []

    try:
        for frame_idx in range(n_frames):
            t = min(frame_idx / fps, duration)
            start_t = t - window_sec * playhead_frac
            end_t = start_t + window_sec
            if start_t < 0:
                end_t -= start_t
                start_t = 0.0
            if end_t > duration:
                start_t = max(0.0, duration - window_sec)
                end_t = duration if duration > window_sec else window_sec
            win_span = max(end_t - start_t, 1e-6)

            canvas = np.full((height, width, 3), bg, dtype=np.uint8)
            header = f"{title} | t={t:05.2f}/{duration:.2f}s | window={window_sec:.1f}s"
            _text(canvas, header, pad, 26, 0.56, text)

            for y0, y1 in [(wave_y0, wave_y1)] + lane_ranges:
                cv2.rectangle(canvas, (plot_x0, y0), (plot_x1, y1), panel, cv2.FILLED)
                cv2.rectangle(canvas, (plot_x0, y0), (plot_x1, y1), axis, 1)

            _draw_segments(canvas, segments, start_t, end_t, plot_x0, plot_x1, wave_y0, wave_y1, cv2)
            _text(canvas, "WAVEFORM", pad, (wave_y0 + wave_y1) // 2 + 5, 0.47, dim)
            for (label, _times, color), (y0, y1) in zip(lanes, lane_ranges):
                _text(canvas, label, pad, (y0 + y1) // 2 + 5, 0.47, color)

            first_grid = math.ceil(start_t)
            for sec in range(first_grid, int(math.floor(end_t)) + 1):
                gx = int(round(plot_x0 + (sec - start_t) / win_span * plot_w))
                cv2.line(canvas, (gx, wave_y0), (gx, lanes_bottom), grid, 1)
                _text(canvas, f"{sec}s", gx - 9, tick_y0 + 17, 0.4, dim)

            _draw_waveform(canvas, y, sr, wf_peak, start_t, end_t, plot_x0, plot_x1, wave_y0, wave_y1, cv2, np, wave, axis)
            for (_label, times, color), (y0, y1) in zip(lanes, lane_ranges):
                vals = np.asarray(times, dtype=float)
                vals = vals[(vals >= start_t) & (vals <= end_t)]
                for event_t in vals:
                    x = int(round(plot_x0 + (event_t - start_t) / win_span * plot_w))
                    cv2.line(canvas, (x, y0 + 4), (x, y1 - 4), color, 2, cv2.LINE_AA)
                    cv2.line(canvas, (x, wave_y0 + 5), (x, wave_y1 - 5), color, 1, cv2.LINE_AA)

            ph_x = int(round(plot_x0 + (t - start_t) / win_span * plot_w))
            cv2.line(canvas, (ph_x, wave_y0 - 3), (ph_x, lanes_bottom + 3), playhead, 2, cv2.LINE_AA)
            cv2.circle(canvas, (ph_x, wave_y0 - 7), 4, playhead, cv2.FILLED)
            writer.write(canvas)
    finally:
        writer.release()

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tmp_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        tmp_path.unlink(missing_ok=True)
    return output_path


def _imports() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Audio overlays require OpenCV and NumPy. Install with: pip install -e '.[video,audio]'"
        ) from exc
    return cv2, np


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _text(img: Any, text: str, x: int, y: int, scale: float, color: tuple[int, int, int]) -> None:
    import cv2

    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_segments(
    img: Any,
    segments: list[dict[str, Any]],
    start_t: float,
    end_t: float,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    cv2: Any,
) -> None:
    span = max(end_t - start_t, 1e-6)
    width = x1 - x0
    for segment in segments:
        a = float(segment.get("start", 0.0))
        b = float(segment.get("end", 0.0))
        if b < start_t or a > end_t:
            continue
        xa = int(round(x0 + (max(a, start_t) - start_t) / span * width))
        xb = int(round(x0 + (min(b, end_t) - start_t) / span * width))
        if xb <= xa:
            continue
        label = str(segment.get("phase_label", segment.get("label", "phase")))
        color = PHASE_COLORS.get(label, (50, 58, 68))
        cv2.rectangle(img, (xa, y0 + 3), (xb, y1 - 3), color, cv2.FILLED)
        if xb - xa > 70:
            _text(img, label, xa + 6, y0 + 22, 0.45, (215, 224, 220))


def _draw_waveform(
    img: Any,
    y: Any,
    sr: int,
    peak: float,
    start_t: float,
    end_t: float,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    cv2: Any,
    np: Any,
    wave_color: tuple[int, int, int],
    axis_color: tuple[int, int, int],
) -> None:
    width = max(1, x1 - x0)
    s0 = max(0, int(start_t * sr))
    s1 = min(y.size, max(s0 + 1, int(end_t * sr)))
    seg = y[s0:s1]
    center = (y0 + y1) // 2
    half = max(2, (y1 - y0) // 2 - 5)
    if seg.size:
        if seg.size < width:
            xs_src = np.linspace(0, max(seg.size - 1, 0), num=seg.size)
            vals = np.interp(np.arange(width), xs_src, seg).astype(np.float32)
            mins = vals
            maxs = vals
        else:
            edges = np.linspace(0, seg.size, width + 1).astype(int)
            mins = np.empty(width, dtype=np.float32)
            maxs = np.empty(width, dtype=np.float32)
            for idx in range(width):
                part = seg[edges[idx]: max(edges[idx + 1], edges[idx] + 1)]
                mins[idx] = float(part.min())
                maxs[idx] = float(part.max())
        top = np.clip(center - (maxs / peak * half).astype(np.int32), y0 + 2, y1 - 2)
        bot = np.clip(center - (mins / peak * half).astype(np.int32), y0 + 2, y1 - 2)
        for px, a, b in zip(range(x0, x0 + width), top, bot):
            cv2.line(img, (int(px), int(a)), (int(px), int(b)), wave_color, 1)
    cv2.line(img, (x0, center), (x1, center), axis_color, 1)
