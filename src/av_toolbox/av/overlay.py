"""Audio-visual overlay rendering."""

from __future__ import annotations

import math
import subprocess
import uuid
from pathlib import Path
from typing import Any


def render_sync_overlay(
    *,
    video_path: str | Path,
    output_path: str | Path,
    y: Any,
    sr: int,
    duration: float,
    motion_samples: list[dict[str, Any]],
    motion_peaks: list[dict[str, Any]],
    audio_events: list[dict[str, Any]],
    sync_matches: list[dict[str, Any]],
    workspace: str | Path,
    fps: float = 15.0,
    width: int = 1280,
    height: int = 720,
    window_sec: float = 6.0,
    playhead_frac: float = 0.35,
) -> Path:
    """Render a video preview plus rolling audio/motion/sync lanes."""
    cv2, np = _imports()
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if window_sec <= 0:
        raise ValueError("window_sec must be greater than zero")

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    tmp_path = workspace_path / f"{output_path.stem}_{uuid.uuid4().hex}_video_only.mp4"

    width = _even(max(480, int(width)))
    height = _even(max(420, int(height)))
    panel_h = min(280, max(220, height // 3))
    header_h = 36
    video_y0 = header_h
    video_y1 = height - panel_h - 8
    panel_y0 = height - panel_h

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions: {video_path}")

    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {tmp_path}")

    y = np.asarray(y, dtype=np.float32)
    wf_peak = float(np.max(np.abs(y))) if y.size else 1.0
    if wf_peak <= 1e-8:
        wf_peak = 1.0

    motion_times = np.asarray([float(row["timestamp"]) for row in motion_samples], dtype=float)
    motion_scores = np.asarray([float(row.get("motion_score", 0.0)) for row in motion_samples], dtype=float)
    peak_times = np.asarray([float(row["timestamp"]) for row in motion_peaks], dtype=float)
    audio_times = np.asarray([float(row["timestamp"]) for row in audio_events], dtype=float)
    match_times = np.asarray([float(row["audio_timestamp"]) for row in sync_matches], dtype=float)

    bg = (18, 20, 24)
    panel_bg = (29, 33, 37)
    axis = (104, 114, 120)
    grid = (48, 55, 60)
    text = (224, 231, 226)
    dim = (148, 160, 160)
    wave = (214, 226, 222)
    motion_color = (86, 158, 242)
    audio_color = (74, 205, 116)
    sync_color = (74, 82, 245)
    playhead = (58, 62, 238)

    label_w = 126
    pad = 18
    plot_x0 = pad + label_w
    plot_x1 = width - pad
    plot_w = max(1, plot_x1 - plot_x0)
    lane_gap = 7
    tick_h = 22
    lane_h = (panel_h - pad * 2 - tick_h - lane_gap * 3) // 4
    wave_y0 = panel_y0 + pad
    lanes = []
    cursor = wave_y0
    for _ in range(4):
        lanes.append((cursor, cursor + lane_h))
        cursor += lane_h + lane_gap
    wave_lane, motion_lane, audio_lane, sync_lane = lanes
    lanes_bottom = sync_lane[1]
    tick_y0 = lanes_bottom + 2

    n_frames = max(1, int(math.ceil(duration * fps)))
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
            span = max(end_t - start_t, 1e-6)

            canvas = np.full((height, width, 3), bg, dtype=np.uint8)
            _text(cv2, canvas, f"AV Sync | t={t:05.2f}/{duration:.2f}s | window={window_sec:.1f}s", pad, 25, 0.56, text)

            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
            ok, frame = cap.read()
            if ok:
                _draw_video(cv2, canvas, frame, 0, video_y0, width, video_y1, src_w, src_h)

            for y0, y1 in lanes:
                cv2.rectangle(canvas, (plot_x0, y0), (plot_x1, y1), panel_bg, cv2.FILLED)
                cv2.rectangle(canvas, (plot_x0, y0), (plot_x1, y1), axis, 1)
            _text(cv2, canvas, "WAVEFORM", pad, (wave_lane[0] + wave_lane[1]) // 2 + 5, 0.47, dim)
            _text(cv2, canvas, "MOTION", pad, (motion_lane[0] + motion_lane[1]) // 2 + 5, 0.47, motion_color)
            _text(cv2, canvas, "AUDIO EVT", pad, (audio_lane[0] + audio_lane[1]) // 2 + 5, 0.47, audio_color)
            _text(cv2, canvas, "SYNC MATCH", pad, (sync_lane[0] + sync_lane[1]) // 2 + 5, 0.43, sync_color)

            for sec in range(math.ceil(start_t), int(math.floor(end_t)) + 1):
                x = int(round(plot_x0 + (sec - start_t) / span * plot_w))
                cv2.line(canvas, (x, wave_lane[0]), (x, sync_lane[1]), grid, 1)
                _text(cv2, canvas, f"{sec}s", x - 9, tick_y0 + 16, 0.38, dim)

            _draw_waveform(cv2, np, canvas, y, sr, wf_peak, start_t, end_t, plot_x0, plot_x1, wave_lane, wave, axis)
            _draw_motion_lane(cv2, canvas, motion_times, motion_scores, peak_times, start_t, end_t, plot_x0, plot_w, motion_lane, motion_color)
            _draw_ticks(cv2, canvas, audio_times, start_t, end_t, plot_x0, plot_w, audio_lane, audio_color, height_frac=0.82)
            _draw_ticks(cv2, canvas, match_times, start_t, end_t, plot_x0, plot_w, sync_lane, sync_color, height_frac=0.9)

            ph_x = int(round(plot_x0 + (t - start_t) / span * plot_w))
            cv2.line(canvas, (ph_x, wave_lane[0] - 3), (ph_x, sync_lane[1] + 3), playhead, 2, cv2.LINE_AA)
            cv2.line(canvas, (ph_x, video_y0), (ph_x, video_y1), playhead, 1, cv2.LINE_AA)
            cv2.circle(canvas, (ph_x, wave_lane[0] - 7), 4, playhead, cv2.FILLED)
            writer.write(canvas)
    finally:
        writer.release()
        cap.release()

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tmp_path),
        "-i", str(video_path),
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
            "AV overlays require OpenCV and NumPy. Install with: pip install -e '.[video,audio]'"
        ) from exc
    return cv2, np


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _text(cv2: Any, img: Any, text: str, x: int, y: int, scale: float, color: tuple[int, int, int]) -> None:
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_video(cv2: Any, canvas: Any, frame: Any, x0: int, y0: int, x1: int, y1: int, src_w: int, src_h: int) -> None:
    area_w = max(1, x1 - x0)
    area_h = max(1, y1 - y0)
    scale = min(area_w / src_w, area_h / src_h)
    target_w = max(1, int(round(src_w * scale)))
    target_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    ox = x0 + (area_w - target_w) // 2
    oy = y0 + (area_h - target_h) // 2
    canvas[oy:oy + target_h, ox:ox + target_w] = resized


def _draw_waveform(
    cv2: Any,
    np: Any,
    img: Any,
    y: Any,
    sr: int,
    peak: float,
    start_t: float,
    end_t: float,
    x0: int,
    x1: int,
    lane: tuple[int, int],
    color: tuple[int, int, int],
    axis_color: tuple[int, int, int],
) -> None:
    width = max(1, x1 - x0)
    y0, y1 = lane
    s0 = max(0, int(start_t * sr))
    s1 = min(y.size, max(s0 + 1, int(end_t * sr)))
    seg = y[s0:s1]
    center = (y0 + y1) // 2
    half = max(2, (y1 - y0) // 2 - 4)
    if seg.size:
        edges = np.linspace(0, seg.size, width + 1).astype(int)
        for idx in range(width):
            part = seg[edges[idx]: max(edges[idx + 1], edges[idx] + 1)]
            top = int(center - float(part.max()) / peak * half)
            bot = int(center - float(part.min()) / peak * half)
            cv2.line(img, (x0 + idx, top), (x0 + idx, bot), color, 1)
    cv2.line(img, (x0, center), (x1, center), axis_color, 1)


def _draw_motion_lane(
    cv2: Any,
    img: Any,
    times: Any,
    scores: Any,
    peak_times: Any,
    start_t: float,
    end_t: float,
    x0: int,
    plot_w: int,
    lane: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    y0, y1 = lane
    span = max(end_t - start_t, 1e-6)
    mask = (times >= start_t) & (times <= end_t)
    for t, score in zip(times[mask], scores[mask]):
        x = int(round(x0 + (float(t) - start_t) / span * plot_w))
        bar_h = int(max(2, min(1.0, float(score)) * (y1 - y0 - 8)))
        cv2.line(img, (x, y1 - 4), (x, y1 - 4 - bar_h), color, 1)
    _draw_ticks(cv2, img, peak_times, start_t, end_t, x0, plot_w, lane, (80, 205, 255), height_frac=0.9)


def _draw_ticks(
    cv2: Any,
    img: Any,
    times: Any,
    start_t: float,
    end_t: float,
    x0: int,
    plot_w: int,
    lane: tuple[int, int],
    color: tuple[int, int, int],
    *,
    height_frac: float,
) -> None:
    span = max(end_t - start_t, 1e-6)
    vals = times[(times >= start_t) & (times <= end_t)]
    y0, y1 = lane
    inset = int((1.0 - height_frac) * (y1 - y0) / 2)
    for t in vals:
        x = int(round(x0 + (float(t) - start_t) / span * plot_w))
        cv2.line(img, (x, y0 + 4 + inset), (x, y1 - 4 - inset), color, 2, cv2.LINE_AA)
