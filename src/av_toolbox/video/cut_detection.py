"""Video cut detection tool with a TransNetV2-first backend."""

from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts
from av_toolbox.video.source_overlay import mux_video_overlay_with_source_audio


class CutDetectionTool(BaseTool):
    """Detect hard cuts and scene-boundary candidates."""

    name = "video.cut_detection"
    category = "video"
    description = "Detect video cuts with TransNetV2 by default."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        max_seconds: float | None = None,
        threshold: float = 0.5,
        backend: str = "transnetv2",
        weights_path: str | None = None,
        min_distance_frames: int = 3,
        scenedetect_threshold: float = 27.0,
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
            raise ValueError("video.cut_detection requires input_path")
        cv2, np = _imports()
        metadata = read_video_metadata(input_path)
        backend_used, fps, n_frames, scores = _detect_scores(
            input_path=input_path,
            backend=backend,
            threshold=threshold,
            weights_path=weights_path,
            min_distance_frames=min_distance_frames,
            scenedetect_threshold=scenedetect_threshold,
            max_seconds=max_seconds,
            device=context.hardware.resolved_device(),
            cv2=cv2,
            np=np,
        )
        cut_threshold = scenedetect_threshold if backend_used == "scenedetect" else threshold
        cuts = _scores_to_cuts(scores, cut_threshold, min_distance_frames, np)
        rows = [
            {
                "frame": int(cut),
                "timestamp": round(float(cut / fps), 6) if fps > 0 else 0.0,
                "score": round(float(scores[cut]), 6),
                "backend": backend_used,
            }
            for cut in cuts
            if 0 <= cut < len(scores)
        ]
        events = [
            {
                "start": row["timestamp"],
                "end": row["timestamp"],
                "label": "cut",
                "data": row,
            }
            for row in rows
        ]
        segments = _segments(rows, metadata.duration if max_seconds is None else min(metadata.duration, max_seconds))
        summary = {
            "cut_count": len(rows),
            "segment_count": len(segments),
            "backend": backend_used,
            "threshold": cut_threshold,
            "fps": round(float(fps), 6),
            "n_frames": int(n_frames),
        }
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "cuts": rows,
            "segments": segments,
        }
        config = {
            "tool_name": self.name,
            "max_seconds": max_seconds,
            "threshold": threshold,
            "backend": backend,
            "weights_path": weights_path,
            "min_distance_frames": min_distance_frames,
            "scenedetect_threshold": scenedetect_threshold,
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
            csv_fields=["frame", "timestamp", "score", "backend"],
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            log_lines=[
                f"tool={self.name}",
                f"input={input_path}",
                f"backend={backend_used}",
                f"cuts={len(rows)}",
            ],
        )
        if export_overlay:
            result.overlay_path = _render_cut_overlay(
                input_path=input_path,
                output_path=artifacts.overlay_path,
                cuts=rows,
                segments=segments,
                duration=metadata.duration if max_seconds is None else min(metadata.duration, max_seconds),
                workspace=context.workspace,
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
                backend=backend_used,
            )
        return result


def _imports() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.cut_detection requires OpenCV and NumPy. Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _detect_scores(
    *,
    input_path: Path,
    backend: str,
    threshold: float,
    weights_path: str | None,
    min_distance_frames: int,
    scenedetect_threshold: float,
    max_seconds: float | None,
    device: str,
    cv2: Any,
    np: Any,
) -> tuple[str, float, int, Any]:
    if backend in ("auto", "transnetv2"):
        try:
            return _detect_transnetv2(input_path, weights_path, max_seconds, device, cv2, np)
        except Exception:
            if backend == "transnetv2":
                raise
    if backend in ("auto", "scenedetect"):
        try:
            return _detect_scenedetect(input_path, max_seconds, min_distance_frames, scenedetect_threshold, np)
        except Exception:
            if backend == "scenedetect":
                raise
    return _detect_lightweight(input_path, max_seconds, cv2, np)


def _detect_transnetv2(
    input_path: Path,
    weights_path: str | None,
    max_seconds: float | None,
    device_name: str,
    cv2: Any,
    np: Any,
) -> tuple[str, float, int, Any]:
    torch, TransNetV2 = _import_transnetv2()
    model = TransNetV2()
    weights = _resolve_weights_path(weights_path)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    device = torch.device(device_name if device_name.startswith("cuda") else "cpu")
    model.eval().to(device)
    frames, fps, n_frames = _read_resized_frames(input_path, 48, 27, max_seconds, cv2, np)
    if len(frames) == 0:
        return "transnetv2", fps, n_frames, np.zeros(0, dtype=np.float32)
    scores = _predict_transnetv2(model, frames, device, torch, np)
    return "transnetv2", fps, n_frames, scores


def _detect_scenedetect(
    input_path: Path,
    max_seconds: float | None,
    min_distance_frames: int,
    scenedetect_threshold: float,
    np: Any,
) -> tuple[str, float, int, Any]:
    from scenedetect import FrameTimecode, SceneManager, StatsManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(input_path))
    fps = float(video.frame_rate)
    duration_frames = int(max_seconds * fps) if max_seconds is not None else None
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=scenedetect_threshold, min_scene_len=min_distance_frames))
    kwargs: dict[str, Any] = {"show_progress": False}
    if duration_frames is not None:
        kwargs["duration"] = FrameTimecode(duration_frames, fps=fps)
    scene_manager.detect_scenes(video, **kwargs)
    cuts = [_frame_num(cut) for cut in scene_manager.get_cut_list()]
    n_frames = duration_frames or max(cuts, default=0) + 1
    scores = np.zeros(max(n_frames, 0), dtype=np.float32)
    score_key = getattr(ContentDetector, "FRAME_SCORE_KEY", "content_val")
    for frame_idx in range(len(scores)):
        try:
            values = stats_manager.get_metrics(frame_idx, [score_key])
        except Exception:
            continue
        if isinstance(values, dict):
            value = values.get(score_key)
        elif values:
            value = values[0]
        else:
            value = None
        if value is not None:
            scores[frame_idx] = float(value)
    for cut in cuts:
        if 0 <= cut < len(scores):
            scores[cut] = max(float(scores[cut]), float(scenedetect_threshold))
    return "scenedetect", fps, len(scores), scores


def _detect_lightweight(input_path: Path, max_seconds: float | None, cv2: Any, np: Any) -> tuple[str, float, int, Any]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = min(total, int(max_seconds * fps)) if max_seconds is not None and total > 0 else total
    scores = np.zeros(max(max_frames, 0), dtype=np.float32)
    prev_gray = None
    prev_hist = None
    try:
        for idx in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                scores = scores[:idx]
                break
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            if prev_gray is not None and prev_hist is not None:
                diff = float(np.mean(cv2.absdiff(prev_gray, gray)) / 255.0)
                hist_dist = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                scores[idx] = max(diff, hist_dist)
            prev_gray = gray
            prev_hist = hist
    finally:
        cap.release()
    return "lightweight", fps, len(scores), scores


def _import_transnetv2() -> tuple[Any, Any]:
    try:
        import torch
        from transnetv2_pytorch import TransNetV2
    except ImportError:
        try:
            import torch
            from transnetv2pt import TransNetV2
        except ImportError as exc:
            raise ImportError(
                "TransNetV2 bindings are unavailable. Install transnetv2-pytorch "
                "or use backend='scenedetect'/'lightweight'."
            ) from exc
    return torch, TransNetV2


def _resolve_weights_path(weights_path: str | None) -> Path | None:
    candidates = []
    if weights_path:
        candidates.append(Path(weights_path))
    if os.environ.get("TRANSNETV2_WEIGHTS"):
        candidates.append(Path(os.environ["TRANSNETV2_WEIGHTS"]))
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_resized_frames(
    input_path: Path,
    width: int,
    height: int,
    max_seconds: float | None,
    cv2: Any,
    np: Any,
) -> tuple[Any, float, int]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = min(total, int(max_seconds * fps)) if max_seconds is not None and total > 0 else total
    frames = []
    try:
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.asarray(frames, dtype=np.uint8), fps, len(frames)


def _predict_transnetv2(model: Any, frames: Any, device: Any, torch: Any, np: Any) -> Any:
    window_size = 100
    pad = window_size - (len(frames) % window_size)
    padded = frames
    if pad != window_size:
        padded = np.pad(frames, ((0, pad), (0, 0), (0, 0), (0, 0)), mode="edge")
    scores = []
    with torch.inference_mode():
        for start in range(0, len(padded), window_size):
            tensor = torch.from_numpy(padded[start:start + window_size]).unsqueeze(0).to(device)
            output = model(tensor.to(torch.uint8))
            single_frame = output[0] if isinstance(output, tuple) else output
            scores.append(torch.sigmoid(single_frame).cpu().numpy().reshape(-1))
    return np.concatenate(scores)[:len(frames)].astype(np.float32)


def _scores_to_cuts(scores: Any, threshold: float, min_distance_frames: int, np: Any) -> list[int]:
    above = np.flatnonzero(scores > threshold)
    if len(above) == 0:
        return []
    cuts = []
    run = [int(above[0])]
    for idx in map(int, above[1:]):
        if idx == run[-1] + 1:
            run.append(idx)
        else:
            cuts.append(max(run, key=lambda i: scores[i]))
            run = [idx]
    cuts.append(max(run, key=lambda i: scores[i]))
    filtered: list[int] = []
    for cut in cuts:
        if not filtered or cut - filtered[-1] >= min_distance_frames:
            filtered.append(cut)
        elif scores[cut] > scores[filtered[-1]]:
            filtered[-1] = cut
    return filtered


def _segments(cuts: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    boundaries = [float(row["timestamp"]) for row in cuts]
    starts = [0.0] + boundaries
    ends = boundaries + [max(0.0, duration)]
    segments = []
    for idx, (start, end) in enumerate(zip(starts, ends)):
        segments.append({
            "index": idx,
            "start": round(start, 6),
            "end": round(max(start, end), 6),
            "duration": round(max(0.0, end - start), 6),
        })
    return segments


def _frame_num(value: Any) -> int:
    if hasattr(value, "get_frames"):
        return int(value.get_frames())
    if hasattr(value, "frame_num"):
        return int(value.frame_num)
    return int(value)


def _render_cut_overlay(
    *,
    input_path: Path,
    output_path: Path,
    cuts: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    duration: float,
    workspace: Path,
    fps: float,
    width: int,
    height: int | None,
    backend: str,
) -> Path:
    """Dark-slate overlay: shot-segment lane with cut ticks (no scalar line chart)."""
    cv2, np = _imports()
    if fps <= 0:
        raise ValueError("overlay_fps must be greater than zero")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for cut overlay: {input_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions: {input_path}")

    duration = max(0.1, float(duration or 0.1))
    out_w = _even(max(480, int(width or 960)))
    video_h = _even(max(240, int(round(out_w * src_h / max(1, src_w)))))
    panel_h = 132
    out_h = _even(max(video_h + panel_h, int(height or 0)))
    if height:
        video_h = _even(max(180, out_h - panel_h))

    tmp_path = workspace / f"{output_path.stem}_{uuid.uuid4().hex}_video_only.mp4"
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {tmp_path}")

    cut_times = [float(c.get("timestamp", 0.0)) for c in cuts]
    n_shots = max(1, len(segments))

    # Cool dark-slate theme (BGR), shared with video.camera_shake.
    letterbox = (20, 18, 16)
    panel_bg = (40, 33, 28)
    text_hi = (243, 240, 238)
    text_lo = (170, 158, 150)
    axis_col = (86, 74, 66)
    seg_a = (54, 46, 39)
    seg_b = (44, 38, 33)
    seg_current = (78, 66, 54)
    accent = (150, 200, 96)        # teal-green : current shot border
    cut_col = (92, 96, 240)        # coral-red  : cut boundary
    playhead_col = (90, 205, 255)  # amber

    lane_x0 = 56
    lane_x1 = out_w - 24
    lane_y0 = video_h + 54
    lane_y1 = out_h - 26
    lane_w = max(1, lane_x1 - lane_x0)
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

            shot_idx = _current_segment_index(segments, t)
            near_cut = any(abs(t - ct) <= 0.16 for ct in cut_times)
            _draw_cut_badge(
                canvas=canvas, cv2=cv2, np=np, shot_idx=shot_idx, n_shots=n_shots,
                near_cut=near_cut, backend=backend, cut_col=cut_col, accent=accent,
                text_hi=text_hi, text_lo=text_lo,
            )

            cv2.rectangle(canvas, (0, video_h), (out_w, out_h), panel_bg, cv2.FILLED)
            cv2.putText(canvas, "Cut detection", (24, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_hi, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{backend}  |  {len(cuts)} cuts  |  {n_shots} shots", (200, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_lo, 1, cv2.LINE_AA)
            time_txt = f"{t:5.2f} / {duration:.2f}s"
            (tw, _th), _ = cv2.getTextSize(time_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, time_txt, (lane_x1 - tw, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_lo, 1, cv2.LINE_AA)

            playhead_x = int(round(lane_x0 + min(max(t / duration, 0.0), 1.0) * lane_w))
            _draw_cut_timeline(
                canvas=canvas, cv2=cv2, segments=segments, cut_times=cut_times,
                duration=duration, shot_idx=shot_idx, lane_x0=lane_x0, lane_x1=lane_x1,
                lane_y0=lane_y0, lane_y1=lane_y1, lane_w=lane_w, axis_col=axis_col,
                seg_a=seg_a, seg_b=seg_b, seg_current=seg_current, accent=accent,
                cut_col=cut_col, text_hi=text_hi, text_lo=text_lo,
                playhead_col=playhead_col, playhead_x=playhead_x,
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


def _current_segment_index(segments: list[dict[str, Any]], t: float) -> int:
    for seg in segments:
        if float(seg.get("start", 0.0)) <= t < float(seg.get("end", 0.0)):
            return int(seg.get("index", 0))
    if segments and t >= float(segments[-1].get("start", 0.0)):
        return int(segments[-1].get("index", 0))
    return 0


def _draw_cut_badge(
    *,
    canvas: Any,
    cv2: Any,
    np: Any,
    shot_idx: int,
    n_shots: int,
    near_cut: bool,
    backend: str,
    cut_col: tuple[int, int, int],
    accent: tuple[int, int, int],
    text_hi: tuple[int, int, int],
    text_lo: tuple[int, int, int],
) -> None:
    """Translucent card: current shot number, flashing CUT at boundaries."""
    x0, y0, w, h = 16, 16, 200, 84
    x1, y1 = x0 + w, y0 + h
    roi = canvas[y0:y1, x0:x1]
    card = np.empty_like(roi)
    card[:] = (26, 22, 18)
    cv2.addWeighted(card, 0.6, roi, 0.4, 0.0, roi)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 60, 52), 1, cv2.LINE_AA)
    cv2.putText(canvas, "CUT DETECTION", (x0 + 14, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)
    if near_cut:
        cv2.putText(canvas, "CUT", (x0 + 14, y0 + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, cut_col, 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, f"Shot {shot_idx + 1}/{n_shots}", (x0 + 14, y0 + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, accent, 2, cv2.LINE_AA)
    cv2.putText(canvas, str(backend), (x0 + 14, y0 + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_hi, 1, cv2.LINE_AA)


def _draw_cut_timeline(
    *,
    canvas: Any,
    cv2: Any,
    segments: list[dict[str, Any]],
    cut_times: list[float],
    duration: float,
    shot_idx: int,
    lane_x0: int,
    lane_x1: int,
    lane_y0: int,
    lane_y1: int,
    lane_w: int,
    axis_col: tuple[int, int, int],
    seg_a: tuple[int, int, int],
    seg_b: tuple[int, int, int],
    seg_current: tuple[int, int, int],
    accent: tuple[int, int, int],
    cut_col: tuple[int, int, int],
    text_hi: tuple[int, int, int],
    text_lo: tuple[int, int, int],
    playhead_col: tuple[int, int, int],
    playhead_x: int,
) -> None:
    """Shot-segment lane tinted per shot, with cut boundary ticks."""
    def _x(ts: float) -> int:
        return int(round(lane_x0 + min(max(float(ts) / duration, 0.0), 1.0) * lane_w))

    for seg in segments:
        idx = int(seg.get("index", 0))
        xs = _x(float(seg.get("start", 0.0)))
        xe = _x(float(seg.get("end", 0.0)))
        xe = max(xe, xs + 1)
        current = idx == shot_idx
        fill = seg_current if current else (seg_a if idx % 2 == 0 else seg_b)
        cv2.rectangle(canvas, (xs, lane_y0), (xe, lane_y1), fill, cv2.FILLED)
        if current:
            cv2.rectangle(canvas, (xs, lane_y0), (xe, lane_y1), accent, 1, cv2.LINE_AA)
        if xe - xs >= 24:
            label = str(idx + 1)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            tx = xs + (xe - xs - lw) // 2
            ty = lane_y0 + (lane_y1 - lane_y0 + lh) // 2
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        text_hi if current else text_lo, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (lane_x0, lane_y0), (lane_x1, lane_y1), axis_col, 1, cv2.LINE_AA)

    for ct in cut_times:
        cx = _x(ct)
        cv2.line(canvas, (cx, lane_y0 - 5), (cx, lane_y1 + 5), cut_col, 2, cv2.LINE_AA)

    cv2.line(canvas, (playhead_x, lane_y0 - 8), (playhead_x, lane_y1 + 8), playhead_col, 2, cv2.LINE_AA)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1
