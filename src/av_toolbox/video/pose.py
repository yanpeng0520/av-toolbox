"""Multi-person pose tool backed by YOLOv8-pose (ultralytics)."""

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
from av_toolbox.video.yolo_utils import (
    resolve_yolo_device,
    resolve_yolo_model_path,
    yolo_predict_with_auto_cpu_retry,
)


# COCO-17 keypoint skeleton used by YOLOv8-pose.
COCO_POSE_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
]
COCO_KEYPOINT_COUNT = 17


class PoseTool(BaseTool):
    """Detect multi-person pose keypoints with YOLOv8-pose."""

    name = "video.pose"
    category = "video"
    description = "Detect multi-person pose keypoints with YOLOv8-pose."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 10.0,
        max_seconds: float | None = None,
        model_name: str | None = "yolov8n-pose.pt",
        confidence: float = 0.4,
        keypoint_threshold: float = 0.3,
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
            raise ValueError("video.pose requires input_path")
        cv2, np = _imports()
        metadata = read_video_metadata(input_path)
        device = resolve_yolo_device(context.hardware)
        model_label = model_name or "yolov8n-pose.pt"
        model_path = resolve_yolo_model_path(model_label, context.cache)
        model = _load_pose_model(model_path)

        rows: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for frame_idx, timestamp, frame in iter_sampled_frames(
            input_path,
            sample_fps=sample_fps,
            max_seconds=max_seconds,
        ):
            people = _predict_people(model, frame, confidence, device, np)
            if not people:
                continue
            for person_idx, (xy_norm, conf) in enumerate(people):
                for keypoint_idx in range(COCO_KEYPOINT_COUNT):
                    rows.append({
                        "timestamp": round(timestamp, 6),
                        "frame_idx": int(frame_idx),
                        "person_idx": int(person_idx),
                        "keypoint_idx": int(keypoint_idx),
                        "x": round(float(xy_norm[keypoint_idx, 0]), 6),
                        "y": round(float(xy_norm[keypoint_idx, 1]), 6),
                        "confidence": round(float(conf[keypoint_idx]), 6),
                    })
            events.append({
                "start": round(timestamp, 6),
                "end": round(timestamp, 6),
                "label": "pose_tracking",
                "person_count": len(people),
                "data": {"frame_idx": int(frame_idx), "person_count": len(people)},
            })

        person_counts = [int(event["person_count"]) for event in events]
        summary = {
            "pose_frame_count": len(events),
            "keypoint_count": len(rows),
            "max_people": max(person_counts, default=0),
            "mean_people": round(sum(person_counts) / len(person_counts), 4) if person_counts else 0.0,
            "model": model_label,
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
            "model_name": model_label,
            "model_path": model_path,
            "confidence": confidence,
            "keypoint_threshold": keypoint_threshold,
            "device": device,
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
            csv_fields=["timestamp", "frame_idx", "person_idx", "keypoint_idx", "x", "y", "confidence"],
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            log_lines=[
                f"tool={self.name}",
                f"input={input_path}",
                f"device={device}",
                f"pose_frames={len(events)}",
                f"keypoints={len(rows)}",
                f"max_people={summary['max_people']}",
            ],
        )
        if export_overlay:
            result.overlay_path = _render_pose_overlay(
                input_path=input_path,
                output_path=artifacts.overlay_path,
                rows=rows,
                duration=metadata.duration,
                workspace=context.workspace,
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
                model_name=model_label,
                model_path=model_path,
                confidence=confidence,
                keypoint_threshold=keypoint_threshold,
                device=device,
            )
        return result


def _imports() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.pose requires OpenCV and NumPy. Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _load_pose_model(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "video.pose requires ultralytics for YOLOv8-pose. "
            "Install with: pip install -e '.[vision-models]'"
        ) from exc
    return YOLO(model_name)


def _predict_people(model: Any, frame: Any, confidence: float, device: str, np: Any) -> list[tuple[Any, Any]]:
    """Return a list of (normalized_xy[17,2], keypoint_conf[17]) tuples, one per person."""
    device_arg = device if str(device).startswith("cuda") else "cpu"
    result = yolo_predict_with_auto_cpu_retry(
        model,
        frame,
        {"conf": confidence, "device": device_arg, "verbose": False},
    )
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xyn is None:
        return []
    xyn = keypoints.xyn.detach().cpu().numpy() if hasattr(keypoints.xyn, "detach") else np.asarray(keypoints.xyn)
    conf = keypoints.conf
    if conf is not None:
        conf = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
    else:
        conf = np.ones(xyn.shape[:2], dtype=float)
    return [(xyn[i], conf[i]) for i in range(xyn.shape[0])]


def _render_pose_overlay(
    *,
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    duration: float,
    workspace: Path,
    fps: float,
    width: int,
    height: int | None,
    model_name: str,
    model_path: str,
    confidence: float,
    keypoint_threshold: float,
    device: str,
) -> Path:
    """Dark-slate overlay with per-frame multi-person YOLOv8 skeletons (time-synced)."""
    cv2, np = _imports()
    if fps <= 0:
        raise ValueError("overlay_fps must be greater than zero")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for pose overlay: {input_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions: {input_path}")

    duration = max(0.1, float(duration or 0.1))
    out_w = _even(max(480, int(width or 960)))
    video_h = _even(max(240, int(round(out_w * src_h / max(1, src_w)))))
    panel_h = 108
    out_h = _even(max(video_h + panel_h, int(height or 0)))
    if height:
        video_h = _even(max(180, out_h - panel_h))

    tmp_path = workspace / f"{output_path.stem}_{uuid.uuid4().hex}_video_only.mp4"
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {tmp_path}")

    people_times, people_counts = _pose_people_series(rows, np)
    v_max = max(int(people_counts.max()) if people_counts.size else 1, 1)

    # Cool dark-slate theme (BGR), shared with video.camera_shake.
    letterbox = (20, 18, 16)
    panel_bg = (40, 33, 28)
    text_hi = (243, 240, 238)
    text_lo = (170, 158, 150)
    axis_col = (86, 74, 66)
    ok_col = (150, 200, 96)
    dim_col = (120, 128, 132)
    playhead_col = (90, 205, 255)
    joint_fill = (245, 248, 246)
    # Distinct per-person skeleton hues.
    palette = [
        (200, 225, 120), (70, 185, 240), (240, 170, 120),
        (200, 130, 240), (120, 240, 190), (120, 170, 245),
    ]

    margin_l = 66
    margin_r = 22
    plot_x0 = margin_l
    plot_x1 = out_w - margin_r
    plot_y0 = video_h + 40
    plot_y1 = out_h - 18
    plot_w = max(1, plot_x1 - plot_x0)
    plot_h = max(1, plot_y1 - plot_y0)

    model = _load_pose_model(model_path)
    kthr = float(keypoint_threshold)
    n_frames = max(1, int(math.ceil(duration * fps)) + 1)
    try:
        for frame_idx in range(n_frames):
            t = min(frame_idx / fps, duration)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
            ok, frame = cap.read()
            canvas = np.full((out_h, out_w, 3), letterbox, dtype=np.uint8)
            live_people = 0
            if ok:
                resized = cv2.resize(frame, (out_w, video_h), interpolation=cv2.INTER_AREA)
                people = _predict_people(model, resized, confidence, device, np)
                live_people = len(people)
                for person_idx, (xy_norm, conf) in enumerate(people):
                    col = palette[person_idx % len(palette)]
                    pts = [
                        (int(round(xy_norm[k, 0] * out_w)), int(round(xy_norm[k, 1] * video_h)))
                        for k in range(COCO_KEYPOINT_COUNT)
                    ]
                    for a, b in COCO_POSE_SKELETON:
                        if conf[a] >= kthr and conf[b] >= kthr:
                            cv2.line(resized, pts[a], pts[b], col, 2, cv2.LINE_AA)
                    for k in range(COCO_KEYPOINT_COUNT):
                        if conf[k] >= kthr:
                            cv2.circle(resized, pts[k], 3, joint_fill, -1, cv2.LINE_AA)
                            cv2.circle(resized, pts[k], 3, col, 1, cv2.LINE_AA)
                canvas[:video_h, :out_w] = resized

            _draw_pose_badge(
                canvas=canvas, cv2=cv2, np=np, people=live_people,
                ok_col=ok_col, dim_col=dim_col, text_hi=text_hi, text_lo=text_lo,
            )

            cv2.rectangle(canvas, (0, video_h), (out_w, out_h), panel_bg, cv2.FILLED)
            cv2.putText(canvas, "Pose", (24, video_h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_hi, 1, cv2.LINE_AA)
            cv2.putText(canvas, "people tracked", (110, video_h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_lo, 1, cv2.LINE_AA)
            time_txt = f"{t:5.2f} / {duration:.2f}s"
            (tw, _th), _ = cv2.getTextSize(time_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, time_txt, (out_w - margin_r - tw, video_h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_lo, 1, cv2.LINE_AA)

            playhead_x = int(round(plot_x0 + min(max(t / duration, 0.0), 1.0) * plot_w))
            _draw_pose_timeline(
                canvas=canvas, cv2=cv2, times=people_times, counts=people_counts,
                duration=duration, v_max=v_max, plot_x0=plot_x0, plot_x1=plot_x1,
                plot_y0=plot_y0, plot_y1=plot_y1, plot_w=plot_w, plot_h=plot_h,
                axis_col=axis_col, text_lo=text_lo, text_hi=text_hi, ok_col=ok_col,
                playhead_col=playhead_col, playhead_x=playhead_x, current_people=live_people,
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


def _pose_people_series(rows: list[dict[str, Any]], np: Any) -> tuple[Any, Any]:
    """Distinct person count per sampled timestamp (for the panel line)."""
    from collections import defaultdict

    persons: dict[float, set] = defaultdict(set)
    for row in rows:
        ts = round(float(row.get("timestamp", 0.0)), 6)
        persons[ts].add(int(row.get("person_idx", 0)))
    times = sorted(persons)
    if not times:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    counts = [len(persons[t]) for t in times]
    return np.asarray(times, dtype=float), np.asarray(counts, dtype=float)


def _draw_pose_badge(
    *,
    canvas: Any,
    cv2: Any,
    np: Any,
    people: int,
    ok_col: tuple[int, int, int],
    dim_col: tuple[int, int, int],
    text_hi: tuple[int, int, int],
    text_lo: tuple[int, int, int],
) -> None:
    """Translucent card: number of people tracked this frame."""
    state_col = ok_col if people > 0 else dim_col
    x0, y0, w, h = 16, 16, 196, 84
    x1, y1 = x0 + w, y0 + h
    roi = canvas[y0:y1, x0:x1]
    card = np.empty_like(roi)
    card[:] = (26, 22, 18)
    cv2.addWeighted(card, 0.6, roi, 0.4, 0.0, roi)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 60, 52), 1, cv2.LINE_AA)
    cv2.putText(canvas, "POSE", (x0 + 14, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)
    label = f"{people} " + ("person" if people == 1 else "people")
    cv2.putText(canvas, label, (x0 + 14, y0 + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, state_col, 2, cv2.LINE_AA)
    cv2.putText(canvas, "TRACKING" if people > 0 else "NO POSE", (x0 + 14, y0 + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_hi, 1, cv2.LINE_AA)


def _draw_pose_timeline(
    *,
    canvas: Any,
    cv2: Any,
    times: Any,
    counts: Any,
    duration: float,
    v_max: int,
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
    playhead_col: tuple[int, int, int],
    playhead_x: int,
    current_people: int,
) -> None:
    """People-count step line over time; no gridlines."""
    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), axis_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, "0", (plot_x0 - 22, plot_y1 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)
    top_label = str(int(v_max))
    cv2.putText(canvas, top_label, (max(6, plot_x0 - 8 - 9 * len(top_label)), plot_y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)

    if getattr(times, "size", 0) != 0 and getattr(counts, "size", 0) != 0:
        points = []
        for row_t, value in zip(times, counts):
            x = int(round(plot_x0 + min(max(float(row_t) / duration, 0.0), 1.0) * plot_w))
            y = int(round(plot_y1 - min(max(float(value) / v_max, 0.0), 1.0) * plot_h))
            points.append((x, y))
        for previous, point in zip(points, points[1:]):
            cv2.line(canvas, previous, point, ok_col, 2, cv2.LINE_AA)

    cv2.line(canvas, (playhead_x, plot_y0 - 6), (playhead_x, plot_y1 + 4), playhead_col, 2, cv2.LINE_AA)
    cy = int(round(plot_y1 - min(max(float(current_people) / v_max, 0.0), 1.0) * plot_h))
    cv2.circle(canvas, (playhead_x, cy), 4, ok_col if current_people > 0 else text_lo, -1, cv2.LINE_AA)
    cv2.circle(canvas, (playhead_x, cy), 4, text_hi, 1, cv2.LINE_AA)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1
