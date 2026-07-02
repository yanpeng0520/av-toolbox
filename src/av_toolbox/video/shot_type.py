"""Frame-level shot-type classification tool."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import resize_to_width, write_standard_artifacts
from av_toolbox.video.source_overlay import mux_video_overlay_with_source_audio


class ShotTypeTool(BaseTool):
    """Classify sampled frames into film shot-type labels."""

    name = "video.shot_type"
    category = "video"
    description = "Classify sampled frames by shot type using a Transformers image classifier."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 1.0,
        max_seconds: float | None = None,
        model_name: str | None = "pszemraj/beit-large-patch16-512-film-shot-classifier",
        downscale_width: int = 512,
        top_k: int = 3,
        offline: bool = False,
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
            raise ValueError("video.shot_type requires input_path")
        cv2, torch, Image, AutoImageProcessor, AutoModelForImageClassification = _imports()
        metadata = read_video_metadata(input_path)
        model_id = model_name or "pszemraj/beit-large-patch16-512-film-shot-classifier"
        processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=str(context.cache.weights_dir),
            local_files_only=offline,
        )
        model = AutoModelForImageClassification.from_pretrained(
            model_id,
            cache_dir=str(context.cache.weights_dir),
            local_files_only=offline,
        )
        device = context.hardware.resolved_device()
        model.to(device)
        model.eval()

        rows = []
        events = []
        with torch.inference_mode():
            for frame_idx, timestamp, frame in iter_sampled_frames(
                input_path,
                sample_fps=sample_fps,
                max_seconds=max_seconds,
            ):
                resized = resize_to_width(frame, downscale_width, cv2)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                inputs = processor(images=image, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                probs = torch.softmax(model(**inputs).logits[0], dim=-1)
                k = min(top_k, int(probs.shape[-1]))
                scores, indices = torch.topk(probs, k=k)
                top = []
                for score, idx in zip(scores.detach().cpu().tolist(), indices.detach().cpu().tolist()):
                    label = model.config.id2label.get(int(idx), str(int(idx)))
                    top.append({"label": label, "confidence": round(float(score), 6)})
                row = {
                    "timestamp": round(timestamp, 6),
                    "frame_idx": int(frame_idx),
                    "label": top[0]["label"] if top else "",
                    "confidence": top[0]["confidence"] if top else 0.0,
                    "top_labels": json.dumps(top),
                }
                rows.append(row)
                events.append({
                    "start": row["timestamp"],
                    "end": row["timestamp"],
                    "label": row["label"],
                    "data": {**row, "top_labels": top},
                })

        summary = _label_summary(rows)
        summary.update({"sample_count": len(rows), "model": model_id})
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
            "model_name": model_id,
            "downscale_width": downscale_width,
            "top_k": top_k,
            "offline": offline,
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
            csv_fields=["timestamp", "frame_idx", "label", "confidence", "top_labels"],
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            log_lines=[
                f"tool={self.name}",
                f"input={input_path}",
                f"samples={len(rows)}",
            ],
        )
        if export_overlay:
            result.overlay_path = _render_shot_type_overlay(
                input_path=input_path,
                output_path=artifacts.overlay_path,
                rows=rows,
                duration=metadata.duration,
                workspace=context.workspace,
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
            )
        return result



def _overlay_imports() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.shot_type overlays require OpenCV and NumPy. "
            "Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _render_shot_type_overlay(
    *,
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    duration: float,
    workspace: Path,
    fps: float,
    width: int,
    height: int | None,
) -> Path:
    """Dark-slate source overlay with shot-type badge and confidence timeline."""
    cv2, np = _overlay_imports()
    if fps <= 0:
        raise ValueError("overlay_fps must be greater than zero")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for shot-type overlay: {input_path}")
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
    conf = np.asarray([float(row.get("confidence", 0.0)) for row in rows], dtype=float)
    labels = [str(row.get("label", "")) for row in rows]

    # Cool dark-slate theme, shared with the current video analysis overlays.
    letterbox = (20, 18, 16)
    panel_bg = (40, 33, 28)
    text_hi = (243, 240, 238)
    text_lo = (170, 158, 150)
    axis_col = (86, 74, 66)
    ok_col = (150, 200, 96)
    dim_col = (120, 128, 132)
    playhead_col = (90, 205, 255)

    margin_l = 66
    margin_r = 22
    plot_x0 = margin_l
    plot_x1 = out_w - margin_r
    plot_y0 = video_h + 54
    plot_y1 = out_h - 22
    plot_w = max(1, plot_x1 - plot_x0)
    plot_h = max(1, plot_y1 - plot_y0)
    threshold = 0.35

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

            idx = _nearest_index(np, times, t)
            cur_label = labels[idx] if idx is not None else ""
            cur_conf = float(conf[idx]) if idx is not None else 0.0
            _draw_shot_type_badge(
                canvas=canvas, cv2=cv2, label=cur_label, confidence=cur_conf,
                threshold=threshold, ok_col=ok_col, dim_col=dim_col, text_hi=text_hi,
            )

            cv2.rectangle(canvas, (0, video_h), (out_w, out_h), panel_bg, cv2.FILLED)
            cv2.putText(canvas, "Shot type", (24, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_hi, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"confidence >= {threshold:.2f}", (166, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_lo, 1, cv2.LINE_AA)
            time_txt = f"{t:5.2f} / {duration:.2f}s"
            (tw, _th), _ = cv2.getTextSize(time_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(canvas, time_txt, (out_w - margin_r - tw, video_h + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_lo, 1, cv2.LINE_AA)

            playhead_x = int(round(plot_x0 + min(max(t / duration, 0.0), 1.0) * plot_w))
            _draw_shot_type_timeline(
                canvas=canvas, cv2=cv2, times=times, conf=conf, labels=labels,
                duration=duration, threshold=threshold,
                plot_x0=plot_x0, plot_x1=plot_x1, plot_y0=plot_y0, plot_y1=plot_y1,
                plot_w=plot_w, plot_h=plot_h, axis_col=axis_col, text_lo=text_lo,
                text_hi=text_hi, ok_col=ok_col, dim_col=dim_col, playhead_col=playhead_col,
                playhead_x=playhead_x, current_conf=cur_conf,
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


def _nearest_index(np: Any, times: Any, timestamp: float) -> int | None:
    if getattr(times, "size", 0) == 0:
        return None
    return int(np.abs(times - float(timestamp)).argmin())


def _format_shot_label(label: str) -> str:
    text = str(label).replace("_", " ").replace("-", " ").strip()
    return text.title() if text else "-"


def _draw_shot_type_badge(
    *,
    canvas: Any,
    cv2: Any,
    label: str,
    confidence: float,
    threshold: float,
    ok_col: tuple[int, int, int],
    dim_col: tuple[int, int, int],
    text_hi: tuple[int, int, int],
) -> None:
    state_col = ok_col if float(confidence) >= float(threshold) else dim_col
    x, y = 18, 34
    headline = f"{_format_shot_label(label)}  {float(confidence) * 100.0:.0f}%"
    cv2.putText(canvas, headline, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (18, 18, 18), 3, cv2.LINE_AA)
    cv2.putText(canvas, headline, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_hi, 1, cv2.LINE_AA)

    bar_x, bar_y = x, y + 10
    bar_w, bar_h = 240, 16
    fill = min(max(float(confidence), 0.0), 1.0)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (38, 38, 38), cv2.FILLED)
    fill_w = int(round(bar_w * fill))
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), state_col, cv2.FILLED)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (170, 170, 170), 1, cv2.LINE_AA)


def _draw_shot_type_timeline(
    *,
    canvas: Any,
    cv2: Any,
    times: Any,
    conf: Any,
    labels: list[str],
    duration: float,
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
    dim_col: tuple[int, int, int],
    playhead_col: tuple[int, int, int],
    playhead_x: int,
    current_conf: float,
) -> None:
    cv2.line(canvas, (plot_x0, plot_y1), (plot_x1, plot_y1), axis_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, "0", (plot_x0 - 22, plot_y1 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)
    cv2.putText(canvas, "1.0", (plot_x0 - 34, plot_y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_lo, 1, cv2.LINE_AA)

    thr_y = int(round(plot_y1 - min(max(float(threshold), 0.0), 1.0) * plot_h))
    cv2.line(canvas, (plot_x0 - 6, thr_y), (plot_x0 + 8, thr_y), (120, 128, 150), 1, cv2.LINE_AA)

    if getattr(times, "size", 0) != 0 and getattr(conf, "size", 0) != 0:
        def _pt(row_t: float, value: float) -> tuple[int, int]:
            x = int(round(plot_x0 + min(max(float(row_t) / duration, 0.0), 1.0) * plot_w))
            yv = int(round(plot_y1 - min(max(float(value), 0.0), 1.0) * plot_h))
            return x, yv

        points = [_pt(times[i], conf[i]) for i in range(len(conf))]
        for i in range(1, len(points)):
            seg_col = ok_col if (float(conf[i]) >= threshold or float(conf[i - 1]) >= threshold) else dim_col
            cv2.line(canvas, points[i - 1], points[i], seg_col, 2, cv2.LINE_AA)
        for i, point in enumerate(points):
            dot_col = ok_col if float(conf[i]) >= threshold else dim_col
            cv2.circle(canvas, point, 3, dot_col, -1, cv2.LINE_AA)

        last_label_x = -10_000
        boundary_col = (118, 102, 92)
        boundary_text_col = (196, 184, 174)
        for i in range(len(labels)):
            if i > 0 and labels[i] == labels[i - 1]:
                continue
            px = points[i][0]
            cv2.line(canvas, (px, plot_y0 - 10), (px, plot_y1 + 4), boundary_col, 1, cv2.LINE_AA)
            cv2.circle(canvas, (px, plot_y0 - 10), 3, boundary_col, -1, cv2.LINE_AA)
            if px < last_label_x + 86:
                continue
            label = _format_shot_label(labels[i])[:14]
            cv2.putText(canvas, label, (px + 4, plot_y0 - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, boundary_text_col, 1, cv2.LINE_AA)
            last_label_x = px

    cv2.line(canvas, (playhead_x, plot_y0 - 6), (playhead_x, plot_y1 + 4), playhead_col, 2, cv2.LINE_AA)
    cy = int(round(plot_y1 - min(max(float(current_conf), 0.0), 1.0) * plot_h))
    cur_col = ok_col if float(current_conf) >= threshold else dim_col
    cv2.circle(canvas, (playhead_x, cy), 4, cur_col, -1, cv2.LINE_AA)
    cv2.circle(canvas, (playhead_x, cy), 4, text_hi, 1, cv2.LINE_AA)

    val_txt = f"{float(current_conf) * 100.0:.0f}%"
    (vw, _vh), _ = cv2.getTextSize(val_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    tx = playhead_x + 8
    if tx + vw > plot_x1:
        tx = playhead_x - 8 - vw
    ty = min(max(cy - 8, plot_y0 + 14), plot_y1 - 4)
    cv2.putText(canvas, val_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_hi, 1, cv2.LINE_AA)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1

def _imports() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import cv2
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise ImportError(
            "video.shot_type requires OpenCV, Pillow, torch, and transformers. "
            "Install with: pip install -e '.[vision-models]'"
        ) from exc
    return cv2, torch, Image, AutoImageProcessor, AutoModelForImageClassification


def _label_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return {"label_counts": counts}
