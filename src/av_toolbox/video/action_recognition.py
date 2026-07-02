"""SlowFast-style action recognition tool."""

from __future__ import annotations

import json
import math
import uuid
from importlib import resources
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts
from av_toolbox.video.source_overlay import mux_video_overlay_with_source_audio


class ActionRecognitionTool(BaseTool):
    """Classify short video windows with a PyTorchVideo action model."""

    name = "video.action_recognition"
    category = "video"
    description = "Classify short video windows with a SlowFast action-recognition model."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        model_name: str | None = "slowfast_r50",
        max_seconds: float | None = None,
        window_seconds: float = 2.0,
        step_seconds: float = 1.0,
        top_k: int = 3,
        confidence: float = 0.3,
        labels_path: str | None = None,
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
            raise ValueError("video.action_recognition requires input_path")
        deps = _imports()
        torch = deps["torch"]
        metadata = read_video_metadata(input_path)
        labels = _load_labels(labels_path)

        model = torch.hub.load("facebookresearch/pytorchvideo", model_name or "slowfast_r50", pretrained=True)
        device = context.hardware.resolved_device()
        model = model.eval().to(device)
        video = deps["EncodedVideo"].from_path(str(input_path))
        transform = _slowfast_transform(deps)

        duration = metadata.duration
        if max_seconds is not None:
            duration = min(duration, max_seconds)
        rows = []
        events = []
        start = 0.0
        with torch.inference_mode():
            while start + window_seconds <= duration + 1e-6:
                end = start + window_seconds
                clip = video.get_clip(start_sec=start, end_sec=end)
                data = transform(clip)
                inputs = [item.unsqueeze(0).to(device) for item in data["video"]]
                probs = torch.softmax(model(inputs), dim=1)[0]
                k = min(top_k, int(probs.shape[-1]))
                scores, indices = torch.topk(probs, k=k)
                top = []
                for score, idx in zip(scores.detach().cpu().tolist(), indices.detach().cpu().tolist()):
                    top.append({
                        "label": labels[int(idx)] if int(idx) < len(labels) else f"Action {int(idx)}",
                        "confidence": round(float(score), 6),
                    })
                if top and top[0]["confidence"] >= confidence:
                    row = {
                        "start": round(start, 6),
                        "end": round(end, 6),
                        "label": top[0]["label"],
                        "confidence": top[0]["confidence"],
                        "top_labels": json.dumps(top),
                    }
                    rows.append(row)
                    events.append({
                        "start": row["start"],
                        "end": row["end"],
                        "label": row["label"],
                        "data": {**row, "top_labels": top},
                    })
                start += step_seconds

        summary = _label_summary(rows)
        summary.update({"event_count": len(events), "model": model_name or "slowfast_r50"})
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "windows": rows,
        }
        config = {
            "tool_name": self.name,
            "model_name": model_name or "slowfast_r50",
            "max_seconds": max_seconds,
            "window_seconds": window_seconds,
            "step_seconds": step_seconds,
            "top_k": top_k,
            "confidence": confidence,
            "labels_path": labels_path,
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
            csv_fields=["start", "end", "label", "confidence", "top_labels"],
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            log_lines=[
                f"tool={self.name}",
                f"input={input_path}",
                f"events={len(events)}",
            ],
        )
        if export_overlay:
            overlay_rows = [
                {**row, "timestamp": round((float(row["start"]) + float(row["end"])) / 2.0, 6)}
                for row in rows
            ]
            result.overlay_path = _render_action_overlay(
                input_path=input_path,
                output_path=artifacts.overlay_path,
                rows=overlay_rows,
                duration=duration,
                workspace=context.workspace,
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
                threshold=confidence,
            )
        return result


class _PackPathway:
    def __init__(self, torch: Any, alpha: int = 4) -> None:
        self.torch = torch
        self.alpha = alpha

    def __call__(self, frames: Any) -> list[Any]:
        fast_pathway = frames
        slow_pathway = self.torch.index_select(
            frames,
            1,
            self.torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long(),
        )
        return [slow_pathway, fast_pathway]


def _imports() -> dict[str, Any]:
    try:
        import sys
        import torch
        try:
            import torchvision.transforms.functional_tensor  # noqa: F401
        except ImportError:
            import torchvision.transforms.functional as functional
            sys.modules["torchvision.transforms.functional_tensor"] = functional
        from pytorchvideo.data.encoded_video import EncodedVideo
        from pytorchvideo.transforms import ApplyTransformToKey, Normalize, ShortSideScale, UniformTemporalSubsample
        from torchvision.transforms import Compose, Lambda
    except ImportError as exc:
        raise ImportError(
            "video.action_recognition requires torch, torchvision, and pytorchvideo. "
            "Install with: pip install -e '.[action]'"
        ) from exc
    return {
        "torch": torch,
        "EncodedVideo": EncodedVideo,
        "ApplyTransformToKey": ApplyTransformToKey,
        "Normalize": Normalize,
        "ShortSideScale": ShortSideScale,
        "UniformTemporalSubsample": UniformTemporalSubsample,
        "Compose": Compose,
        "Lambda": Lambda,
    }


def _slowfast_transform(deps: dict[str, Any]) -> Any:
    return deps["ApplyTransformToKey"](
        key="video",
        transform=deps["Compose"]([
            deps["UniformTemporalSubsample"](32),
            deps["Lambda"](lambda x: x / 255.0),
            deps["Normalize"]([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            deps["ShortSideScale"](size=256),
            _PackPathway(deps["torch"]),
        ]),
    )


def _load_labels(labels_path: str | None) -> list[str]:
    candidate_paths = []

    if labels_path:
        candidate_paths.append(Path(labels_path))

    try:
        bundled = resources.files("av_toolbox").joinpath("resources/action_labels.txt")
        candidate_paths.append(Path(str(bundled)))
    except Exception:
        pass

    candidate_paths.append(Path(__file__).resolve().parents[1] / "resources" / "action_labels.txt")

    for path in candidate_paths:
        if path.exists():
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [f"Action {idx}" for idx in range(1000)]


def _label_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return {"label_counts": counts}


def _overlay_imports() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "video.action_recognition overlays require OpenCV and NumPy. "
            "Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _render_action_overlay(
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
    """Render source video with only a compact top-left action badge."""
    cv2, np = _overlay_imports()
    if fps <= 0:
        raise ValueError("overlay_fps must be greater than zero")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for action overlay: {input_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions: {input_path}")

    duration = max(0.1, float(duration or 0.1))
    out_w = _even(max(480, int(width or 960)))
    out_h = _even(max(240, int(round(out_w * src_h / max(1, src_w)))))
    if height:
        out_h = _even(max(240, int(height)))

    tmp_path = workspace / f"{output_path.stem}_{uuid.uuid4().hex}_video_only.mp4"
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {tmp_path}")

    times = np.asarray([float(row.get("timestamp", 0.0)) for row in rows], dtype=float)
    conf = np.asarray([float(row.get("confidence", 0.0)) for row in rows], dtype=float)
    labels = [str(row.get("label", "")) for row in rows]

    letterbox = (20, 18, 16)
    text_hi = (243, 240, 238)
    ok_col = (150, 200, 96)
    dim_col = (120, 128, 132)
    threshold = min(max(float(threshold), 0.0), 1.0)

    n_frames = max(1, int(math.ceil(duration * fps)) + 1)
    try:
        for frame_idx in range(n_frames):
            t = min(frame_idx / fps, duration)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
            ok, frame = cap.read()
            canvas = np.full((out_h, out_w, 3), letterbox, dtype=np.uint8)
            if ok:
                resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                canvas[:out_h, :out_w] = resized

            idx = _nearest_index(np, times, t)
            cur_label = labels[idx] if idx is not None else ""
            cur_conf = float(conf[idx]) if idx is not None else 0.0

            _draw_action_badge(
                canvas=canvas, cv2=cv2, label=cur_label, confidence=cur_conf,
                threshold=threshold, ok_col=ok_col, dim_col=dim_col, text_hi=text_hi,
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


def _nearest_index(np: Any, times: Any, timestamp: float) -> int | None:
    if getattr(times, "size", 0) == 0:
        return None
    return int(np.abs(times - float(timestamp)).argmin())


def _format_action(label: str) -> str:
    text = str(label).replace("_", " ").replace("-", " ").strip()
    return text.title() if text else "-"


def _draw_action_badge(
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
    """Simple top-left badge: action label over a confidence fill bar."""
    state_col = ok_col if float(confidence) >= float(threshold) else dim_col
    x, y = 18, 34
    headline = f"{_format_action(label)}  {float(confidence) * 100.0:.0f}%"
    # Dark outline + light fill so the label stays legible over any footage.
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


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1
