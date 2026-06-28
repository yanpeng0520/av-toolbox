"""Video cut detection tool with optional TransNetV2/PySceneDetect backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts


class CutDetectionTool(BaseTool):
    """Detect hard cuts and scene-boundary candidates."""

    name = "video.cut_detection"
    category = "video"
    description = "Detect video cuts with TransNetV2, PySceneDetect, or a lightweight fallback."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        max_seconds: float | None = None,
        threshold: float = 0.5,
        backend: str = "auto",
        weights_path: str | None = None,
        min_distance_frames: int = 3,
        scenedetect_threshold: float = 27.0,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
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
        }
        _, result = write_standard_artifacts(
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
