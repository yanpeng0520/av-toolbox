"""SlowFast-style action recognition tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts


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
        }
        _, result = write_standard_artifacts(
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
    if labels_path:
        path = Path(labels_path)
        if path.exists():
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [f"Action {idx}" for idx in range(1000)]


def _label_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return {"label_counts": counts}
