"""Frame-level shot-type classification tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import resize_to_width, write_standard_artifacts


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
        }
        _, result = write_standard_artifacts(
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
        return result


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
