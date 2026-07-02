"""Spatio-temporal action recognition wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import read_video_metadata
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts
from av_toolbox.video.source_overlay import render_source_video_overlay


class STActionTool(BaseTool):
    """Run an MMAction2-style spatio-temporal action recognizer."""

    name = "video.st_action"
    category = "video"
    description = "Run spatio-temporal action recognition through MMAction2 when configured."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        config_path: str | None = None,
        checkpoint: str | None = None,
        top_k: int = 3,
        confidence: float = 0.0,
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
            raise ValueError("video.st_action requires input_path")
        if not config_path or not checkpoint:
            raise ValueError("video.st_action requires config_path and checkpoint")
        init_recognizer, inference_recognizer = _imports()
        metadata = read_video_metadata(input_path)
        model = init_recognizer(config_path, checkpoint, device=context.hardware.resolved_device())
        result = inference_recognizer(model, str(input_path))
        predictions = _extract_predictions(result, model, top_k)

        rows = [
            {
                "start": 0.0,
                "end": round(metadata.duration, 6),
                "label": item["label"],
                "confidence": item["confidence"],
                "rank": rank,
                "top_labels": json.dumps(predictions),
            }
            for rank, item in enumerate(predictions)
            if item["confidence"] >= confidence
        ]
        events = [
            {
                "start": row["start"],
                "end": row["end"],
                "label": row["label"],
                "data": {**row, "top_labels": predictions},
            }
            for row in rows[:1]
        ]
        summary = {
            "event_count": len(events),
            "prediction_count": len(rows),
            "model": Path(config_path).name,
        }
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "predictions": rows,
        }
        config = {
            "tool_name": self.name,
            "config_path": config_path,
            "checkpoint": checkpoint,
            "top_k": top_k,
            "confidence": confidence,
            "export_overlay": export_overlay,
            "overlay_fps": overlay_fps,
            "overlay_width": overlay_width,
            "overlay_height": overlay_height,
        }
        artifacts, result_obj = write_standard_artifacts(
            tool_name=self.name,
            input_path=input_path,
            context=context,
            config=config,
            timeline_payload=timeline_payload,
            rows=rows,
            csv_fields=["start", "end", "rank", "label", "confidence", "top_labels"],
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            log_lines=[
                f"tool={self.name}",
                f"input={input_path}",
                f"predictions={len(rows)}",
            ],
        )
        if export_overlay:
            overlay_rows = [
                {**row, "timestamp": round((float(row["start"]) + float(row["end"])) / 2.0, 6)}
                for row in rows
            ]
            result_obj.overlay_path = render_source_video_overlay(
                input_path=input_path,
                output_path=artifacts.overlay_path,
                rows=overlay_rows,
                events=events,
                duration=metadata.duration,
                workspace=context.workspace,
                tool_label="st action",
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
                mode="labels",
                metric_key="confidence",
                metric_label="confidence",
            )
        return result_obj


def _imports() -> tuple[Any, Any]:
    try:
        from mmaction.apis import inference_recognizer, init_recognizer
    except ImportError as exc:
        raise ImportError(
            "video.st_action requires MMAction2. Install mmaction2/mmcv and pass "
            "config_path plus checkpoint."
        ) from exc
    return init_recognizer, inference_recognizer


def _extract_predictions(result: Any, model: Any, top_k: int) -> list[dict[str, Any]]:
    pred_score = getattr(result, "pred_score", None)
    if pred_score is None and isinstance(result, dict):
        pred_score = result.get("pred_score")
    if pred_score is None:
        return []
    scores = pred_score.detach().cpu() if hasattr(pred_score, "detach") else pred_score
    values, indices = scores.topk(top_k) if hasattr(scores, "topk") else ([], [])
    classes = getattr(model, "dataset_meta", {}).get("classes", [])
    predictions = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        label = classes[int(idx)] if int(idx) < len(classes) else f"Action {int(idx)}"
        predictions.append({"label": label, "confidence": round(float(value), 6)})
    return predictions
