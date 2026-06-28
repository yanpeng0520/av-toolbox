"""Speech transcription tool backed by faster-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts


class TranscriptionTool(BaseTool):
    """Transcribe speech from audio or video containers using faster-whisper."""

    name = "audio.transcription"
    category = "audio"
    description = "Transcribe speech segments from audio or video with Whisper."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        model_name: str | None = "base",
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        max_seconds: float | None = None,
        compute_type: str | None = None,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("audio.transcription requires input_path")
        WhisperModel = _import_faster_whisper()

        device = context.hardware.resolved_device()
        if compute_type is None:
            compute_type = "float16" if device.startswith("cuda") and context.hardware.fp16 else "int8"
        model = WhisperModel(
            model_name or "base",
            device="cuda" if device.startswith("cuda") else "cpu",
            compute_type=compute_type,
            download_root=str(context.cache.weights_dir),
        )
        transcribe_kwargs: dict[str, Any] = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
        }
        if language:
            transcribe_kwargs["language"] = language
        if max_seconds is not None:
            transcribe_kwargs["clip_timestamps"] = f"0,{float(max_seconds)}"

        segments, info = model.transcribe(str(input_path), **transcribe_kwargs)
        rows = []
        events = []
        for idx, segment in enumerate(segments):
            row = {
                "index": idx,
                "start": round(float(segment.start), 6),
                "end": round(float(segment.end), 6),
                "label": "speech",
                "text": segment.text.strip(),
                "avg_logprob": round(float(getattr(segment, "avg_logprob", 0.0) or 0.0), 6),
                "no_speech_prob": round(float(getattr(segment, "no_speech_prob", 0.0) or 0.0), 6),
            }
            rows.append(row)
            events.append({
                "start": row["start"],
                "end": row["end"],
                "label": "speech",
                "content": row["text"],
                "data": row,
            })

        detected_language = getattr(info, "language", language or "")
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        summary = {
            "segment_count": len(rows),
            "duration": round(duration, 6),
            "language": detected_language,
            "language_probability": round(language_probability, 6),
            "model": model_name or "base",
        }
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": {
                "path": str(input_path),
                "duration": round(duration, 6),
                "language": detected_language,
            },
            "summary": summary,
            "events": events,
            "segments": rows,
        }
        config = {
            "tool_name": self.name,
            "model_name": model_name or "base",
            "language": language,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "max_seconds": max_seconds,
            "compute_type": compute_type,
        }
        _, result = write_standard_artifacts(
            tool_name=self.name,
            input_path=input_path,
            context=context,
            config=config,
            timeline_payload=timeline_payload,
            rows=rows,
            csv_fields=[
                "index",
                "start",
                "end",
                "label",
                "text",
                "avg_logprob",
                "no_speech_prob",
            ],
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            log_lines=[
                f"tool={self.name}",
                f"input={input_path}",
                f"segments={len(rows)}",
                f"language={detected_language}",
            ],
        )
        return result


def _import_faster_whisper() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "audio.transcription requires faster-whisper. Install with: "
            "pip install -e '.[transcription]'"
        ) from exc
    return WhisperModel
