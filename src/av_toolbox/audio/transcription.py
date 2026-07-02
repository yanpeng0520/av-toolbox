"""Speech transcription tool backed by faster-whisper."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
        _ensure_pyav_audio_namespace()

        requested_device = context.hardware.device
        device = _resolve_faster_whisper_device(
            candidate_device=context.hardware.resolved_device(),
            requested_device=requested_device,
        )
        if compute_type is None:
            compute_type = "float16" if device.startswith("cuda") and context.hardware.fp16 else "int8"
        model = WhisperModel(
            model_name or "base",
            device=device,
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

        audio = _decode_audio_for_whisper(input_path, max_seconds=max_seconds)
        segments, info = model.transcribe(audio, **transcribe_kwargs)
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
            "device": device,
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
            "device": device,
            "requested_device": requested_device,
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
                f"device={device}",
            ],
        )
        return result


def _import_pyav() -> Any:
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "audio.transcription requires PyAV through faster-whisper. Install with: "
            "pip install -e '.[transcription]'"
        ) from exc

    if not _is_project_av_shadow(av):
        return av

    sys.modules.pop("av", None)
    package_dir = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    try:
        sys.path[:] = [
            entry
            for entry in sys.path
            if _resolved_path(entry or ".") != package_dir
        ]
        try:
            import av as pyav
        except ImportError as exc:
            raise ImportError(
                "audio.transcription imported the project av_toolbox.av package "
                "instead of PyAV, and PyAV could not be imported after removing "
                "the shadowing path. Install with: pip install -e '.[transcription]'"
            ) from exc
    finally:
        sys.path[:] = original_path

    if _is_project_av_shadow(pyav):
        raise ImportError(
            "audio.transcription imported the project av_toolbox.av package instead of PyAV. "
            f"Loaded av from {getattr(pyav, '__file__', 'unknown')}."
        )
    return pyav


def _is_project_av_shadow(module: Any) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    project_av_dir = Path(__file__).resolve().parents[1] / "av"
    try:
        return Path(module_file).resolve().is_relative_to(project_av_dir)
    except OSError:
        return False


def _resolved_path(value: str) -> Path | None:
    try:
        return Path(value).resolve()
    except OSError:
        return None


def _decode_audio_for_whisper(input_path: Path, *, max_seconds: float | None = None) -> np.ndarray:
    """Decode media to the 16 kHz mono float32 waveform expected by Whisper."""
    sampling_rate = 16000
    av = _import_pyav()

    AudioResampler = _pyav_audio_resampler_class(av)
    resampler = AudioResampler(format="s16", layout="mono", rate=sampling_rate)

    chunks: list[np.ndarray] = []
    dtype: np.dtype[Any] | None = None
    max_samples = int(float(max_seconds) * sampling_rate) if max_seconds is not None else None
    decoded_samples = 0

    def append_frame(frame: Any) -> bool:
        nonlocal dtype, decoded_samples
        array = frame.to_ndarray()
        dtype = array.dtype
        flat = np.asarray(array).reshape(-1)
        if max_samples is not None:
            remaining = max_samples - decoded_samples
            if remaining <= 0:
                return False
            flat = flat[:remaining]
        chunks.append(flat.copy())
        decoded_samples += int(flat.size)
        return max_samples is None or decoded_samples < max_samples

    with av.open(str(input_path), mode="r", metadata_errors="ignore") as container:
        for frame in container.decode(audio=0):
            try:
                resampled_frames = resampler.resample(frame)
            except Exception:
                continue
            for resampled in resampled_frames:
                if not append_frame(resampled):
                    break
            if max_samples is not None and decoded_samples >= max_samples:
                break
        if max_samples is None or decoded_samples < max_samples:
            for resampled in resampler.resample(None):
                if not append_frame(resampled):
                    break

    if not chunks or dtype is None:
        raise RuntimeError(f"No audio decoded from {input_path}")
    audio = np.concatenate(chunks).astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scale = max(abs(info.min), abs(info.max))
        if scale:
            audio = audio / float(scale)
    else:
        audio = audio.astype(np.float32, copy=False)
    return audio


def _pyav_audio_resampler_class(av_module: Any) -> Any:
    resampler_class = getattr(av_module, "AudioResampler", None)
    if resampler_class is not None:
        return resampler_class

    try:
        audio_module = importlib.import_module("av.audio")
        resampler_module = importlib.import_module("av.audio.resampler")
    except ImportError as exc:
        raise ImportError(
            "audio.transcription requires a PyAV build with AudioResampler support. "
            f"Loaded av from {getattr(av_module, '__file__', 'unknown')}."
        ) from exc

    if not hasattr(av_module, "audio"):
        av_module.audio = audio_module
    resampler_class = getattr(resampler_module, "AudioResampler", None)
    if resampler_class is None:
        raise ImportError(
            "audio.transcription requires a PyAV build with AudioResampler support. "
            f"Loaded av from {getattr(av_module, '__file__', 'unknown')}."
        )
    return resampler_class


def _resolve_faster_whisper_device(*, candidate_device: str, requested_device: str) -> str:
    """Resolve the runtime device using CTranslate2's actual CUDA support."""
    if not candidate_device.startswith("cuda"):
        return "cpu"
    if _ctranslate2_cuda_device_count() > 0:
        return "cuda"
    if requested_device != "auto":
        raise ValueError(
            "audio.transcription requested CUDA, but the installed CTranslate2 "
            "package does not report CUDA support. Run with device='cpu' or "
            "install a CUDA-enabled CTranslate2/faster-whisper build."
        )
    return "cpu"


def _ctranslate2_cuda_device_count() -> int:
    try:
        import ctranslate2
    except ImportError:
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _ensure_pyav_audio_namespace() -> None:
    """Load PyAV audio submodules used by faster-whisper's decoder."""
    try:
        av = _import_pyav()
    except ImportError:
        return

    try:
        audio_module = importlib.import_module("av.audio")
        importlib.import_module("av.audio.fifo")
        importlib.import_module("av.audio.resampler")
    except ImportError:
        return

    if not hasattr(av, "audio"):
        av.audio = audio_module


def _import_faster_whisper() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "audio.transcription requires faster-whisper. Install with: "
            "pip install -e '.[transcription]'"
        ) from exc
    return WhisperModel
