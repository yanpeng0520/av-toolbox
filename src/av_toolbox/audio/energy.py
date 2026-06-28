"""Audio energy feature extraction tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from av_toolbox.core.audio_io import load_audio_samples
from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.result import AVResult
from av_toolbox.core.simple_outputs import write_standard_artifacts


class AudioEnergyTool(BaseTool):
    """Compute RMS, dB energy, and spectral centroid over audio windows."""

    name = "audio.energy"
    category = "audio"
    description = "Compute RMS, dB energy, and spectral centroid over audio windows."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_rate: int | None = 22050,
        hop_length: int | None = 512,
        frame_length: int = 2048,
        max_seconds: float | None = None,
        rms_floor_db: float = -60.0,
        silence_threshold_db: float = -45.0,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("audio.energy requires input_path")
        librosa, np = _imports()

        sr_target = sample_rate or 22050
        hop = hop_length or 512
        y, sr, metadata = load_audio_samples(
            input_path,
            sample_rate=sr_target,
            max_seconds=max_seconds,
        )
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=frame_length,
            hop_length=hop,
        )[0]
        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=frame_length,
            hop_length=hop,
        )[0]
        n = min(len(rms), len(centroid), len(zcr))
        times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)
        rms_db = 20.0 * np.log10(np.maximum(rms[:n], 10 ** (rms_floor_db / 20.0)))

        rows = [
            {
                "timestamp": round(float(times[idx]), 6),
                "rms": round(float(rms[idx]), 8),
                "rms_db": round(float(rms_db[idx]), 4),
                "spectral_centroid": round(float(centroid[idx]), 4),
                "zero_crossing_rate": round(float(zcr[idx]), 6),
                "is_silent": bool(rms_db[idx] <= silence_threshold_db),
            }
            for idx in range(n)
        ]
        events = [
            {
                "start": row["timestamp"],
                "end": round(row["timestamp"] + hop / sr, 6),
                "label": "silence" if row["is_silent"] else "energy",
                "data": row,
            }
            for row in rows
            if row["is_silent"]
        ]
        summary = _summary(rows)
        summary["detector"] = "librosa-rms-centroid"
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
            "sample_rate": sr,
            "hop_length": hop,
            "frame_length": frame_length,
            "max_seconds": max_seconds,
            "rms_floor_db": rms_floor_db,
            "silence_threshold_db": silence_threshold_db,
        }
        _, result = write_standard_artifacts(
            tool_name=self.name,
            input_path=input_path,
            context=context,
            config=config,
            timeline_payload=timeline_payload,
            rows=rows,
            csv_fields=[
                "timestamp",
                "rms",
                "rms_db",
                "spectral_centroid",
                "zero_crossing_rate",
                "is_silent",
            ],
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            log_lines=[
                f"tool={self.name}",
                f"input={input_path}",
                f"duration={metadata.duration:.6f}",
                f"samples={len(rows)}",
                f"events={len(events)}",
            ],
        )
        return result


def _imports() -> tuple[Any, Any]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "audio.energy requires librosa and NumPy. Install with: pip install -e '.[audio]'"
        ) from exc
    return librosa, np


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "mean_rms_db": 0.0,
            "peak_rms_db": 0.0,
            "silence_count": 0,
        }
    return {
        "sample_count": len(rows),
        "mean_rms_db": round(sum(float(row["rms_db"]) for row in rows) / len(rows), 4),
        "peak_rms_db": round(max(float(row["rms_db"]) for row in rows), 4),
        "mean_spectral_centroid": round(
            sum(float(row["spectral_centroid"]) for row in rows) / len(rows),
            4,
        ),
        "silence_count": sum(1 for row in rows if row["is_silent"]),
    }
