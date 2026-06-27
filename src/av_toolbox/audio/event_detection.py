"""Audio event detection tool."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from av_toolbox.audio.overlay import render_timeline_overlay
from av_toolbox.core.audio_io import load_audio_samples
from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.outputs import make_artifact_paths
from av_toolbox.core.result import AVResult

IMPACT_COLOR = (50, 145, 245)
SPECTRAL_COLOR = (92, 185, 96)
TONAL_COLOR = (196, 116, 215)


class AudioEventDetectionTool(BaseTool):
    """Detect lightweight audio events and energy regions."""

    name = "audio.event_detection"
    category = "audio"
    description = "Detect impacts, energy regions, spectral changes, and tonal shifts."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_rate: int | None = 22050,
        hop_length: int | None = 512,
        max_seconds: float | None = None,
        impact_delta: float = 0.28,
        spectral_delta: float = 0.34,
        tonal_delta: float = 0.34,
        silence_threshold: float = 0.08,
        high_energy_quantile: float = 0.75,
        min_region_seconds: float = 0.35,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        export_overlay: bool = True,
        overlay_fps: float | None = 15.0,
        window_sec: float | None = 8.0,
        overlay_width: int = 1280,
        overlay_height: int = 560,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("audio.event_detection requires input_path")

        sr_target = sample_rate or 22050
        hop = hop_length or 512
        y, sr, metadata = load_audio_samples(input_path, sample_rate=sr_target, max_seconds=max_seconds)
        analysis = analyze_audio_events(
            y,
            sr,
            hop_length=hop,
            impact_delta=impact_delta,
            spectral_delta=spectral_delta,
            tonal_delta=tonal_delta,
            silence_threshold=silence_threshold,
            high_energy_quantile=high_energy_quantile,
            min_region_seconds=min_region_seconds,
        )
        artifacts = make_artifact_paths(
            input_path=input_path,
            output_dir=context.output_dir,
            tool_name=self.name,
        )

        config = {
            "tool_name": self.name,
            "sample_rate": sr,
            "hop_length": hop,
            "max_seconds": max_seconds,
            "impact_delta": impact_delta,
            "spectral_delta": spectral_delta,
            "tonal_delta": tonal_delta,
            "silence_threshold": silence_threshold,
            "high_energy_quantile": high_energy_quantile,
            "min_region_seconds": min_region_seconds,
            "overlay_fps": overlay_fps,
            "window_sec": window_sec,
        }
        artifacts.config_path.write_text(yaml.safe_dump(config, sort_keys=True))

        rows = event_rows(analysis)
        events = [
            {
                "start": row["start"],
                "end": row["end"],
                "label": row["event_type"],
                "data": row,
            }
            for row in rows
        ]
        summary = {
            "duration": round(metadata.duration, 6),
            "event_count": len(events),
            "impact_count": len(analysis["impact_times"]),
            "spectral_change_count": len(analysis["spectral_change_times"]),
            "tonal_shift_count": len(analysis["tonal_shift_times"]),
            "high_energy_count": sum(1 for item in analysis["energy_regions"] if item["event_type"] == "high_energy"),
            "low_energy_count": sum(1 for item in analysis["energy_regions"] if item["event_type"] == "low_energy"),
            "detector": "librosa-feature-event-heuristic",
        }
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "energy_regions": analysis["energy_regions"],
            "impact_times": [round(float(t), 6) for t in analysis["impact_times"]],
            "spectral_change_times": [round(float(t), 6) for t in analysis["spectral_change_times"]],
            "tonal_shift_times": [round(float(t), 6) for t in analysis["tonal_shift_times"]],
            "samples": feature_rows(analysis["features"]),
        }

        if export_json:
            artifacts.timeline_json.write_text(json.dumps(timeline_payload, indent=2))
        if export_csv:
            _write_csv(artifacts.csv_path, rows)
        if export_report:
            artifacts.report_html.write_text(_html_report(timeline_payload))

        overlay_path = None
        if export_overlay:
            overlay_path = render_timeline_overlay(
                audio_path=input_path,
                output_path=artifacts.overlay_path,
                y=y,
                sr=sr,
                duration=metadata.duration,
                lanes=[
                    ("IMPACTS", analysis["impact_times"], IMPACT_COLOR),
                    ("SPECTRAL", analysis["spectral_change_times"], SPECTRAL_COLOR),
                    ("TONAL", analysis["tonal_shift_times"], TONAL_COLOR),
                ],
                title="audio.event_detection",
                workspace=context.workspace,
                segments=analysis["energy_regions"],
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
                window_sec=window_sec or 8.0,
            )

        artifacts.log_path.write_text(
            "\n".join([
                f"tool={self.name}",
                f"input={input_path}",
                f"duration={metadata.duration:.6f}",
                f"events={len(events)}",
                f"impacts={len(analysis['impact_times'])}",
                f"spectral_changes={len(analysis['spectral_change_times'])}",
                f"tonal_shifts={len(analysis['tonal_shift_times'])}",
            ])
            + "\n"
        )

        return AVResult(
            tool_name=self.name,
            input_path=input_path,
            output_dir=context.output_dir,
            overlay_path=overlay_path,
            timeline_json=artifacts.timeline_json if export_json else None,
            csv_path=artifacts.csv_path if export_csv else None,
            report_html=artifacts.report_html if export_report else None,
            config_path=artifacts.config_path,
            log_path=artifacts.log_path,
            metadata={
                "media": metadata.to_dict(),
                "summary": summary,
                "events": len(events),
            },
        )


def analyze_audio_events(
    y: Any,
    sr: int,
    *,
    hop_length: int = 512,
    impact_delta: float = 0.28,
    spectral_delta: float = 0.34,
    tonal_delta: float = 0.34,
    silence_threshold: float = 0.08,
    high_energy_quantile: float = 0.75,
    min_region_seconds: float = 0.35,
) -> dict[str, Any]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "audio.event_detection requires librosa and NumPy. Install with: pip install -e '.[audio]'"
        ) from exc

    y = np.asarray(y, dtype=np.float32)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    n = min(len(rms), len(onset), len(centroid), len(bandwidth))
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    features = {
        "times": times,
        "rms": rms[:n].astype(float),
        "onset": onset[:n].astype(float),
        "centroid": centroid[:n].astype(float),
        "bandwidth": bandwidth[:n].astype(float),
    }

    event_gap = max(1, int(round(0.16 * sr / hop_length)))
    impact_times, impact_scores = _peak_times(
        features["onset"],
        times,
        delta=impact_delta,
        wait=event_gap,
        librosa=librosa,
        np=np,
    )
    spectral_novelty = (
        np.abs(np.diff(_norm(features["centroid"], np), prepend=_norm(features["centroid"], np)[0]))
        + 0.5 * np.abs(np.diff(_norm(features["bandwidth"], np), prepend=_norm(features["bandwidth"], np)[0]))
    )
    spectral_times, spectral_scores = _peak_times(
        spectral_novelty,
        times,
        delta=spectral_delta,
        wait=max(event_gap, int(round(0.24 * sr / hop_length))),
        librosa=librosa,
        np=np,
    )
    tonal_novelty = _tonal_novelty(y, sr, hop_length, n, librosa, np)
    tonal_times, tonal_scores = _peak_times(
        tonal_novelty,
        times,
        delta=tonal_delta,
        wait=max(event_gap, int(round(0.32 * sr / hop_length))),
        librosa=librosa,
        np=np,
    )
    energy_regions = _energy_regions(
        features,
        silence_threshold=silence_threshold,
        high_energy_quantile=high_energy_quantile,
        min_region_seconds=min_region_seconds,
        np=np,
    )
    return {
        "features": {
            **features,
            "spectral_novelty": spectral_novelty[:n].astype(float),
            "tonal_novelty": tonal_novelty[:n].astype(float),
        },
        "impact_times": impact_times,
        "impact_scores": impact_scores,
        "spectral_change_times": spectral_times,
        "spectral_change_scores": spectral_scores,
        "tonal_shift_times": tonal_times,
        "tonal_shift_scores": tonal_scores,
        "energy_regions": energy_regions,
    }


def event_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _add_point_rows(rows, "impact", analysis["impact_times"], analysis["impact_scores"])
    _add_point_rows(rows, "spectral_change", analysis["spectral_change_times"], analysis["spectral_change_scores"])
    _add_point_rows(rows, "tonal_shift", analysis["tonal_shift_times"], analysis["tonal_shift_scores"])
    rows.extend(analysis["energy_regions"])
    return sorted(rows, key=lambda row: (float(row["start"]), str(row["event_type"])))


def feature_rows(features: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for idx, t in enumerate(features["times"]):
        rows.append({
            "timestamp": round(float(t), 6),
            "rms": round(float(features["rms"][idx]), 6),
            "onset_strength": round(float(features["onset"][idx]), 6),
            "spectral_centroid": round(float(features["centroid"][idx]), 4),
            "spectral_bandwidth": round(float(features["bandwidth"][idx]), 4),
            "spectral_novelty": round(float(features["spectral_novelty"][idx]), 6),
            "tonal_novelty": round(float(features["tonal_novelty"][idx]), 6),
        })
    return rows


def _add_point_rows(rows: list[dict[str, Any]], label: str, times: Any, scores: Any) -> None:
    for idx, t in enumerate(times):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        rows.append({
            "event_type": label,
            "start": round(float(t), 6),
            "end": round(float(t) + 0.05, 6),
            "duration": 0.05,
            "score": round(score, 6),
            "feature_value": round(score, 6),
            "details": "",
        })


def _peak_times(values: Any, times: Any, *, delta: float, wait: int, librosa: Any, np: Any) -> tuple[Any, Any]:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    norm = _norm(vals, np)
    peaks = librosa.util.peak_pick(
        norm,
        pre_max=max(1, wait),
        post_max=max(1, wait),
        pre_avg=max(1, wait * 2),
        post_avg=max(1, wait * 2),
        delta=delta,
        wait=max(1, wait),
    )
    peaks = peaks[peaks < len(times)]
    return np.asarray(times[peaks], dtype=float), np.asarray(norm[peaks], dtype=float)


def _tonal_novelty(y: Any, sr: int, hop_length: int, n: int, librosa: Any, np: Any) -> Any:
    if n <= 0:
        return np.empty(0, dtype=float)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    if chroma.shape[1] < 2:
        return np.zeros(n, dtype=float)
    chroma_n = chroma / np.clip(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-8, None)
    novelty = np.r_[0.0, np.sqrt(np.sum(np.diff(chroma_n, axis=1) ** 2, axis=0))]
    if novelty.size < n:
        novelty = np.pad(novelty, (0, n - novelty.size))
    return novelty[:n]


def _energy_regions(
    features: dict[str, Any],
    *,
    silence_threshold: float,
    high_energy_quantile: float,
    min_region_seconds: float,
    np: Any,
) -> list[dict[str, Any]]:
    times = np.asarray(features["times"], dtype=float)
    rms_norm = _norm(features["rms"], np)
    if times.size == 0:
        return []
    step = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    smooth_frames = max(1, int(round(0.5 / max(step, 1e-6))))
    energy = _smooth(rms_norm, smooth_frames, np)
    high_threshold = float(np.quantile(energy, min(max(high_energy_quantile, 0.0), 1.0)))
    regions = []
    regions.extend(_mask_regions(times, energy <= silence_threshold, "low_energy", min_region_seconds, energy, np))
    regions.extend(_mask_regions(times, energy >= high_threshold, "high_energy", min_region_seconds, energy, np))
    return sorted(regions, key=lambda item: (item["start"], item["event_type"]))


def _smooth(values: Any, width: int, np: Any) -> Any:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0 or width <= 1:
        return vals
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(vals, kernel, mode="same")


def _mask_regions(times: Any, mask: Any, label: str, min_seconds: float, values: Any, np: Any) -> list[dict[str, Any]]:
    if not np.any(mask):
        return []
    step = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    regions = []
    start_idx = None
    for idx, active in enumerate(mask):
        if active and start_idx is None:
            start_idx = idx
        if start_idx is not None and (not active or idx == len(mask) - 1):
            end_idx = idx if active and idx == len(mask) - 1 else idx - 1
            start = float(times[start_idx])
            end = float(times[end_idx] + step)
            duration = max(0.0, end - start)
            if duration >= min_seconds:
                score = float(np.mean(values[start_idx:end_idx + 1])) if end_idx >= start_idx else 0.0
                regions.append({
                    "event_type": label,
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "duration": round(duration, 6),
                    "score": round(score, 6),
                    "feature_value": round(score, 6),
                    "details": "rms_region",
                    "label": label,
                })
            start_idx = None
    return regions


def _norm(values: Any, np: Any) -> Any:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return vals
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo <= 1e-8:
        return np.zeros_like(vals)
    return (vals - lo) / (hi - lo)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "event_type",
        "start",
        "end",
        "duration",
        "score",
        "feature_value",
        "details",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    title = "audio.event_detection report"
    return "\n".join([
        "<!doctype html>",
        "<html><head>",
        f"<title>{title}</title>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>Input: {payload['input_path']}</p>",
        "<ul>",
        f"<li>Duration: {summary.get('duration', 0.0)}s</li>",
        f"<li>Events: {summary.get('event_count', 0)}</li>",
        f"<li>Impacts: {summary.get('impact_count', 0)}</li>",
        f"<li>Spectral changes: {summary.get('spectral_change_count', 0)}</li>",
        f"<li>Tonal shifts: {summary.get('tonal_shift_count', 0)}</li>",
        f"<li>High-energy regions: {summary.get('high_energy_count', 0)}</li>",
        f"<li>Low-energy regions: {summary.get('low_energy_count', 0)}</li>",
        "</ul>",
        "</body></html>",
    ])
