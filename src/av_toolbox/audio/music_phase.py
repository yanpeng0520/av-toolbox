"""Music phase detection tool."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from av_toolbox.audio.beat_detection import BEAT_COLOR, DOWNBEAT_COLOR, analyze_beats, clean_times
from av_toolbox.audio.overlay import render_timeline_overlay
from av_toolbox.core.audio_io import load_audio_samples
from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.outputs import make_artifact_paths
from av_toolbox.core.result import AVResult

PHASE_BOUNDARY_COLOR = (180, 120, 60)


class MusicPhaseTool(BaseTool):
    """Estimate coarse music phases such as intro, verse, hook, and bridge."""

    name = "audio.music_phase"
    category = "audio"
    description = "Estimate coarse music phase segments from audio features."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_rate: int | None = 22050,
        hop_length: int | None = 512,
        max_seconds: float | None = None,
        min_phase_seconds: float = 6.0,
        phrase_bars: int = 4,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        export_overlay: bool = True,
        overlay_fps: float | None = 15.0,
        window_sec: float | None = 10.0,
        overlay_width: int = 1280,
        overlay_height: int = 560,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("audio.music_phase requires input_path")

        sr_target = sample_rate or 22050
        hop = hop_length or 512
        y, sr, metadata = load_audio_samples(input_path, sample_rate=sr_target, max_seconds=max_seconds)
        beats = analyze_beats(y, sr, hop_length=hop)
        features = extract_phase_features(y, sr, hop)
        segments = detect_music_phases(
            features,
            beats,
            duration=metadata.duration,
            min_phase_seconds=min_phase_seconds,
            phrase_bars=phrase_bars,
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
            "min_phase_seconds": min_phase_seconds,
            "phrase_bars": phrase_bars,
            "beat_detector": beats["detector"],
            "overlay_fps": overlay_fps,
            "window_sec": window_sec,
        }
        artifacts.config_path.write_text(yaml.safe_dump(config, sort_keys=True))

        events = [
            {
                "start": row["start"],
                "end": row["end"],
                "label": row["phase_label"],
                "data": row,
            }
            for row in segments
        ]
        phase_boundaries = [float(row["start"]) for row in segments[1:]]
        summary = {
            "duration": round(metadata.duration, 6),
            "phase_count": len(segments),
            "tempo_bpm": round(float(beats["tempo_bpm"]), 4),
            "beat_count": int(len(beats["beats"])),
            "downbeat_count": int(len(beats["downbeats"])),
            "detector": "librosa-feature-phrase-heuristic",
        }
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "segments": segments,
            "beats": [round(float(t), 6) for t in beats["beats"]],
            "downbeats": [round(float(t), 6) for t in beats["downbeats"]],
            "samples": feature_rows(features),
        }

        if export_json:
            artifacts.timeline_json.write_text(json.dumps(timeline_payload, indent=2))
        if export_csv:
            _write_csv(artifacts.csv_path, segments)
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
                    ("PHASES", phase_boundaries, PHASE_BOUNDARY_COLOR),
                    ("DOWNBEATS", beats["downbeats"], DOWNBEAT_COLOR),
                    ("BEATS", beats["beats"], BEAT_COLOR),
                ],
                title="audio.music_phase",
                workspace=context.workspace,
                segments=segments,
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
                window_sec=window_sec or 10.0,
            )

        artifacts.log_path.write_text(
            "\n".join([
                f"tool={self.name}",
                f"input={input_path}",
                f"duration={metadata.duration:.6f}",
                f"phases={len(segments)}",
                f"beats={len(beats['beats'])}",
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
                "segments": len(segments),
            },
        )


def extract_phase_features(y: Any, sr: int, hop_length: int) -> dict[str, Any]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "audio.music_phase requires librosa and NumPy. Install with: pip install -e '.[audio]'"
        ) from exc
    y = np.asarray(y, dtype=np.float32)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    n = min(len(rms), len(onset), len(centroid))
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    return {
        "times": times,
        "rms": rms[:n].astype(float),
        "onset": onset[:n].astype(float),
        "centroid": centroid[:n].astype(float),
    }


def detect_music_phases(
    features: dict[str, Any],
    beats: dict[str, Any],
    *,
    duration: float,
    min_phase_seconds: float = 6.0,
    phrase_bars: int = 4,
) -> list[dict[str, Any]]:
    import numpy as np

    boundaries = _phase_boundaries(features, beats, duration, min_phase_seconds, phrase_bars, np)
    segments: list[dict[str, Any]] = []
    scores: list[float] = []
    stats: list[dict[str, float]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_stats = _segment_stats(features, float(start), float(end), np)
        score = 0.55 * segment_stats["rms_norm"] + 0.3 * segment_stats["onset_norm"] + 0.15 * segment_stats["centroid_norm"]
        scores.append(float(score))
        stats.append(segment_stats)

    if not scores:
        return [{
            "index": 0,
            "start": 0.0,
            "end": round(float(duration), 6),
            "duration": round(float(duration), 6),
            "phase_label": "intro",
            "energy_score": 0.0,
            "mean_rms": 0.0,
            "mean_onset_strength": 0.0,
            "mean_spectral_centroid": 0.0,
        }]

    high = float(np.quantile(scores, 0.72))
    low = float(np.quantile(scores, 0.25))
    for idx, ((start, end), score, segment_stats) in enumerate(zip(zip(boundaries[:-1], boundaries[1:]), scores, stats)):
        label = _phase_label(idx, len(scores), score, high, low, segment_stats)
        segments.append({
            "index": idx,
            "start": round(float(start), 6),
            "end": round(float(end), 6),
            "duration": round(max(0.0, float(end) - float(start)), 6),
            "phase_label": label,
            "energy_score": round(score, 6),
            "mean_rms": round(segment_stats["mean_rms"], 6),
            "mean_onset_strength": round(segment_stats["mean_onset"], 6),
            "mean_spectral_centroid": round(segment_stats["mean_centroid"], 4),
        })
    return segments


def feature_rows(features: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for t, rms, onset, centroid in zip(features["times"], features["rms"], features["onset"], features["centroid"]):
        rows.append({
            "timestamp": round(float(t), 6),
            "rms": round(float(rms), 6),
            "onset_strength": round(float(onset), 6),
            "spectral_centroid": round(float(centroid), 4),
        })
    return rows


def _phase_boundaries(features: dict[str, Any], beats: dict[str, Any], duration: float, min_phase_seconds: float, phrase_bars: int, np: Any) -> Any:
    boundaries = [0.0]
    downbeats = np.asarray(beats.get("downbeats", []), dtype=float)
    if downbeats.size >= phrase_bars + 1:
        for t in downbeats[phrase_bars::phrase_bars]:
            if min_phase_seconds <= t <= duration - min_phase_seconds:
                boundaries.append(float(t))
    if len(boundaries) <= 1:
        times = np.asarray(features["times"], dtype=float)
        rms = _norm(features["rms"], np)
        onset = _norm(features["onset"], np)
        centroid = _norm(features["centroid"], np)
        novelty = np.abs(np.diff(rms, prepend=rms[0])) + 0.5 * _norm(onset, np) + 0.2 * np.abs(np.diff(centroid, prepend=centroid[0]))
        if novelty.size:
            import librosa
            wait = max(1, int(round(min_phase_seconds / max(float(times[1] - times[0]) if times.size > 1 else 0.1, 1e-6))))
            peaks = librosa.util.peak_pick(
                novelty / max(float(novelty.max()), 1e-8),
                pre_max=max(1, wait // 2),
                post_max=max(1, wait // 2),
                pre_avg=wait,
                post_avg=wait,
                delta=0.18,
                wait=wait,
            )
            for t in times[peaks]:
                if min_phase_seconds <= t <= duration - min_phase_seconds:
                    boundaries.append(float(t))
    while len(boundaries) <= 1 and duration > min_phase_seconds * 1.8:
        step = max(min_phase_seconds, min(12.0, duration / 3.0))
        t = step
        while t <= duration - min_phase_seconds:
            boundaries.append(float(t))
            t += step
    boundaries.append(float(duration))
    return clean_times(boundaries, np)


def _segment_stats(features: dict[str, Any], start: float, end: float, np: Any) -> dict[str, float]:
    times = np.asarray(features["times"], dtype=float)
    mask = (times >= start) & (times < end)
    if not mask.any():
        mask = np.ones_like(times, dtype=bool)
    rms = np.asarray(features["rms"], dtype=float)
    onset = np.asarray(features["onset"], dtype=float)
    centroid = np.asarray(features["centroid"], dtype=float)
    return {
        "mean_rms": float(np.mean(rms[mask])) if rms.size else 0.0,
        "mean_onset": float(np.mean(onset[mask])) if onset.size else 0.0,
        "mean_centroid": float(np.mean(centroid[mask])) if centroid.size else 0.0,
        "rms_norm": float(np.mean(_norm(rms, np)[mask])) if rms.size else 0.0,
        "onset_norm": float(np.mean(_norm(onset, np)[mask])) if onset.size else 0.0,
        "centroid_norm": float(np.mean(_norm(centroid, np)[mask])) if centroid.size else 0.0,
    }


def _norm(values: Any, np: Any) -> Any:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return vals
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo <= 1e-8:
        return np.zeros_like(vals)
    return (vals - lo) / (hi - lo)


def _phase_label(idx: int, count: int, score: float, high: float, low: float, stats: dict[str, float]) -> str:
    if idx == 0 and (count > 1 or stats["rms_norm"] < 0.5):
        return "intro"
    if idx == count - 1 and count > 2 and stats["rms_norm"] < 0.35:
        return "outro"
    if score >= high:
        return "hook"
    if score <= low and count > 2:
        return "bridge"
    return "verse"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        "start",
        "end",
        "duration",
        "phase_label",
        "energy_score",
        "mean_rms",
        "mean_onset_strength",
        "mean_spectral_centroid",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    title = "audio.music_phase report"
    items = [
        "<!doctype html>",
        "<html><head>",
        f"<title>{title}</title>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>Input: {payload['input_path']}</p>",
        "<ul>",
        f"<li>Duration: {summary.get('duration', 0.0)}s</li>",
        f"<li>Tempo: {summary.get('tempo_bpm', 0.0)} BPM</li>",
        f"<li>Phases: {summary.get('phase_count', 0)}</li>",
        "</ul>",
        "<ol>",
    ]
    for segment in payload.get("segments", []):
        items.append(
            f"<li>{segment['phase_label']}: {segment['start']}s - {segment['end']}s "
            f"({segment['duration']}s)</li>"
        )
    items.extend(["</ol>", "</body></html>"])
    return "\n".join(items)
