"""Beat and downbeat detection tool."""

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

BEAT_COLOR = (90, 170, 60)
DOWNBEAT_COLOR = (40, 130, 250)
ONSET_COLOR = (180, 120, 60)


class BeatDetectionTool(BaseTool):
    """Detect beats/downbeats and render a rolling verification overlay."""

    name = "audio.beat_detection"
    category = "audio"
    description = "Detect beats and heuristic downbeats from audio."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_rate: int | None = 22050,
        hop_length: int | None = 512,
        max_seconds: float | None = None,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        export_overlay: bool = True,
        overlay_fps: float | None = 15.0,
        window_sec: float | None = 8.0,
        overlay_width: int = 1280,
        overlay_height: int = 540,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("audio.beat_detection requires input_path")

        sr_target = sample_rate or 22050
        hop = hop_length or 512
        y, sr, metadata = load_audio_samples(input_path, sample_rate=sr_target, max_seconds=max_seconds)
        analysis = analyze_beats(y, sr, hop_length=hop)
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
            "detector": analysis["detector"],
            "overlay_fps": overlay_fps,
            "window_sec": window_sec,
        }
        artifacts.config_path.write_text(yaml.safe_dump(config, sort_keys=True))

        rows = _beat_rows(analysis)
        events = _beat_events(rows)
        summary = {
            "duration": round(metadata.duration, 6),
            "tempo_bpm": round(float(analysis["tempo_bpm"]), 4),
            "beat_count": int(len(analysis["beats"])),
            "downbeat_count": int(len(analysis["downbeats"])),
            "onset_count": int(len(analysis["onsets"])),
            "detector": analysis["detector"],
        }
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": metadata.to_dict(),
            "summary": summary,
            "events": events,
            "beats": [round(float(t), 6) for t in analysis["beats"]],
            "downbeats": [round(float(t), 6) for t in analysis["downbeats"]],
            "onsets": [round(float(t), 6) for t in analysis["onsets"]],
            "samples": rows,
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
                    ("DOWNBEATS", analysis["downbeats"], DOWNBEAT_COLOR),
                    ("BEATS", analysis["beats"], BEAT_COLOR),
                    ("ONSETS", analysis["onsets"], ONSET_COLOR),
                ],
                title="audio.beat_detection",
                workspace=context.workspace,
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
                f"beats={len(analysis['beats'])}",
                f"downbeats={len(analysis['downbeats'])}",
                f"onsets={len(analysis['onsets'])}",
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


def analyze_beats(y: Any, sr: int, *, hop_length: int = 512) -> dict[str, Any]:
    """Return beat, downbeat, onset, and tempo arrays using librosa fallbacks."""
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "audio.beat_detection requires librosa and NumPy. Install with: pip install -e '.[audio]'"
        ) from exc

    y = np.asarray(y, dtype=np.float32)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    detector = "librosa.beat_track"
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            units="frames",
        )
    except Exception:
        tempo = 0.0
        beat_frames = np.empty(0, dtype=int)
        detector = "librosa.onset_fallback"

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    if beat_times.size < 2:
        norm = onset_env / max(float(onset_env.max()) if onset_env.size else 0.0, 1e-8)
        wait = max(1, int(round(0.28 * sr / hop_length)))
        peaks = librosa.util.peak_pick(
            norm,
            pre_max=wait,
            post_max=wait,
            pre_avg=wait * 2,
            post_avg=wait * 2,
            delta=0.25,
            wait=wait,
        )
        beat_frames = peaks
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        detector = "librosa.onset_fallback"
        if beat_times.size >= 2:
            intervals = np.diff(beat_times)
            median_interval = float(np.median(intervals))
            tempo = 60.0 / median_interval if median_interval > 0 else 0.0

    tempo_bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    downbeats = beat_times[::4]
    onset_peaks = _onset_peaks(onset_env, sr, hop_length, librosa, np)
    strengths = _event_strengths(beat_times, onset_env, sr, hop_length, librosa, np)
    return {
        "detector": detector,
        "tempo_bpm": tempo_bpm,
        "beats": clean_times(beat_times, np),
        "downbeats": clean_times(downbeats, np),
        "onsets": clean_times(onset_peaks, np),
        "beat_strengths": strengths,
        "onset_envelope": onset_env,
    }


def clean_times(times: Any, np: Any | None = None) -> Any:
    if np is None:
        import numpy as np
    vals = np.asarray(times, dtype=float)
    vals = vals[np.isfinite(vals)]
    return np.sort(np.unique(vals))


def _onset_peaks(onset_env: Any, sr: int, hop_length: int, librosa: Any, np: Any) -> Any:
    if onset_env.size == 0:
        return np.empty(0, dtype=float)
    norm = onset_env / max(float(onset_env.max()), 1e-8)
    wait = max(1, int(round(0.12 * sr / hop_length)))
    peaks = librosa.util.peak_pick(
        norm,
        pre_max=wait,
        post_max=wait,
        pre_avg=wait * 2,
        post_avg=wait * 2,
        delta=0.2,
        wait=wait,
    )
    return librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)


def _event_strengths(times: Any, onset_env: Any, sr: int, hop_length: int, librosa: Any, np: Any) -> Any:
    if len(times) == 0 or onset_env.size == 0:
        return np.empty(0, dtype=float)
    frame_times = librosa.frames_to_time(np.arange(onset_env.size), sr=sr, hop_length=hop_length)
    env = onset_env / max(float(onset_env.max()), 1e-8)
    return np.interp(times, frame_times, env).astype(float)


def _beat_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    beats = analysis["beats"]
    downbeat_set = {round(float(t), 6) for t in analysis["downbeats"]}
    strengths = analysis.get("beat_strengths", [])
    rows: list[dict[str, Any]] = []
    prev_t = None
    for idx, beat_t in enumerate(beats):
        t = float(beat_t)
        interval = None if prev_t is None else t - prev_t
        rows.append({
            "timestamp": round(t, 6),
            "index": idx,
            "event_type": "downbeat" if round(t, 6) in downbeat_set else "beat",
            "beat_interval": round(interval, 6) if interval is not None else "",
            "tempo_bpm": round(float(analysis["tempo_bpm"]), 4),
            "strength": round(float(strengths[idx]), 6) if idx < len(strengths) else 0.0,
            "is_downbeat": round(t, 6) in downbeat_set,
        })
        prev_t = t
    return rows


def _beat_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": row["timestamp"],
            "end": round(float(row["timestamp"]) + 0.05, 6),
            "label": row["event_type"],
            "data": row,
        }
        for row in rows
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "timestamp",
        "index",
        "event_type",
        "beat_interval",
        "tempo_bpm",
        "strength",
        "is_downbeat",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    title = "audio.beat_detection report"
    return "\n".join([
        "<!doctype html>",
        "<html><head>",
        f"<title>{title}</title>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>Input: {payload['input_path']}</p>",
        "<ul>",
        f"<li>Duration: {summary.get('duration', 0.0)}s</li>",
        f"<li>Tempo: {summary.get('tempo_bpm', 0.0)} BPM</li>",
        f"<li>Beats: {summary.get('beat_count', 0)}</li>",
        f"<li>Downbeats: {summary.get('downbeat_count', 0)}</li>",
        f"<li>Onsets: {summary.get('onset_count', 0)}</li>",
        "</ul>",
        "</body></html>",
    ])
