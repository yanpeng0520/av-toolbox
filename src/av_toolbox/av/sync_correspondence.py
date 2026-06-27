"""Lightweight audio-visual sync and correspondence analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from av_toolbox.audio.event_detection import analyze_audio_events
from av_toolbox.av.overlay import render_sync_overlay
from av_toolbox.core.audio_io import load_audio_samples
from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.media_io import iter_sampled_frames, read_video_metadata
from av_toolbox.core.outputs import make_artifact_paths
from av_toolbox.core.result import AVResult


class SyncCorrespondenceTool(BaseTool):
    """Find coarse sync matches between audio impacts and video motion spikes."""

    name = "av.sync_correspondence"
    category = "av"
    description = "Match audio events to nearby visual motion spikes."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        sample_fps: float = 8.0,
        sample_rate: int | None = 22050,
        hop_length: int | None = 512,
        max_seconds: float | None = None,
        downscale_width: int = 512,
        motion_quantile: float = 0.72,
        min_motion_score: float = 0.18,
        sync_window: float = 0.18,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        export_overlay: bool = True,
        overlay_fps: float | None = 15.0,
        window_sec: float | None = 6.0,
        overlay_width: int = 1280,
        overlay_height: int = 720,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("av.sync_correspondence requires input_path")

        sr_target = sample_rate or 22050
        hop = hop_length or 512
        y, sr, audio_metadata = load_audio_samples(input_path, sample_rate=sr_target, max_seconds=max_seconds)
        video_metadata = read_video_metadata(input_path)
        duration = min(audio_metadata.duration, video_metadata.duration)
        if max_seconds is not None:
            duration = min(duration, float(max_seconds))

        audio_analysis = analyze_audio_events(
            y,
            sr,
            hop_length=hop,
            impact_delta=0.24,
            spectral_delta=0.34,
            tonal_delta=0.34,
            min_region_seconds=0.3,
        )
        audio_events = _audio_events(audio_analysis)
        motion_analysis = analyze_video_motion(
            input_path,
            sample_fps=sample_fps,
            max_seconds=duration,
            downscale_width=downscale_width,
            motion_quantile=motion_quantile,
            min_motion_score=min_motion_score,
        )
        sync_matches = match_sync_events(
            audio_events,
            motion_analysis["motion_peaks"],
            sync_window=sync_window,
        )

        artifacts = make_artifact_paths(
            input_path=input_path,
            output_dir=context.output_dir,
            tool_name=self.name,
        )
        config = {
            "tool_name": self.name,
            "sample_fps": sample_fps,
            "sample_rate": sr,
            "hop_length": hop,
            "max_seconds": max_seconds,
            "downscale_width": downscale_width,
            "motion_quantile": motion_quantile,
            "min_motion_score": min_motion_score,
            "sync_window": sync_window,
            "overlay_fps": overlay_fps,
            "window_sec": window_sec,
        }
        artifacts.config_path.write_text(yaml.safe_dump(config, sort_keys=True))

        offsets = [abs(float(row["offset_seconds"])) for row in sync_matches]
        summary = {
            "duration": round(duration, 6),
            "audio_event_count": len(audio_events),
            "motion_sample_count": len(motion_analysis["samples"]),
            "motion_peak_count": len(motion_analysis["motion_peaks"]),
            "sync_match_count": len(sync_matches),
            "mean_abs_offset": round(sum(offsets) / len(offsets), 6) if offsets else None,
            "max_abs_offset": round(max(offsets), 6) if offsets else None,
            "detector": "audio-event-motion-peak-nearest-neighbor",
        }
        events = [
            {
                "start": row["audio_timestamp"],
                "end": row["motion_timestamp"],
                "label": "sync_match",
                "data": row,
            }
            for row in sync_matches
        ]
        timeline_payload = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "media": {
                "audio": audio_metadata.to_dict(),
                "video": video_metadata.to_dict(),
            },
            "summary": summary,
            "events": events,
            "sync_matches": sync_matches,
            "audio_events": audio_events,
            "motion_peaks": motion_analysis["motion_peaks"],
            "motion_samples": motion_analysis["samples"],
        }

        if export_json:
            artifacts.timeline_json.write_text(json.dumps(timeline_payload, indent=2))
        if export_csv:
            _write_csv(artifacts.csv_path, sync_matches)
        if export_report:
            artifacts.report_html.write_text(_html_report(timeline_payload))

        overlay_path = None
        if export_overlay:
            overlay_path = render_sync_overlay(
                video_path=input_path,
                output_path=artifacts.overlay_path,
                y=y,
                sr=sr,
                duration=duration,
                motion_samples=motion_analysis["samples"],
                motion_peaks=motion_analysis["motion_peaks"],
                audio_events=audio_events,
                sync_matches=sync_matches,
                workspace=context.workspace,
                fps=overlay_fps or 15.0,
                width=overlay_width,
                height=overlay_height,
                window_sec=window_sec or 6.0,
            )

        artifacts.log_path.write_text(
            "\n".join([
                f"tool={self.name}",
                f"input={input_path}",
                f"duration={duration:.6f}",
                f"audio_events={len(audio_events)}",
                f"motion_peaks={len(motion_analysis['motion_peaks'])}",
                f"sync_matches={len(sync_matches)}",
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
                "media": {
                    "audio": audio_metadata.to_dict(),
                    "video": video_metadata.to_dict(),
                },
                "summary": summary,
                "events": len(events),
            },
        )


def analyze_video_motion(
    input_path: str | Path,
    *,
    sample_fps: float,
    max_seconds: float,
    downscale_width: int,
    motion_quantile: float,
    min_motion_score: float,
) -> dict[str, Any]:
    cv2, np = _imports()
    rows: list[dict[str, Any]] = []
    prev_gray = None
    for frame_idx, timestamp, frame in iter_sampled_frames(input_path, sample_fps=sample_fps, max_seconds=max_seconds):
        resized = _resize(frame, downscale_width, cv2)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue
        diff = cv2.absdiff(prev_gray, gray)
        _, active = cv2.threshold(diff, 15.0, 255, cv2.THRESH_BINARY)
        active_pct = float(cv2.countNonZero(active)) / float(active.size) * 100.0 if active.size else 0.0
        rows.append({
            "timestamp": round(float(timestamp), 6),
            "frame_idx": int(frame_idx),
            "mean_diff": round(float(np.mean(diff)), 4),
            "active_pct": round(active_pct, 4),
            "motion_score": 0.0,
            "is_motion_peak": False,
        })
        prev_gray = gray

    if not rows:
        return {"samples": rows, "motion_peaks": []}

    mean_vals = np.asarray([float(row["mean_diff"]) for row in rows], dtype=float)
    active_vals = np.asarray([float(row["active_pct"]) for row in rows], dtype=float)
    scores = 0.65 * _norm(mean_vals, np) + 0.35 * _norm(active_vals, np)
    threshold = max(min_motion_score, float(np.quantile(scores, min(max(motion_quantile, 0.0), 1.0))))
    peaks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        score = float(scores[idx])
        row["motion_score"] = round(score, 6)
        prev_score = float(scores[idx - 1]) if idx > 0 else -1.0
        next_score = float(scores[idx + 1]) if idx < len(scores) - 1 else -1.0
        if score >= threshold and score >= prev_score and score >= next_score:
            row["is_motion_peak"] = True
            peaks.append({
                "timestamp": row["timestamp"],
                "frame_idx": row["frame_idx"],
                "motion_score": row["motion_score"],
                "mean_diff": row["mean_diff"],
                "active_pct": row["active_pct"],
            })
    return {"samples": rows, "motion_peaks": peaks}


def match_sync_events(
    audio_events: list[dict[str, Any]],
    motion_peaks: list[dict[str, Any]],
    *,
    sync_window: float,
) -> list[dict[str, Any]]:
    if not audio_events or not motion_peaks:
        return []
    matches = []
    for audio_event in audio_events:
        at = float(audio_event["timestamp"])
        nearest = min(motion_peaks, key=lambda row: abs(float(row["timestamp"]) - at))
        mt = float(nearest["timestamp"])
        offset = mt - at
        if abs(offset) > sync_window:
            continue
        closeness = 1.0 - min(1.0, abs(offset) / max(sync_window, 1e-9))
        audio_score = float(audio_event.get("score", 0.0))
        motion_score = float(nearest.get("motion_score", 0.0))
        score = 0.6 * closeness + 0.2 * audio_score + 0.2 * motion_score
        matches.append({
            "match_id": len(matches),
            "audio_timestamp": round(at, 6),
            "motion_timestamp": round(mt, 6),
            "offset_seconds": round(offset, 6),
            "abs_offset_seconds": round(abs(offset), 6),
            "score": round(score, 6),
            "audio_score": round(audio_score, 6),
            "motion_score": round(motion_score, 6),
            "audio_event_type": audio_event["event_type"],
            "motion_frame_idx": nearest["frame_idx"],
        })
    return matches


def _audio_events(audio_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for label, times_key, scores_key in (
        ("impact", "impact_times", "impact_scores"),
        ("spectral_change", "spectral_change_times", "spectral_change_scores"),
    ):
        times = audio_analysis[times_key]
        scores = audio_analysis[scores_key]
        for idx, t in enumerate(times):
            events.append({
                "timestamp": round(float(t), 6),
                "event_type": label,
                "score": round(float(scores[idx]), 6) if idx < len(scores) else 0.0,
            })
    return sorted(events, key=lambda row: (row["timestamp"], row["event_type"]))


def _imports():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "av.sync_correspondence requires OpenCV and NumPy. Install with: pip install -e '.[video]'"
        ) from exc
    return cv2, np


def _resize(frame: Any, downscale_width: int, cv2: Any) -> Any:
    if downscale_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= downscale_width:
        return frame
    scale = downscale_width / float(width)
    size = (downscale_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


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
        "match_id",
        "audio_timestamp",
        "motion_timestamp",
        "offset_seconds",
        "abs_offset_seconds",
        "score",
        "audio_score",
        "motion_score",
        "audio_event_type",
        "motion_frame_idx",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    title = "av.sync_correspondence report"
    return "\n".join([
        "<!doctype html>",
        "<html><head>",
        f"<title>{title}</title>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>Input: {payload['input_path']}</p>",
        "<ul>",
        f"<li>Duration: {summary.get('duration', 0.0)}s</li>",
        f"<li>Audio events: {summary.get('audio_event_count', 0)}</li>",
        f"<li>Motion peaks: {summary.get('motion_peak_count', 0)}</li>",
        f"<li>Sync matches: {summary.get('sync_match_count', 0)}</li>",
        f"<li>Mean |offset|: {summary.get('mean_abs_offset')}</li>",
        "</ul>",
        "</body></html>",
    ])
