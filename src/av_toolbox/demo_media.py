"""Synthetic demo media generation for av-toolbox."""

from __future__ import annotations

import json
import math
import subprocess
import uuid
from pathlib import Path
from typing import Any


def generate_synthetic_hiphop(
    *,
    output_dir: str | Path = "data_segments",
    duration: float = 60.0,
    sample_rate: int = 44100,
    stem: str = "synthetic_hiphop_60s",
    fps: float = 24.0,
    width: int = 1280,
    height: int = 720,
) -> dict[str, Any]:
    """Generate a varied synthetic hip-hop WAV and an MP4 visualizer."""
    np, sf, cv2 = _imports()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    wav_path = output / f"{stem}.wav"
    mp4_path = output / f"{stem}.mp4"
    meta_path = output / f"{stem}_meta.json"

    y, sections = _synthesize_track(np, duration=duration, sr=sample_rate)
    sf.write(str(wav_path), y, sample_rate, subtype="PCM_16")
    _render_demo_video(
        cv2=cv2,
        np=np,
        y=y,
        sr=sample_rate,
        wav_path=wav_path,
        mp4_path=mp4_path,
        sections=sections,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        workspace=output,
    )
    payload = {
        "wav_path": str(wav_path),
        "mp4_path": str(mp4_path),
        "meta_path": str(meta_path),
        "duration": float(duration),
        "sample_rate": int(sample_rate),
        "bpm": 92,
        "sections": sections,
        "note": "MP4 audio is AAC muxed from the generated WAV for broad player compatibility.",
    }
    meta_path.write_text(json.dumps(payload, indent=2))
    return payload


def _imports():
    try:
        import cv2
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "Demo media generation requires NumPy, soundfile, and OpenCV. Install with: pip install -e '.[all]'"
        ) from exc
    return np, sf, cv2


def _synthesize_track(np: Any, *, duration: float, sr: int) -> tuple[Any, list[dict[str, Any]]]:
    n = int(round(duration * sr))
    y = np.zeros(n, dtype=np.float32)
    bpm = 92.0
    beat = 60.0 / bpm
    bar = beat * 4.0
    sections = [
        {"label": "intro", "start": 0.0, "end": min(8.0, duration), "energy": 0.45},
        {"label": "verse", "start": 8.0, "end": min(24.0, duration), "energy": 0.75},
        {"label": "hook", "start": 24.0, "end": min(40.0, duration), "energy": 1.0},
        {"label": "breakdown", "start": 40.0, "end": min(48.0, duration), "energy": 0.52},
        {"label": "hook", "start": 48.0, "end": duration, "energy": 0.95},
    ]
    sections = [s for s in sections if s["start"] < duration and s["end"] > s["start"]]

    rng = np.random.default_rng(10)
    total_bars = int(math.ceil(duration / bar))
    for bar_idx in range(total_bars):
        bar_start = bar_idx * bar
        section = _section_at(sections, bar_start)
        energy = float(section["energy"])
        label = str(section["label"])
        kick_pattern = [0.0, 1.5, 2.75] if label != "breakdown" else [0.0, 2.0]
        if label == "hook":
            kick_pattern += [3.5]
        for pos in kick_pattern:
            _add_kick(np, y, sr, bar_start + pos * beat, amp=0.95 * energy)
            _add_bass(np, y, sr, bar_start + pos * beat, note_hz=_bass_note(bar_idx, pos), amp=0.42 * energy)
        for pos in ([1.0, 3.0] if label != "intro" else [3.0]):
            _add_snare(np, y, sr, bar_start + pos * beat, rng, amp=0.55 * energy)
        hat_step = beat / (4.0 if label == "hook" else 2.0)
        t = bar_start
        while t < bar_start + bar and t < duration:
            swing = 0.035 if int(round((t - bar_start) / hat_step)) % 2 else 0.0
            _add_hat(np, y, sr, t + swing, rng, amp=0.18 * energy)
            t += hat_step
        if label in ("verse", "hook"):
            _add_chord(np, y, sr, bar_start, duration=bar * 0.92, root_hz=_chord_root(bar_idx), amp=0.16 * energy)
        if label == "hook":
            for pos in [0.25, 0.75, 2.25, 2.75, 3.25]:
                _add_pluck(np, y, sr, bar_start + pos * beat, freq=_melody_note(bar_idx, pos), amp=0.28)
        if label == "breakdown":
            _add_noise_riser(np, y, sr, bar_start + 2.0 * beat, rng, amp=0.12)

    fade_len = min(n, int(sr * 4.0))
    if fade_len > 0:
        y[-fade_len:] *= np.linspace(1.0, 0.15, fade_len, dtype=np.float32)
    y += 0.012 * rng.standard_normal(n).astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 1e-8:
        y = 0.92 * y / peak
    return y.astype(np.float32), sections


def _section_at(sections: list[dict[str, Any]], t: float) -> dict[str, Any]:
    for section in sections:
        if float(section["start"]) <= t < float(section["end"]):
            return section
    return sections[-1]


def _slice(np: Any, y: Any, sr: int, start: float, dur: float) -> tuple[int, Any]:
    i0 = max(0, int(round(start * sr)))
    i1 = min(y.size, i0 + max(1, int(round(dur * sr))))
    return i0, np.arange(i1 - i0, dtype=np.float32) / sr


def _add_kick(np: Any, y: Any, sr: int, start: float, amp: float) -> None:
    i0, t = _slice(np, y, sr, start, 0.42)
    if t.size == 0:
        return
    freq = 42 + 58 * np.exp(-t * 28)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-t * 9.0)
    click = np.exp(-t * 120.0) * np.sin(2 * np.pi * 1600 * t)
    y[i0:i0 + t.size] += amp * (np.sin(phase) * env + 0.16 * click)


def _add_snare(np: Any, y: Any, sr: int, start: float, rng: Any, amp: float) -> None:
    i0, t = _slice(np, y, sr, start, 0.26)
    if t.size == 0:
        return
    noise = rng.standard_normal(t.size).astype(np.float32)
    noise = noise - np.convolve(noise, np.ones(48) / 48, mode="same")
    tone = np.sin(2 * np.pi * 185 * t)
    env = np.exp(-t * 18.0)
    y[i0:i0 + t.size] += amp * (0.78 * noise + 0.22 * tone) * env


def _add_hat(np: Any, y: Any, sr: int, start: float, rng: Any, amp: float) -> None:
    i0, t = _slice(np, y, sr, start, 0.075)
    if t.size == 0:
        return
    noise = rng.standard_normal(t.size).astype(np.float32)
    noise = np.diff(noise, prepend=noise[0])
    env = np.exp(-t * 55.0)
    y[i0:i0 + t.size] += amp * noise * env


def _add_bass(np: Any, y: Any, sr: int, start: float, note_hz: float, amp: float) -> None:
    i0, t = _slice(np, y, sr, start, 0.7)
    if t.size == 0:
        return
    env = np.exp(-t * 2.2)
    tone = np.sin(2 * np.pi * note_hz * t) + 0.35 * np.sin(2 * np.pi * note_hz * 2 * t)
    y[i0:i0 + t.size] += amp * tone * env


def _add_chord(np: Any, y: Any, sr: int, start: float, duration: float, root_hz: float, amp: float) -> None:
    i0, t = _slice(np, y, sr, start, duration)
    if t.size == 0:
        return
    freqs = [root_hz, root_hz * 2 ** (3 / 12), root_hz * 2 ** (7 / 12)]
    chord = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    env = np.minimum(1.0, t / 0.08) * np.exp(-t * 0.16)
    y[i0:i0 + t.size] += amp * chord * env


def _add_pluck(np: Any, y: Any, sr: int, start: float, freq: float, amp: float) -> None:
    i0, t = _slice(np, y, sr, start, 0.34)
    if t.size == 0:
        return
    env = np.exp(-t * 9.5)
    tone = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2.01 * t)
    y[i0:i0 + t.size] += amp * tone * env


def _add_noise_riser(np: Any, y: Any, sr: int, start: float, rng: Any, amp: float) -> None:
    i0, t = _slice(np, y, sr, start, 1.6)
    if t.size == 0:
        return
    noise = rng.standard_normal(t.size).astype(np.float32)
    env = np.linspace(0.0, 1.0, t.size, dtype=np.float32) ** 1.5
    y[i0:i0 + t.size] += amp * np.diff(noise, prepend=noise[0]) * env


def _bass_note(bar_idx: int, pos: float) -> float:
    notes = [46.25, 51.91, 41.20, 55.0]
    return notes[(bar_idx + int(pos)) % len(notes)]


def _chord_root(bar_idx: int) -> float:
    roots = [138.59, 155.56, 123.47, 164.81]
    return roots[bar_idx % len(roots)]


def _melody_note(bar_idx: int, pos: float) -> float:
    notes = [277.18, 311.13, 369.99, 415.30, 466.16]
    return notes[(bar_idx + int(pos * 4)) % len(notes)]


def _render_demo_video(
    *,
    cv2: Any,
    np: Any,
    y: Any,
    sr: int,
    wav_path: Path,
    mp4_path: Path,
    sections: list[dict[str, Any]],
    duration: float,
    fps: float,
    width: int,
    height: int,
    workspace: Path,
) -> None:
    width = width if width % 2 == 0 else width + 1
    height = height if height % 2 == 0 else height + 1
    tmp_path = workspace / f"{mp4_path.stem}_{uuid.uuid4().hex}_video_only.mp4"
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {tmp_path}")
    n_frames = max(1, int(math.ceil(duration * fps)))
    peak = max(float(np.max(np.abs(y))), 1e-8)
    try:
        for frame_idx in range(n_frames):
            t = min(frame_idx / fps, duration)
            section = _section_at(sections, t)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            base = _section_color(section["label"])
            canvas[:] = (18 + base[0] // 8, 18 + base[1] // 8, 22 + base[2] // 8)
            cv2.putText(canvas, "av-toolbox synthetic hip-hop", (48, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (236, 240, 232), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{section['label']}  |  {t:05.2f}s / {duration:.1f}s", (50, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (185, 205, 198), 1, cv2.LINE_AA)
            _draw_section_strip(cv2, canvas, sections, t, duration, width, height)
            _draw_demo_wave(cv2, np, canvas, y, sr, peak, t, duration, width, height, base)
            _draw_pulse_grid(cv2, np, canvas, t, width, height, base)
            writer.write(canvas)
    finally:
        writer.release()
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(tmp_path),
        "-i", str(wav_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(mp4_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def _section_color(label: str) -> tuple[int, int, int]:
    return {
        "intro": (88, 86, 56),
        "verse": (52, 112, 82),
        "hook": (82, 84, 164),
        "breakdown": (122, 72, 132),
    }.get(label, (72, 88, 102))


def _draw_section_strip(cv2: Any, canvas: Any, sections: list[dict[str, Any]], t: float, duration: float, width: int, height: int) -> None:
    x0, x1 = 50, width - 50
    y0, y1 = height - 84, height - 54
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (44, 48, 52), cv2.FILLED)
    for section in sections:
        a = int(round(x0 + section["start"] / duration * (x1 - x0)))
        b = int(round(x0 + section["end"] / duration * (x1 - x0)))
        cv2.rectangle(canvas, (a, y0), (b, y1), _section_color(section["label"]), cv2.FILLED)
        if b - a > 80:
            cv2.putText(canvas, section["label"], (a + 8, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (238, 240, 232), 1, cv2.LINE_AA)
    px = int(round(x0 + t / duration * (x1 - x0)))
    cv2.line(canvas, (px, y0 - 8), (px, y1 + 8), (55, 55, 240), 3, cv2.LINE_AA)


def _draw_demo_wave(cv2: Any, np: Any, canvas: Any, y: Any, sr: int, peak: float, t: float, duration: float, width: int, height: int, color: tuple[int, int, int]) -> None:
    x0, x1 = 70, width - 70
    y0, y1 = 210, height - 150
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (28, 31, 35), cv2.FILLED)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (94, 104, 110), 1)
    window = 7.0
    start = max(0.0, min(duration - window, t - window * 0.35))
    end = start + window
    s0 = int(start * sr)
    s1 = min(y.size, int(end * sr))
    seg = y[s0:s1]
    plot_w = x1 - x0
    if seg.size:
        edges = np.linspace(0, seg.size, plot_w + 1).astype(int)
        center = (y0 + y1) // 2
        half = (y1 - y0) // 2 - 8
        for idx in range(plot_w):
            part = seg[edges[idx]: max(edges[idx + 1], edges[idx] + 1)]
            a = int(center - float(part.max()) / peak * half)
            b = int(center - float(part.min()) / peak * half)
            cv2.line(canvas, (x0 + idx, a), (x0 + idx, b), (min(255, color[0] + 120), min(255, color[1] + 110), min(255, color[2] + 90)), 1)
        ph = int(round(x0 + (t - start) / window * plot_w))
        cv2.line(canvas, (ph, y0 - 4), (ph, y1 + 4), (55, 55, 240), 2, cv2.LINE_AA)


def _draw_pulse_grid(cv2: Any, np: Any, canvas: Any, t: float, width: int, height: int, color: tuple[int, int, int]) -> None:
    phase = (math.sin(t * math.tau * 0.5) + 1.0) * 0.5
    for idx in range(12):
        x = int(70 + idx * (width - 140) / 11)
        h = int(34 + 95 * abs(math.sin(t * 2.1 + idx * 0.6)))
        cv2.rectangle(canvas, (x - 8, 158 - h // 2), (x + 8, 158 + h // 2), (color[0], color[1], min(255, color[2] + int(50 * phase))), cv2.FILLED)
