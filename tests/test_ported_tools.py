from __future__ import annotations

import inspect
import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

import av_toolbox


NEW_TOOL_NAMES = {
    "audio.energy",
    "audio.transcription",
    "video.action_recognition",
    "video.camera_shake",
    "video.cut_detection",
    "video.foreground_motion",
    "video.image_quality",
    "video.object_detection",
    "video.optical_flow",
    "video.pose",
    "video.segmentation",
    "video.shot_type",
}


def _write_audio(path: Path, *, duration: float = 1.0, sr: int = 22050) -> Path:
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    y = 0.55 * np.sin(2 * np.pi * 220.0 * t)
    pcm = np.clip(y * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return path


def _write_motion_video(path: Path, *, duration: float = 1.2, fps: float = 10.0) -> Path:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    if not writer.isOpened():
        pytest.skip(f"OpenCV cannot write {path}")
    try:
        for idx in range(int(duration * fps)):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[:] = (20, 22, 24)
            offset = int((idx % 6) * 8)
            cv2.rectangle(frame, (20 + offset, 35), (70 + offset, 85), (220, 80, 40), -1)
            cv2.circle(frame, (120 - offset // 2, 35 + offset), 12, (40, 180, 220), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _write_uniform_video(path: Path, *, duration: float = 0.8, fps: float = 5.0) -> Path:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (80, 60))
    if not writer.isOpened():
        pytest.skip(f"OpenCV cannot write {path}")
    try:
        for _ in range(int(duration * fps)):
            writer.write(np.full((60, 80, 3), 90, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_ported_tools_are_registered() -> None:
    names = {tool["name"] for tool in av_toolbox.list_tools()}
    assert NEW_TOOL_NAMES <= names


def test_audio_energy_creates_declared_artifacts(tmp_path) -> None:
    audio = _write_audio(tmp_path / "tone.wav")

    result = av_toolbox.run_tool(
        "audio.energy",
        input_path=audio,
        output_dir=tmp_path / "out",
        sample_rate=22050,
        max_seconds=0.8,
        export_overlay=False,
        device="cpu",
    )

    assert result.timeline_json is not None
    assert result.csv_path is not None
    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "audio.energy"
    assert payload["summary"]["sample_count"] > 0
    assert Path(result.csv_path).exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for audio-energy overlay muxing")
def test_audio_energy_creates_overlay(tmp_path) -> None:
    audio = _write_audio(tmp_path / "tone.wav")

    result = av_toolbox.run_tool(
        "audio.energy",
        input_path=audio,
        output_dir=tmp_path / "out",
        sample_rate=22050,
        max_seconds=0.8,
        overlay_fps=5.0,
        overlay_width=480,
        overlay_height=320,
        device="cpu",
    )

    assert result.overlay_path is not None
    assert Path(result.overlay_path).exists()


def test_video_image_quality_detects_uniform_bright_frames(tmp_path) -> None:
    video = _write_uniform_video(tmp_path / "uniform.mp4")

    result = av_toolbox.run_tool(
        "video.image_quality",
        input_path=video,
        output_dir=tmp_path / "out",
        sample_fps=5,
        max_seconds=0.5,
        device="cpu",
    )

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "video.image_quality"
    assert payload["summary"]["obstructed_count"] > 0


def test_video_optical_flow_creates_declared_artifacts(tmp_path) -> None:
    video = _write_motion_video(tmp_path / "motion.mp4")

    result = av_toolbox.run_tool(
        "video.optical_flow",
        input_path=video,
        output_dir=tmp_path / "out",
        sample_fps=5,
        max_seconds=1.0,
        device="cpu",
    )

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "video.optical_flow"
    assert payload["summary"]["sample_count"] > 0
    assert Path(result.csv_path).exists()


def test_video_foreground_motion_runs_without_model_mask(tmp_path) -> None:
    video = _write_motion_video(tmp_path / "foreground.mp4")

    result = av_toolbox.run_tool(
        "video.foreground_motion",
        input_path=video,
        output_dir=tmp_path / "out",
        sample_fps=5,
        max_seconds=1.0,
        mask_mode="none",
        device="cpu",
    )

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "video.foreground_motion"
    assert payload["summary"]["mask_mode"] == "none"
    assert payload["summary"]["sample_count"] > 0
    assert result.overlay_path is not None
    assert Path(result.overlay_path).exists()


def test_video_camera_shake_creates_declared_artifacts(tmp_path) -> None:
    video = _write_motion_video(tmp_path / "shake.mp4")

    result = av_toolbox.run_tool(
        "video.camera_shake",
        input_path=video,
        output_dir=tmp_path / "out",
        sample_fps=10,
        max_seconds=1.0,
        device="cpu",
    )

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "video.camera_shake"
    assert payload["summary"]["sample_count"] > 0
    assert result.overlay_path is not None
    assert Path(result.overlay_path).exists()


def test_video_cut_detection_defaults_to_transnetv2() -> None:
    signature = inspect.signature(av_toolbox.get_tool("video.cut_detection")._run)

    assert signature.parameters["backend"].default == "transnetv2"


def test_video_cut_detection_lightweight_backend_creates_artifacts(tmp_path) -> None:
    video = _write_motion_video(tmp_path / "cuts.mp4")

    result = av_toolbox.run_tool(
        "video.cut_detection",
        input_path=video,
        output_dir=tmp_path / "out",
        max_seconds=1.0,
        backend="lightweight",
        threshold=0.2,
        device="cpu",
    )

    payload = json.loads(Path(result.timeline_json).read_text())
    assert payload["tool_name"] == "video.cut_detection"
    assert payload["summary"]["backend"] == "lightweight"
    assert "segments" in payload
    assert result.overlay_path is not None
    assert Path(result.overlay_path).exists()
