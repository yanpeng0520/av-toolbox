"""Shared guardrails for the public av-toolbox demo UI."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from av_toolbox.ui_defaults import RUN_WORKFLOWS, workflow_by_name


DEFAULT_PUBLIC_MAX_SECONDS = 20.0
DEFAULT_PUBLIC_MAX_UPLOAD_MB = 10
DEFAULT_LOCAL_PAGE_TITLE = "AV Toolbox Demo"
DEFAULT_PUBLIC_PAGE_TITLE = "AV Toolbox Demo"
PUBLIC_OUTPUT_SUBDIR = "public_runs"
PUBLIC_UPLOAD_SUBDIR = "public_uploads"
PUBLIC_SAMPLE_SUBDIR = "public_sample"
PUBLIC_SAMPLE_STEM = "av_toolbox_public_sample"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
PUBLIC_BASE_WORKFLOW_NAMES = (
    "Motion",
    "Image Quality",
    "Blur Exposure",
    "Shot Boundaries",
    "Obstruction",
    "Optical Flow",
    "Foreground Motion",
    "Camera Shake",
    "Object Detection",
    "Segmentation",
    "Pose",
    "Shot Type",
    "Action Recognition",
    "Beats",
    "Audio Energy",
    "Audio Events",
    "Music Phase",
    "Transcription",
)
PUBLIC_DENSEAV_WORKFLOW_NAMES = ("DenseAV",)


def env_flag(name: str, default: bool = False) -> bool:
    """Read a common true/false environment flag."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_text(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def public_workflow_names(*, enable_denseav: bool = False) -> list[str]:
    available = {str(workflow["name"]) for workflow in RUN_WORKFLOWS}
    names = list(PUBLIC_BASE_WORKFLOW_NAMES)
    if enable_denseav:
        names.extend(PUBLIC_DENSEAV_WORKFLOW_NAMES)
    return [name for name in names if name in available]


def default_public_workflow_name(*, enable_denseav: bool = False) -> str:
    return public_workflow_names(enable_denseav=enable_denseav)[0]


def public_tool_name(workflow_name: str | None, *, enable_denseav: bool = False) -> str:
    workflow = workflow_by_name(workflow_name or default_public_workflow_name(enable_denseav=enable_denseav))
    name = str(workflow["name"])
    if name not in public_workflow_names(enable_denseav=enable_denseav):
        raise ValueError(f"Workflow is not available in public demo mode: {name}")
    return str(workflow["values"]["tool_name"])


def public_run_kwargs(
    *,
    max_seconds: float = DEFAULT_PUBLIC_MAX_SECONDS,
    export_overlay: bool = True,
) -> dict[str, Any]:
    """Return bounded runtime options for public uploads."""
    return {
        "max_seconds": max(0.1, float(max_seconds)),
        "export_json": True,
        "export_csv": True,
        "export_report": True,
        "export_overlay": bool(export_overlay),
    }


def public_upload_bytes(max_upload_mb: int = DEFAULT_PUBLIC_MAX_UPLOAD_MB) -> int:
    return max(1, int(max_upload_mb)) * 1024 * 1024


def public_upload_dir(output_root: str | Path) -> Path:
    return Path(output_root).expanduser() / PUBLIC_UPLOAD_SUBDIR


def is_lfs_pointer_file(path: str | Path) -> bool:
    try:
        with Path(path).open("rb") as handle:
            return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def public_sample_media_path(output_root: str | Path) -> Path:
    sample_dir = Path(output_root).expanduser() / PUBLIC_SAMPLE_SUBDIR
    sample_path = sample_dir / f"{PUBLIC_SAMPLE_STEM}.mp4"
    if sample_path.exists():
        return sample_path
    try:
        from av_toolbox.demo_media import generate_synthetic_hiphop

        payload = generate_synthetic_hiphop(
            output_dir=sample_dir,
            duration=4.0,
            sample_rate=22050,
            stem=PUBLIC_SAMPLE_STEM,
            fps=12.0,
            width=640,
            height=360,
        )
    except Exception as exc:
        raise ValueError(
            "Sample media is not available. Upload a file or install the demo media dependencies."
        ) from exc
    return Path(payload["mp4_path"])


def public_run_output_dir(output_root: str | Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    return Path(output_root).expanduser() / PUBLIC_OUTPUT_SUBDIR / f"{stamp}-{run_id}"


def public_mode_from_env(default: bool = False) -> bool:
    return env_flag("AV_TOOLBOX_PUBLIC_DEMO", default)


def public_enable_denseav_from_env(default: bool = False) -> bool:
    return env_flag("AV_TOOLBOX_PUBLIC_ENABLE_DENSEAV", default)


def public_max_seconds_from_env(default: float = DEFAULT_PUBLIC_MAX_SECONDS) -> float:
    return env_float("AV_TOOLBOX_PUBLIC_MAX_SECONDS", default)


def public_max_upload_mb_from_env(default: int = DEFAULT_PUBLIC_MAX_UPLOAD_MB) -> int:
    return env_int("AV_TOOLBOX_PUBLIC_MAX_UPLOAD_MB", default)


def page_title_from_env(default: str) -> str:
    return env_text("AV_TOOLBOX_PAGE_TITLE", default)
