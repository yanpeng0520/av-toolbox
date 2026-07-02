"""Shared defaults and parameter metadata for av-toolbox web interfaces."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


DEFAULT_MEDIA_PATH = "data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4"
DEFAULT_OUTPUT_ROOT = "outputs/web_runs"
DEFAULT_OUTPUT_SUBDIR_PREFIX = "run"
DEFAULT_TOOL_NAME = "video.motion"


BASE_FORM_VALUES: dict[str, Any] = {
    "tool_name": DEFAULT_TOOL_NAME,
    "media_path": DEFAULT_MEDIA_PATH,
    "override_defaults": False,
    "device": "",
    "batch_size": "",
    "fp16": False,
    "cache_dir": "",
    "workspace_dir": "",
    "keep_workspace": False,
    "export_overlay": True,
    "export_json": True,
    "export_csv": True,
    "export_report": True,
}


RUN_WORKFLOWS: tuple[dict[str, Any], ...] = (
    {
        "name": "Motion",
        "description": "Frame-to-frame motion intensity.",
        "values": {"tool_name": "video.motion"},
    },
    {
        "name": "Image Quality",
        "description": "Sharpness/blur, luma/exposure, contrast, and lens obstruction.",
        "values": {"tool_name": "video.image_quality"},
    },
    {
        "name": "Blur Exposure",
        "description": "Per-frame blur, luminance, dark-frame, and overexposure checks.",
        "values": {"tool_name": "video.blur_exposure"},
    },
    {
        "name": "Shot Boundaries",
        "description": "TransNetV2 cut and scene segment detection.",
        "values": {"tool_name": "video.cut_detection", "backend": "transnetv2"},
    },
    {
        "name": "Obstruction",
        "description": "Bright low-variance obstruction detection.",
        "values": {"tool_name": "video.obstruction"},
    },
    {
        "name": "Optical Flow",
        "description": "Dense optical-flow magnitude and mask overlay.",
        "values": {"tool_name": "video.optical_flow"},
    },
    {
        "name": "Foreground Motion",
        "description": "Foreground-biased optical-flow motion overlay.",
        "values": {"tool_name": "video.foreground_motion"},
    },
    {
        "name": "Camera Shake",
        "description": "Sparse optical-flow translation jitter and shake events.",
        "values": {"tool_name": "video.camera_shake"},
    },
    {
        "name": "Object Detection",
        "description": "YOLO object boxes and class confidences.",
        "values": {"tool_name": "video.object_detection"},
    },
    {
        "name": "Segmentation",
        "description": "YOLO instance masks, boxes, classes, and confidences.",
        "values": {"tool_name": "video.segmentation"},
    },
    {
        "name": "Pose",
        "description": "MediaPipe human pose landmark overlay.",
        "values": {"tool_name": "video.pose"},
    },
    {
        "name": "Shot Type",
        "description": "Frame-level shot-type labels and top-k probabilities.",
        "values": {"tool_name": "video.shot_type"},
    },
    {
        "name": "Action Recognition",
        "description": "SlowFast/PyTorchVideo action labels over sampled windows.",
        "values": {"tool_name": "video.action_recognition"},
    },
    {
        "name": "ST Action",
        "description": "MMAction2 spatio-temporal action recognition when configured.",
        "values": {"tool_name": "video.st_action"},
    },
    {
        "name": "Beats",
        "description": "Beat, downbeat, and onset timeline overlay.",
        "values": {"tool_name": "audio.beat_detection"},
    },
    {
        "name": "Audio Energy",
        "description": "RMS, dB energy, spectral centroid, and silence windows.",
        "values": {"tool_name": "audio.energy"},
    },
    {
        "name": "Audio Events",
        "description": "Impacts, energy regions, spectral changes, and tonal shifts.",
        "values": {"tool_name": "audio.event_detection"},
    },
    {
        "name": "Music Phase",
        "description": "Coarse music phase segmentation.",
        "values": {"tool_name": "audio.music_phase"},
    },
    {
        "name": "Transcription",
        "description": "Whisper speech transcription segments.",
        "values": {"tool_name": "audio.transcription"},
    },
    {
        "name": "AV Sync",
        "description": "Audio event to video motion correspondence.",
        "values": {"tool_name": "av.sync_correspondence"},
    },
    {
        "name": "DenseAV",
        "description": "DenseAV audio-visual attention overlay.",
        "values": {"tool_name": "av.denseav"},
    },
)


TOOL_CATEGORY_ORDER: tuple[str, ...] = ("video", "audio", "av")
TOOL_CATEGORY_LABELS: dict[str, str] = {
    "video": "Video",
    "audio": "Audio",
    "av": "Audio-Visual",
}


def workflow_by_tool_name(tool_name: str | None) -> dict[str, Any] | None:
    if not tool_name:
        return None
    for workflow in RUN_WORKFLOWS:
        if workflow["values"].get("tool_name") == tool_name:
            return workflow
    return None


def tool_display_name(tool_name: str) -> str:
    workflow = workflow_by_tool_name(tool_name)
    return str(workflow["name"]) if workflow else tool_name


def tool_description(tool_name: str, fallback: str = "") -> str:
    workflow = workflow_by_tool_name(tool_name)
    if workflow:
        return str(workflow.get("description") or fallback)
    return fallback


def category_label(category: str) -> str:
    return TOOL_CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def category_from_label(label: str) -> str:
    for category, category_label_value in TOOL_CATEGORY_LABELS.items():
        if category_label_value == label:
            return category
    return label.lower().replace("-", "_").replace(" ", "_")


def tool_category(tool_name: str, tools: list[dict[str, str]]) -> str:
    for tool in tools:
        if tool.get("name") == tool_name:
            return str(tool.get("category") or "")
    return tool_name.split(".", 1)[0] if "." in tool_name else ""


def tool_type_labels(tools: list[dict[str, str]]) -> list[str]:
    categories = {str(tool.get("category") or "") for tool in tools}
    ordered = [category for category in TOOL_CATEGORY_ORDER if category in categories]
    ordered.extend(sorted(category for category in categories if category and category not in TOOL_CATEGORY_ORDER))
    return [category_label(category) for category in ordered]


def default_tool_type_label(tool_name: str, tools: list[dict[str, str]]) -> str:
    category = tool_category(tool_name, tools)
    labels = tool_type_labels(tools)
    label = category_label(category)
    return label if label in labels else labels[0]


def tool_names_for_category(tools: list[dict[str, str]], category: str) -> list[str]:
    available = {
        str(tool.get("name"))
        for tool in tools
        if str(tool.get("category") or "") == category
    }
    ordered = [
        str(workflow["values"]["tool_name"])
        for workflow in RUN_WORKFLOWS
        if str(workflow["values"].get("tool_name")) in available
    ]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def tool_name_by_display_name(display_name: str, tool_names: list[str]) -> str:
    for tool_name in tool_names:
        if tool_display_name(tool_name) == display_name:
            return tool_name
    if display_name in tool_names:
        return display_name
    return tool_names[0]


PARAMETER_SPECS: dict[str, dict[str, Any]] = {
    "sample_fps": {"label": "Sample FPS", "kind": "float", "step": "0.5"},
    "max_seconds": {"label": "Max Seconds", "kind": "float", "step": "1"},
    "sample_rate": {"label": "Sample Rate", "kind": "int", "step": "1000"},
    "hop_length": {"label": "Hop Length", "kind": "int", "step": "128"},
    "window_sec": {"label": "Window Seconds", "kind": "float", "step": "1"},
    "overlay_fps": {"label": "Overlay FPS", "kind": "float", "step": "1"},
    "overlay_width": {"label": "Overlay Width", "kind": "int", "step": "64"},
    "overlay_height": {"label": "Overlay Height", "kind": "int", "step": "64"},
    "threshold": {"label": "Threshold", "kind": "float", "step": "0.01"},
    "active_pct_threshold": {"label": "Active % Threshold", "kind": "float", "step": "0.5"},
    "downscale_width": {"label": "Downscale Width", "kind": "int", "step": "32"},
    "blur_threshold": {"label": "Blur Threshold", "kind": "float", "step": "1"},
    "dark_threshold": {"label": "Dark Threshold", "kind": "float", "step": "1"},
    "super_dark_threshold": {"label": "Super Dark Threshold", "kind": "float", "step": "1"},
    "overexposed_threshold": {"label": "Overexposed Threshold", "kind": "float", "step": "1"},
    "min_scene_seconds": {"label": "Min Scene Seconds", "kind": "float", "step": "0.1"},
    "backend": {
        "label": "Cut Backend",
        "kind": "choice",
        "choices": ["transnetv2", "scenedetect", "lightweight", "auto"],
    },
    "impact_delta": {"label": "Impact Delta", "kind": "float", "step": "0.01"},
    "spectral_delta": {"label": "Spectral Delta", "kind": "float", "step": "0.01"},
    "tonal_delta": {"label": "Tonal Delta", "kind": "float", "step": "0.01"},
    "silence_threshold": {"label": "Silence Threshold", "kind": "float", "step": "0.01"},
    "high_energy_quantile": {"label": "High Energy Quantile", "kind": "float", "step": "0.01"},
    "min_region_seconds": {"label": "Min Region Seconds", "kind": "float", "step": "0.05"},
    "min_phase_seconds": {"label": "Min Phase Seconds", "kind": "float", "step": "1"},
    "phrase_bars": {"label": "Phrase Bars", "kind": "int", "step": "1"},
    "motion_quantile": {"label": "Motion Quantile", "kind": "float", "step": "0.01"},
    "min_motion_score": {"label": "Min Motion Score", "kind": "float", "step": "0.01"},
    "sync_window": {"label": "Sync Window", "kind": "float", "step": "0.01"},
    "model_name": {
        "label": "Model",
        "kind": "choice",
        "choices": ["sound_and_language", "sound"],
    },
    "checkpoint": {"label": "Checkpoint", "kind": "text"},
    "audio_sample_rate": {"label": "DenseAV Audio Sample Rate", "kind": "int", "step": "1000"},
    "load_size": {"label": "Load Size", "kind": "int", "step": "32"},
    "plot_size": {"label": "Plot Size", "kind": "int", "step": "32"},
    "expected_sha256": {"label": "Expected SHA256", "kind": "text"},
    "offline": {"label": "Offline", "kind": "bool"},
    "include_sim_matrix": {"label": "Sim Matrix", "kind": "bool"},
    "export_overlay": {"label": "Overlay", "kind": "bool", "export": True},
    "export_json": {"label": "JSON", "kind": "bool", "export": True},
    "export_csv": {"label": "CSV", "kind": "bool", "export": True},
    "export_report": {"label": "HTML", "kind": "bool", "export": True},
}


RUNTIME_SPECS: dict[str, dict[str, Any]] = {
    "device": {"label": "Device", "kind": "text"},
    "batch_size": {"label": "Batch Size", "kind": "int", "step": "1"},
    "fp16": {"label": "FP16", "kind": "bool"},
    "cache_dir": {"label": "Cache Dir", "kind": "text"},
    "workspace_dir": {"label": "Workspace Dir", "kind": "text"},
    "keep_workspace": {"label": "Keep Workspace", "kind": "bool"},
}


def workflow_names() -> list[str]:
    return [str(workflow["name"]) for workflow in RUN_WORKFLOWS]


def default_workflow_name() -> str:
    return str(RUN_WORKFLOWS[0]["name"])


def workflow_by_name(name: str | None) -> dict[str, Any]:
    if not name:
        return RUN_WORKFLOWS[0]
    for workflow in RUN_WORKFLOWS:
        if workflow["name"] == name:
            return workflow
    return RUN_WORKFLOWS[0]


def default_form_values(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    workflow_name: str | None = None,
) -> dict[str, Any]:
    values = dict(BASE_FORM_VALUES)
    values.update(workflow_by_name(workflow_name)["values"])
    values["output_dir"] = default_output_dir(output_root)
    return values


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_dir(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> str:
    return str(Path(output_root) / default_output_subdir())


def default_output_subdir() -> str:
    return f"{DEFAULT_OUTPUT_SUBDIR_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"


def resolve_existing_input_path(path_value: str | Path | None) -> Path:
    path = Path(path_value or "").expanduser()
    if path.is_absolute() or path.exists():
        return path
    bundled = project_root() / path
    if bundled.exists():
        return bundled
    return path


def ensure_writable_output_dir(
    path_value: str | Path | None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    requested = Path(path_value or default_output_dir(output_root)).expanduser()
    if _is_writable_dir(requested):
        return requested
    return reserve_output_dir(output_root)


def reserve_output_dir(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    root = Path(output_root).expanduser()
    stem = default_output_subdir()
    for index in range(100):
        suffix = "" if index == 0 else f"_{index:02d}"
        candidate = root / f"{stem}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        if _is_writable_dir(candidate):
            return candidate
    raise PermissionError(f"No writable output directory available under {root}")


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".av_toolbox_write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def parameter_defaults_from_callable(func: Callable[..., Any]) -> dict[str, Any]:
    """Return UI-ready defaults for supported parameters on a tool callable."""
    defaults: dict[str, Any] = {}
    signature = inspect.signature(func)
    for name, parameter in signature.parameters.items():
        if name not in PARAMETER_SPECS:
            continue
        if parameter.default is inspect.Signature.empty:
            continue
        defaults[name] = value_to_ui(parameter.default)
    return defaults


def value_to_ui(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    return str(value)


def parse_field_value(name: str, value: Any) -> Any:
    spec = PARAMETER_SPECS.get(name) or RUNTIME_SPECS.get(name)
    if spec is None:
        return value
    kind = spec["kind"]
    if kind == "bool":
        return bool(value)
    if value in (None, ""):
        return None
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    return str(value)


# Compatibility aliases for local callers that still use the previous names.
preset_names = workflow_names
default_preset_name = default_workflow_name
preset_by_name = workflow_by_name
