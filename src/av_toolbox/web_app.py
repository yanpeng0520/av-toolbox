"""Streamlit application for local av-toolbox runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import av_toolbox


VIDEO_AUDIO_EXTENSIONS = [
    "mp4",
    "mov",
    "mkv",
    "webm",
    "avi",
    "wav",
    "mp3",
    "flac",
    "m4a",
    "aac",
    "ogg",
]

ARTIFACT_FIELDS = [
    ("overlay_path", "Overlay MP4"),
    ("timeline_json", "Timeline JSON"),
    ("csv_path", "Features CSV"),
    ("report_html", "Report HTML"),
    ("config_path", "Config YAML"),
    ("log_path", "Run Log"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-root", default="outputs/web_runs")
    return parser.parse_known_args(argv)[0]


def tool_choices() -> list[str]:
    return [tool["name"] for tool in av_toolbox.list_tools()]


def artifact_items(result: av_toolbox.AVResult) -> list[tuple[str, Path]]:
    items = []
    payload = result.to_dict()
    for field, label in ARTIFACT_FIELDS:
        value = payload.get(field)
        if not value:
            continue
        path = Path(value)
        if path.exists():
            items.append((label, path))
    return items


def uploaded_media_path(uploaded_file: Any, output_root: Path) -> Path | None:
    if uploaded_file is None:
        return None
    upload_dir = output_root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    target = upload_dir / f"{int(time.time())}_{safe_name}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def build_run_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "sample_fps": values["sample_fps"],
        "max_seconds": values["max_seconds"] or None,
        "sample_rate": values["sample_rate"] or None,
        "hop_length": values["hop_length"] or None,
        "window_sec": values["window_sec"] or None,
        "overlay_fps": values["overlay_fps"] or None,
        "model_name": values["model_name"] or None,
        "checkpoint": values["checkpoint"] or None,
        "offline": values["offline"],
        "include_sim_matrix": values["include_sim_matrix"],
        "load_size": values["load_size"] or None,
        "plot_size": values["plot_size"] or None,
        "device": values["device"] or None,
        "batch_size": values["batch_size"] or None,
        "fp16": values["fp16"],
        "cache_dir": values["cache_dir"] or None,
        "workspace_dir": values["workspace_dir"] or None,
        "keep_workspace": values["keep_workspace"],
        "export_json": values["export_json"],
        "export_csv": values["export_csv"],
        "export_report": values["export_report"],
        "export_overlay": values["export_overlay"],
    }
    return kwargs


def main() -> None:
    args = parse_args()
    st = _streamlit()

    st.set_page_config(
        page_title="av-toolbox",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("av-toolbox")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tools = tool_choices()
    with st.sidebar:
        tool_name = st.selectbox("Tool", tools, index=0)
        uploaded_file = st.file_uploader("Upload", type=VIDEO_AUDIO_EXTENSIONS)
        input_path_text = st.text_input(
            "Media Path",
            value="data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4",
        )
        output_dir_text = st.text_input("Output Directory", value=str(output_root / "latest"))

        st.divider()
        device = st.text_input("Device", value="auto")
        batch_size = st.number_input("Batch Size", min_value=0, value=0, step=1)
        fp16 = st.checkbox("FP16", value=False)
        cache_dir = st.text_input("Cache Dir", value="")
        workspace_dir = st.text_input("Workspace Dir", value="")
        keep_workspace = st.checkbox("Keep Workspace", value=False)

        st.divider()
        sample_fps = st.number_input("Sample FPS", min_value=0.1, value=5.0, step=0.5)
        max_seconds = st.number_input("Max Seconds", min_value=0.0, value=0.0, step=1.0)
        sample_rate = st.number_input("Sample Rate", min_value=0, value=0, step=1000)
        hop_length = st.number_input("Hop Length", min_value=0, value=0, step=128)
        window_sec = st.number_input("Window Seconds", min_value=0.0, value=0.0, step=1.0)
        overlay_fps = st.number_input("Overlay FPS", min_value=0.0, value=0.0, step=1.0)

        denseav_selected = tool_name == "av.denseav"
        model_name = ""
        checkpoint = ""
        load_size = 0
        plot_size = 0
        offline = False
        include_sim_matrix = False
        if denseav_selected:
            st.divider()
            model_name = st.selectbox("Model", ["sound_and_language", "sound"], index=0)
            checkpoint = st.text_input("Checkpoint", value="")
            load_size = st.number_input("Load Size", min_value=0, value=224, step=32)
            plot_size = st.number_input("Plot Size", min_value=0, value=720, step=32)
            offline = st.checkbox("Offline", value=False)
            include_sim_matrix = st.checkbox("Sim Matrix", value=False)

        st.divider()
        export_overlay = st.checkbox("Overlay", value=True)
        export_json = st.checkbox("JSON", value=True)
        export_csv = st.checkbox("CSV", value=True)
        export_report = st.checkbox("HTML", value=True)

        run_clicked = st.button("Run", type="primary", use_container_width=True)

    uploaded_path = uploaded_media_path(uploaded_file, output_root)
    input_path = uploaded_path or Path(input_path_text).expanduser()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Input")
        if input_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"} and input_path.exists():
            st.video(str(input_path))
        elif input_path.exists():
            st.audio(str(input_path))
        st.code(str(input_path))

    if run_clicked:
        values = {
            "sample_fps": sample_fps,
            "max_seconds": max_seconds,
            "sample_rate": int(sample_rate),
            "hop_length": int(hop_length),
            "window_sec": window_sec,
            "overlay_fps": overlay_fps,
            "model_name": model_name,
            "checkpoint": checkpoint,
            "offline": offline,
            "include_sim_matrix": include_sim_matrix,
            "load_size": int(load_size),
            "plot_size": int(plot_size),
            "device": device,
            "batch_size": int(batch_size),
            "fp16": fp16,
            "cache_dir": cache_dir,
            "workspace_dir": workspace_dir,
            "keep_workspace": keep_workspace,
            "export_json": export_json,
            "export_csv": export_csv,
            "export_report": export_report,
            "export_overlay": export_overlay,
        }
        try:
            with st.status(f"Running {tool_name}", expanded=True):
                result = av_toolbox.run_tool(
                    tool_name,
                    input_path=input_path,
                    output_dir=Path(output_dir_text).expanduser(),
                    **build_run_kwargs(values),
                )
            st.session_state["last_result"] = result.to_dict()
            st.success("Done")
        except Exception as exc:  # pragma: no cover - exercised by Streamlit runtime
            st.error(f"{type(exc).__name__}: {exc}")

    result_payload = st.session_state.get("last_result")
    with right:
        st.subheader("Output")
        if result_payload:
            overlay = result_payload.get("overlay_path")
            if overlay and Path(overlay).exists():
                st.video(str(overlay))
            st.json(result_payload)
            for label, path in artifact_items(_result_from_dict(result_payload)):
                st.download_button(
                    label=label,
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=_mime_for(path),
                    use_container_width=True,
                )
        else:
            st.empty()


def _result_from_dict(payload: dict[str, Any]) -> av_toolbox.AVResult:
    return av_toolbox.AVResult(
        tool_name=payload["tool_name"],
        input_path=payload.get("input_path"),
        output_dir=payload.get("output_dir"),
        overlay_path=payload.get("overlay_path"),
        timeline_json=payload.get("timeline_json"),
        csv_path=payload.get("csv_path"),
        report_html=payload.get("report_html"),
        config_path=payload.get("config_path"),
        log_path=payload.get("log_path"),
        metadata=payload.get("metadata", {}),
    )


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".json":
        return "application/json"
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".html", ".htm"}:
        return "text/html"
    if suffix in {".yaml", ".yml"}:
        return "application/yaml"
    return "text/plain"


def _streamlit() -> Any:
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "av-toolbox web UI requires Streamlit. Install with: pip install -e '.[web]'"
        ) from exc
    return st


if __name__ == "__main__":
    main()
