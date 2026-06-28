"""Streamlit application for av-toolbox runs."""

from __future__ import annotations

import argparse
from html import escape
import threading
import time
from pathlib import Path
from typing import Any

import av_toolbox
from av_toolbox.public_demo import (
    DEFAULT_LOCAL_PAGE_TITLE,
    DEFAULT_PUBLIC_MAX_SECONDS,
    DEFAULT_PUBLIC_MAX_UPLOAD_MB,
    DEFAULT_PUBLIC_PAGE_TITLE,
    default_public_workflow_name,
    page_title_from_env,
    public_enable_denseav_from_env,
    public_max_seconds_from_env,
    public_max_upload_mb_from_env,
    public_mode_from_env,
    public_run_kwargs,
    public_run_output_dir,
    public_tool_name,
    public_upload_bytes,
    public_upload_dir,
    public_workflow_names,
)
from av_toolbox.ui_defaults import (
    BASE_FORM_VALUES,
    DEFAULT_MEDIA_PATH,
    DEFAULT_OUTPUT_ROOT,
    PARAMETER_SPECS,
    RUNTIME_SPECS,
    default_form_values,
    ensure_writable_output_dir,
    parameter_defaults_from_callable,
    parse_field_value,
    resolve_existing_input_path,
    workflow_names,
)


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

PUBLIC_RUN_LOCK = threading.Lock()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--page-title", default=None)
    parser.add_argument("--public-demo", action="store_true", default=public_mode_from_env())
    parser.add_argument("--public-max-seconds", type=float, default=public_max_seconds_from_env(DEFAULT_PUBLIC_MAX_SECONDS))
    parser.add_argument("--public-max-upload-mb", type=int, default=public_max_upload_mb_from_env(DEFAULT_PUBLIC_MAX_UPLOAD_MB))
    parser.add_argument("--public-enable-denseav", action="store_true", default=public_enable_denseav_from_env())
    return parser.parse_known_args(argv)[0]


def tool_choices(*, public_demo: bool = False, public_enable_denseav: bool = False) -> list[str]:
    if not public_demo:
        return [tool["name"] for tool in av_toolbox.list_tools()]
    return [
        public_tool_name(name, enable_denseav=public_enable_denseav)
        for name in public_workflow_names(enable_denseav=public_enable_denseav)
    ]


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


def uploaded_media_path(
    uploaded_file: Any,
    output_root: Path,
    *,
    public_demo: bool = False,
    public_max_upload_mb: int = DEFAULT_PUBLIC_MAX_UPLOAD_MB,
) -> Path | None:
    if uploaded_file is None:
        return None
    payload = uploaded_file.getbuffer()
    if public_demo and len(payload) > public_upload_bytes(public_max_upload_mb):
        raise ValueError(f"Upload is larger than {public_max_upload_mb} MB")
    upload_dir = public_upload_dir(output_root) if public_demo else output_root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    target = upload_dir / f"{int(time.time())}_{safe_name}"
    target.write_bytes(payload)
    return target


def build_run_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    if not values.get("override_defaults"):
        return {}

    tool_name = str(values.get("tool_name") or "")
    tool_defaults = tool_parameter_defaults(tool_name) if tool_name else {}
    kwargs: dict[str, Any] = {}
    for name, default_value in tool_defaults.items():
        raw_value = values[name] if name in values else default_value
        value = parse_field_value(name, raw_value)
        default = parse_field_value(name, default_value)
        if value is None:
            continue
        if PARAMETER_SPECS[name].get("export") and value == default:
            continue
        if PARAMETER_SPECS[name]["kind"] == "bool" and value is False and default is False:
            continue
        kwargs[name] = value

    for name in RUNTIME_SPECS:
        value = parse_field_value(name, values.get(name))
        if value in (None, False):
            continue
        kwargs[name] = value
    return kwargs


def tool_parameter_defaults(tool_name: str) -> dict[str, Any]:
    return parameter_defaults_from_callable(av_toolbox.get_tool(tool_name)._run)


def main() -> None:
    args = parse_args()
    st = _streamlit()
    public_demo = bool(args.public_demo)
    default_page_title = DEFAULT_PUBLIC_PAGE_TITLE if public_demo else DEFAULT_LOCAL_PAGE_TITLE
    page_title = page_title_from_env(args.page_title or default_page_title)

    st.set_page_config(
        page_title=page_title,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_pipeline_theme(st, public_demo=public_demo, page_title=page_title)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_slot = st.empty()

    tools = tool_choices(public_demo=public_demo, public_enable_denseav=args.public_enable_denseav)
    _ensure_default_state(st, output_root, public_demo=public_demo, public_enable_denseav=args.public_enable_denseav)
    public_input_mode = "Sample video"
    public_max_seconds = args.public_max_seconds
    public_export_overlay = True

    with st.sidebar:
        if public_demo:
            workflow_options = public_workflow_names(enable_denseav=args.public_enable_denseav)
            workflow_name = st.selectbox("Workflow", workflow_options, index=0, key="workflow_name")
            tool_name = public_tool_name(workflow_name, enable_denseav=args.public_enable_denseav)
            st.session_state["tool_name"] = tool_name
            public_input_mode = st.radio(
                "Input",
                ["Sample video", "Upload media"],
                index=_option_index(["Sample video", "Upload media"], st.session_state.get("public_input_mode", "Sample video")),
                horizontal=True,
                key="public_input_mode",
            )
            uploaded_file = (
                st.file_uploader("Upload", type=VIDEO_AUDIO_EXTENSIONS, key="upload")
                if public_input_mode == "Upload media"
                else None
            )
            public_max_seconds = st.number_input(
                "Analyze Seconds",
                min_value=0.1,
                max_value=max(0.1, float(args.public_max_seconds)),
                value=min(10.0, max(0.1, float(args.public_max_seconds))),
                step=1.0,
                key="public_max_seconds_value",
            )
            public_export_overlay = st.checkbox("Overlay", value=True, key="public_export_overlay")
            output_dir_text = ""
        else:
            workflow_name = st.selectbox("Workflow", workflow_names(), index=0, key="workflow_name")
            _sync_workflow_tool(st, workflow_name, output_root)
            tool_name = st.selectbox(
                "Tool",
                tools,
                index=_option_index(tools, st.session_state.get("tool_name", tools[0])),
                key="tool_name",
            )
            _sync_tool_parameter_defaults(st, tool_name)
            uploaded_file = st.file_uploader("Upload", type=VIDEO_AUDIO_EXTENSIONS, key="upload")
            st.text_input("Media Path", key="media_path")
            output_dir_text = st.text_input("Output Directory", key="output_dir")

            with st.expander("Advanced", expanded=False):
                st.checkbox("Override tool defaults", key="override_defaults")
                if st.session_state.get("override_defaults"):
                    _render_parameter_controls(st, tool_parameter_defaults(tool_name))
                    _render_runtime_controls(st)

        run_clicked = st.button("Run", type="primary", use_container_width=True)

    upload_error = None
    try:
        uploaded_path = uploaded_media_path(
            uploaded_file,
            output_root,
            public_demo=public_demo,
            public_max_upload_mb=args.public_max_upload_mb,
        )
    except ValueError as exc:
        uploaded_path = None
        upload_error = str(exc)

    if public_demo:
        input_path = (
            uploaded_path
            if public_input_mode == "Upload media"
            else resolve_existing_input_path(DEFAULT_MEDIA_PATH)
        )
    else:
        input_path = uploaded_path or resolve_existing_input_path(st.session_state.get("media_path", ""))

    if run_clicked:
        _render_top_progress(progress_slot)
        try:
            if public_demo:
                _run_public(st, tool_name, input_path, output_root, public_max_seconds, public_export_overlay)
            else:
                _run_local(st, tool_name, input_path, output_dir_text, output_root)
        finally:
            progress_slot.empty()

    result_payload = st.session_state.get("last_result")
    if result_payload:
        input_col, output_col = st.columns([1, 1], gap="large")
        with input_col:
            _render_input_panel(st, input_path, upload_error, public_demo=public_demo)
        with output_col:
            _render_overlay_panel(st, result_payload, public_demo=public_demo)
        st.divider()
        _render_result_details(st, result_payload)
    else:
        left, right = st.columns([1, 1], gap="large")
        with left:
            _render_input_panel(st, input_path, upload_error, public_demo=public_demo)
        with right:
            st.subheader("Output")
            st.empty()


def _render_top_progress(slot: Any) -> None:
    slot.markdown(
        '<div class="av-run-progress" role="progressbar" aria-label="Processing video input"></div>',
        unsafe_allow_html=True,
    )


def _render_input_panel(st: Any, input_path: Path | None, upload_error: str | None, *, public_demo: bool = False) -> None:
    st.subheader("Input")
    if upload_error:
        st.error(upload_error)
    if input_path and input_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"} and input_path.exists():
        st.video(str(input_path))
    elif input_path and input_path.exists():
        st.audio(str(input_path))
    if input_path and not public_demo:
        st.code(str(input_path))
    elif input_path and public_demo:
        st.caption(input_path.name)


def _render_overlay_panel(st: Any, result_payload: dict[str, Any], *, public_demo: bool = False) -> None:
    st.subheader("Output")
    overlay = result_payload.get("overlay_path")
    if overlay and Path(overlay).exists():
        st.video(str(overlay))
        if not public_demo:
            st.code(str(overlay))
    else:
        st.empty()


def _render_result_details(st: Any, result_payload: dict[str, Any]) -> None:
    st.subheader("Results")
    metadata = result_payload.get("metadata") or {}
    summary = metadata.get("summary") if isinstance(metadata, dict) else None
    if summary:
        st.json(summary, expanded=False)

    artifacts = artifact_items(_result_from_dict(result_payload))
    if artifacts:
        st.markdown("Artifacts")
    for label, path in artifacts:
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=_mime_for(path),
            use_container_width=True,
        )

    with st.expander("JSON", expanded=False):
        st.json(result_payload)


def _run_public(
    st: Any,
    tool_name: str,
    input_path: Path | None,
    output_root: Path,
    max_seconds: float,
    export_overlay: bool,
) -> None:
    if input_path is None:
        st.error("Upload a media file first.")
        return
    if not PUBLIC_RUN_LOCK.acquire(blocking=False):
        st.error("Another run is in progress. Try again shortly.")
        return
    try:
        output_dir = public_run_output_dir(output_root)
        with st.status(f"Running {tool_name}", expanded=True):
            result = av_toolbox.run_tool(
                tool_name,
                input_path=input_path,
                output_dir=output_dir,
                **public_run_kwargs(max_seconds=max_seconds, export_overlay=export_overlay),
            )
        st.session_state["last_result"] = result.to_dict()
        st.success("Done")
    except Exception as exc:  # pragma: no cover - exercised by Streamlit runtime
        st.error(f"{type(exc).__name__}: {exc}")
    finally:
        PUBLIC_RUN_LOCK.release()


def _run_local(st: Any, tool_name: str, input_path: Path, output_dir_text: str, output_root: Path) -> None:
    try:
        output_dir = ensure_writable_output_dir(output_dir_text, output_root)
        st.session_state["output_dir"] = str(output_dir)
        with st.status(f"Running {tool_name}", expanded=True):
            result = av_toolbox.run_tool(
                tool_name,
                input_path=input_path,
                output_dir=output_dir,
                **build_run_kwargs(_state_values(st)),
            )
        st.session_state["last_result"] = result.to_dict()
        st.success("Done")
    except Exception as exc:  # pragma: no cover - exercised by Streamlit runtime
        st.error(f"{type(exc).__name__}: {exc}")


def _ensure_default_state(st: Any, output_root: Path, *, public_demo: bool, public_enable_denseav: bool) -> None:
    if st.session_state.get("_av_toolbox_initialized"):
        return
    workflow_name = (
        default_public_workflow_name(enable_denseav=public_enable_denseav)
        if public_demo
        else workflow_names()[0]
    )
    for key, value in default_form_values(output_root, workflow_name).items():
        st.session_state[key] = value
    if public_demo:
        st.session_state["tool_name"] = public_tool_name(workflow_name, enable_denseav=public_enable_denseav)
    st.session_state["workflow_name"] = workflow_name
    st.session_state["_av_toolbox_active_workflow"] = workflow_name
    st.session_state["_av_toolbox_active_tool"] = st.session_state["tool_name"]
    _set_tool_defaults_in_state(st, tool_parameter_defaults(st.session_state["tool_name"]))
    st.session_state["_av_toolbox_initialized"] = True


def _sync_workflow_tool(st: Any, workflow_name: str, output_root: Path) -> None:
    state_key = "_av_toolbox_active_workflow"
    if st.session_state.get(state_key) == workflow_name:
        return
    defaults = default_form_values(output_root, workflow_name)
    st.session_state["tool_name"] = defaults["tool_name"]
    st.session_state[state_key] = workflow_name


def _sync_tool_parameter_defaults(st: Any, tool_name: str) -> None:
    state_key = "_av_toolbox_active_tool"
    if st.session_state.get(state_key) == tool_name:
        return
    _set_tool_defaults_in_state(st, tool_parameter_defaults(tool_name))
    st.session_state[state_key] = tool_name


def _set_tool_defaults_in_state(st: Any, defaults: dict[str, Any]) -> None:
    for name, spec in PARAMETER_SPECS.items():
        st.session_state[name] = False if spec["kind"] == "bool" else ""
    for name, value in defaults.items():
        st.session_state[name] = value


def _render_parameter_controls(st: Any, defaults: dict[str, Any]) -> None:
    if defaults:
        st.markdown("Tool Parameters")
    for name in defaults:
        _render_field(st, name, PARAMETER_SPECS[name])


def _render_runtime_controls(st: Any) -> None:
    st.markdown("Runtime")
    for name, spec in RUNTIME_SPECS.items():
        _render_field(st, name, spec)


def _render_field(st: Any, name: str, spec: dict[str, Any]) -> None:
    label = str(spec["label"])
    kind = spec["kind"]
    if kind == "bool":
        st.checkbox(label, key=name)
        return
    if kind == "choice":
        choices = [""] + list(spec.get("choices", []))
        st.selectbox(
            label,
            choices,
            index=_option_index(choices, st.session_state.get(name, "")),
            key=name,
        )
        return
    st.text_input(label, key=name)


def _state_values(st: Any) -> dict[str, Any]:
    values = dict(BASE_FORM_VALUES)
    values["tool_name"] = st.session_state.get("tool_name", values["tool_name"])
    values["override_defaults"] = st.session_state.get("override_defaults", False)
    for name in PARAMETER_SPECS:
        values[name] = st.session_state.get(name)
    for name in RUNTIME_SPECS:
        values[name] = st.session_state.get(name)
    return values


def _option_index(options: list[str], selected: Any) -> int:
    try:
        return options.index(str(selected))
    except ValueError:
        return 0


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


def _apply_pipeline_theme(st: Any, *, public_demo: bool, page_title: str) -> None:
    mode = "public" if public_demo else "local"
    safe_page_title = escape(page_title)
    chip_class = "warn" if public_demo else "ok"
    st.markdown(
        f"""
        <style>
          :root {{ --av-bg: #f3f5f4; --av-surface: #ffffff; --av-panel: #eef2f1; --av-border: #cfd8d5; --av-ink: #17201d; --av-muted: #63716c; --av-accent: #287c72; --av-warn: #b26b24; --av-ok: #2f7a45; }}
          .stApp {{ background: var(--av-bg); color: var(--av-ink); }}
          [data-testid="stSidebar"] {{ background: var(--av-panel); border-right: 1px solid var(--av-border); }}
          [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.65rem; }}
          .block-container {{ padding-top: 1.1rem; max-width: 1500px; }}
          div[data-testid="stExpander"] {{ border: 1px solid var(--av-border); border-radius: 8px; background: var(--av-surface); }}
          div[data-testid="stMetric"] {{ border: 1px solid var(--av-border); border-radius: 8px; background: var(--av-panel); padding: 8px 10px; }}
          div[data-testid="stTabs"] button {{ font-weight: 700; }}
          .av-topbar {{ min-height: 64px; margin: -0.2rem 0 1rem; padding: 0 4px 12px; border-bottom: 1px solid var(--av-border); display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
          .av-brand h1 {{ margin: 0; font-size: 22px; line-height: 1.15; }}
          .av-brand span {{ color: var(--av-muted); font-size: 12px; font-weight: 650; }}
          .av-chip {{ display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border: 1px solid var(--av-border); border-radius: 999px; background: var(--av-surface); color: var(--av-muted); font-size: 12px; font-weight: 750; }}
          .av-chip.ok {{ color: var(--av-ok); border-color: #b7d2bd; background: #eef7ef; }}
          .av-chip.warn {{ color: var(--av-warn); border-color: #dec8aa; background: #fff7ed; }}
          .av-run-progress {{ position: fixed; inset: 0 0 auto 0; height: 3px; z-index: 100000; pointer-events: none; overflow: hidden; background: rgba(40, 124, 114, 0.12); }}
          .av-run-progress::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 42%; background: var(--av-accent); animation: av-run-progress 1.08s ease-in-out infinite; }}
          @keyframes av-run-progress {{ 0% {{ transform: translateX(-105%); }} 55% {{ transform: translateX(72vw); }} 100% {{ transform: translateX(105vw); }} }}
        </style>
        <div class="av-topbar">
          <div class="av-brand"><h1>{safe_page_title}</h1><span>av-toolbox</span></div>
          <div class="av-chip {chip_class}">{mode}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
