"""Streamlit application for av-toolbox runs."""

from __future__ import annotations

import argparse
import json
from html import escape
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _normalize_streamlit_sys_path() -> None:
    """Prevent the Streamlit script directory from shadowing top-level packages."""
    package_dir = Path(__file__).resolve().parent
    src_dir = package_dir.parent
    cleaned = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            cleaned.append(entry)
            continue
        if resolved == package_dir:
            continue
        cleaned.append(entry)
    if not any(Path(entry or ".").resolve() == src_dir for entry in cleaned):
        cleaned.insert(0, str(src_dir))
    sys.path[:] = cleaned


_normalize_streamlit_sys_path()

import av_toolbox  # noqa: E402
from av_toolbox.public_demo import (  # noqa: E402
    DEFAULT_LOCAL_PAGE_TITLE,
    DEFAULT_PUBLIC_MAX_SECONDS,
    DEFAULT_PUBLIC_MAX_UPLOAD_MB,
    DEFAULT_PUBLIC_PAGE_TITLE,
    default_public_workflow_name,
    is_lfs_pointer_file,
    page_title_from_env,
    public_enable_denseav_from_env,
    public_max_seconds_from_env,
    public_max_upload_mb_from_env,
    public_mode_from_env,
    public_run_kwargs,
    public_run_output_dir,
    public_sample_media_path,
    public_tool_name,
    public_upload_bytes,
    public_upload_dir,
    public_workflow_names,
)
from av_toolbox.ui_defaults import (  # noqa: E402
    BASE_FORM_VALUES,
    DEFAULT_MEDIA_PATH,
    DEFAULT_OUTPUT_ROOT,
    PARAMETER_SPECS,
    RUNTIME_SPECS,
    category_from_label,
    default_form_values,
    default_tool_type_label,
    ensure_writable_output_dir,
    parameter_defaults_from_callable,
    parse_field_value,
    resolve_existing_input_path,
    tool_description,
    tool_display_name,
    tool_name_by_display_name,
    tool_names_for_category,
    tool_type_labels,
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


def tool_records(*, public_demo: bool = False, public_enable_denseav: bool = False) -> list[dict[str, str]]:
    records = av_toolbox.list_tools()
    if not public_demo:
        return records
    public_names = {
        public_tool_name(name, enable_denseav=public_enable_denseav)
        for name in public_workflow_names(enable_denseav=public_enable_denseav)
    }
    return [tool for tool in records if tool["name"] in public_names]


def tool_choices(*, public_demo: bool = False, public_enable_denseav: bool = False) -> list[str]:
    return [
        tool["name"]
        for tool in tool_records(public_demo=public_demo, public_enable_denseav=public_enable_denseav)
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


def _tool_description_from_records(tool_name: str, tools: list[dict[str, str]]) -> str:
    for tool in tools:
        if tool.get("name") == tool_name:
            return str(tool.get("description") or "")
    return ""


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

    tools = tool_records(public_demo=public_demo, public_enable_denseav=args.public_enable_denseav)
    tool_names = [tool["name"] for tool in tools]
    _ensure_default_state(st, output_root, public_demo=public_demo, public_enable_denseav=args.public_enable_denseav)
    public_input_mode = "Sample video"
    public_max_seconds = args.public_max_seconds
    public_export_overlay = True

    with st.sidebar:
        type_options = tool_type_labels(tools)
        current_tool = st.session_state.get("tool_name", tool_names[0])
        current_type = default_tool_type_label(current_tool, tools)
        if "tool_type" not in st.session_state:
            st.session_state["tool_type"] = current_type
            st.session_state["_av_toolbox_active_type"] = current_type
        tool_type = st.radio(
            "Type",
            type_options,
            index=_option_index(type_options, st.session_state.get("tool_type", current_type)),
            horizontal=True,
            key="tool_type",
        )
        category = category_from_label(tool_type)
        category_tool_names = tool_names_for_category(tools, category)
        if not category_tool_names:
            category_tool_names = tool_names
        if st.session_state.get("_av_toolbox_active_type") != tool_type:
            st.session_state["tool_name"] = category_tool_names[0]
            st.session_state["_av_toolbox_active_type"] = tool_type

        current_tool = st.session_state.get("tool_name", category_tool_names[0])
        if current_tool not in category_tool_names:
            current_tool = category_tool_names[0]
            st.session_state["tool_name"] = current_tool
        tool_label_options = [tool_display_name(name) for name in category_tool_names]
        selected_tool_label = st.selectbox(
            "Tool",
            tool_label_options,
            index=_option_index(tool_label_options, tool_display_name(current_tool)),
            key="tool_display_name",
        )
        tool_name = tool_name_by_display_name(selected_tool_label, category_tool_names)
        st.session_state["tool_name"] = tool_name
        _sync_tool_parameter_defaults(st, tool_name)
        description = tool_description(tool_name, _tool_description_from_records(tool_name, tools))
        if description:
            st.caption(description)

        if public_demo:
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
                value=max(0.1, float(args.public_max_seconds)),
                step=1.0,
                key="public_max_seconds_value",
            )
            public_export_overlay = st.checkbox("Overlay", value=True, key="public_export_overlay")
            output_dir_text = ""
        else:
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
        default_sample_path = resolve_existing_input_path(DEFAULT_MEDIA_PATH)
        input_path = (
            uploaded_path
            if public_input_mode == "Upload media"
            else default_sample_path
            if default_sample_path.exists() and not is_lfs_pointer_file(default_sample_path)
            else public_sample_media_path(output_root)
        )
    else:
        input_path = uploaded_path or resolve_existing_input_path(st.session_state.get("media_path", ""))

    result_payload = st.session_state.get("last_result")
    input_col, output_col = st.columns([1, 1], gap="large")
    with input_col:
        _render_input_panel(st, input_path, upload_error, public_demo=public_demo)
    output_slot = output_col.empty()
    details_slot = st.empty()

    if run_clicked:
        with output_slot.container():
            _render_running_output_panel(st, tool_name)
        if public_demo:
            _run_public(st, tool_name, input_path, output_root, public_max_seconds, public_export_overlay, progress_slot)
        else:
            _run_local(st, tool_name, input_path, output_dir_text, output_root, progress_slot=progress_slot)
        result_payload = st.session_state.get("last_result")

    if result_payload:
        with output_slot.container():
            _render_overlay_panel(st, result_payload, public_demo=public_demo)
        with details_slot.container():
            st.divider()
            _render_result_details(st, result_payload)
    elif not run_clicked:
        with output_slot.container():
            st.subheader("Output")
            st.empty()


def _render_run_progress(slot: Any, tool_name: str, percent: int = 0) -> None:
    del tool_name
    safe_percent = max(0, min(100, int(percent)))
    slot.markdown(
        f"""<div class="av-run-progress" role="progressbar" aria-label="Run progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{safe_percent}">
  <span class="av-progress-fill" style="width: {safe_percent}%"></span>
  <em>{safe_percent}%</em>
</div>""",
        unsafe_allow_html=True,
    )


def _render_run_complete(slot: Any, tool_name: str, *, success: bool) -> None:
    del tool_name
    percent = 100 if success else 0
    state = "done" if success else "failed"
    slot.markdown(
        f"""<div class="av-run-progress {state}" role="progressbar" aria-label="Run progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">
  <span class="av-progress-fill" style="width: {percent}%"></span>
  <em>{percent}%</em>
</div>""",
        unsafe_allow_html=True,
    )


def _render_running_output_panel(st: Any, tool_name: str) -> None:
    del tool_name
    st.subheader("Output")
    st.markdown(
        """<div class="av-output-progress" role="status" aria-live="polite">
  <span>Output will appear here when ready.</span>
</div>""",
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
        return

    input_value = result_payload.get("input_path")
    input_path = Path(input_value) if input_value else None
    if input_path and input_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"} and input_path.exists():
        st.video(str(input_path))
        st.caption(f"{result_payload.get('tool_name', 'This tool')} did not produce an overlay MP4; showing the source preview. Result artifacts are below.")
        return
    if input_path and input_path.exists():
        st.audio(str(input_path))
        st.caption(f"{result_payload.get('tool_name', 'This tool')} did not produce an overlay MP4; result artifacts are below.")
        return

    st.info("No overlay MP4 was produced for this tool. Result artifacts are below.")


def _render_result_details(st: Any, result_payload: dict[str, Any]) -> None:
    st.subheader("Results")
    metadata = result_payload.get("metadata") or {}
    summary = metadata.get("summary") if isinstance(metadata, dict) else None
    if summary:
        st.json(summary, expanded=False)

    transcript = _transcript_text_from_result(result_payload)
    if transcript:
        st.markdown("Transcript")
        st.text_area(
            "Transcript",
            transcript,
            height=260,
            label_visibility="collapsed",
        )

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
        st.json(_ui_result_payload(result_payload))


def _transcript_text_from_result(result_payload: dict[str, Any]) -> str:
    if result_payload.get("tool_name") != "audio.transcription":
        return ""
    timeline_value = result_payload.get("timeline_json")
    if not timeline_value:
        return ""
    timeline_path = Path(str(timeline_value))
    if not timeline_path.exists():
        return ""
    try:
        payload = json.loads(timeline_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""

    rows = payload.get("segments") or payload.get("events") or []
    if not isinstance(rows, list):
        return ""
    lines = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("content") or "").strip()
        if not text:
            continue
        start = _format_transcript_timestamp(row.get("start"))
        end = _format_transcript_timestamp(row.get("end"))
        if start and end:
            lines.append(f"{start} - {end}  {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def _format_transcript_timestamp(value: Any) -> str:
    try:
        seconds = max(0.0, float(value))
    except (TypeError, ValueError):
        return ""
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


def _ui_result_payload(result_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result_payload)
    payload.pop("log_path", None)
    return payload


def _visual_progress_value(elapsed_seconds: float) -> int:
    if elapsed_seconds <= 0:
        return 0
    return min(95, int(95 * (1 - 0.78 ** elapsed_seconds)))


def _run_with_top_progress(progress_slot: Any | None, tool_name: str, runner: Any) -> Any:
    if progress_slot is None:
        return runner()

    state: dict[str, Any] = {}

    def target() -> None:
        try:
            state["result"] = runner()
        except Exception as exc:  # pragma: no cover - exercised by Streamlit runtime
            state["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    progress = 0
    started = time.monotonic()
    _render_run_progress(progress_slot, tool_name, progress)
    worker.start()
    while worker.is_alive():
        next_progress = max(progress, _visual_progress_value(time.monotonic() - started))
        if next_progress != progress:
            progress = next_progress
            _render_run_progress(progress_slot, tool_name, progress)
        time.sleep(0.25)
    worker.join()

    if "error" in state:
        _render_run_complete(progress_slot, tool_name, success=False)
        raise state["error"]
    _render_run_complete(progress_slot, tool_name, success=True)
    return state.get("result")


def _run_public(
    st: Any,
    tool_name: str,
    input_path: Path | None,
    output_root: Path,
    max_seconds: float,
    export_overlay: bool,
    progress_slot: Any | None = None,
) -> bool:
    if input_path is None:
        st.error("Upload a media file first.")
        return False
    if not PUBLIC_RUN_LOCK.acquire(blocking=False):
        st.error("Another run is in progress. Try again shortly.")
        return False
    try:
        output_dir = public_run_output_dir(output_root)

        def runner() -> Any:
            return av_toolbox.run_tool(
                tool_name,
                input_path=input_path,
                output_dir=output_dir,
                **public_run_kwargs(max_seconds=max_seconds, export_overlay=export_overlay),
            )

        result = _run_with_top_progress(progress_slot, tool_name, runner)
        st.session_state["last_result"] = result.to_dict()
        st.success("Done")
        return True
    except Exception as exc:  # pragma: no cover - exercised by Streamlit runtime
        st.error(f"{type(exc).__name__}: {exc}")
        return False
    finally:
        PUBLIC_RUN_LOCK.release()


def _run_local(
    st: Any,
    tool_name: str,
    input_path: Path,
    output_dir_text: str,
    output_root: Path,
    progress_slot: Any | None = None,
) -> bool:
    try:
        output_dir = ensure_writable_output_dir(output_dir_text, output_root)
        st.session_state["_last_output_dir"] = str(output_dir)
        run_kwargs = build_run_kwargs(_state_values(st))

        def runner() -> Any:
            return av_toolbox.run_tool(
                tool_name,
                input_path=input_path,
                output_dir=output_dir,
                **run_kwargs,
            )

        if progress_slot is None:
            with st.status(f"Running {tool_name}", expanded=False):
                result = runner()
        else:
            result = _run_with_top_progress(progress_slot, tool_name, runner)
        st.session_state["last_result"] = result.to_dict()
        st.success("Done")
        return True
    except Exception as exc:  # pragma: no cover - exercised by Streamlit runtime
        st.error(f"{type(exc).__name__}: {exc}")
        return False


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
    active_tool = st.session_state["tool_name"]
    st.session_state["workflow_name"] = workflow_name
    st.session_state["tool_display_name"] = tool_display_name(active_tool)
    st.session_state["_av_toolbox_active_workflow"] = workflow_name
    st.session_state["_av_toolbox_active_tool"] = active_tool
    _set_tool_defaults_in_state(st, tool_parameter_defaults(active_tool))
    st.session_state["_av_toolbox_initialized"] = True


def _sync_workflow_tool(st: Any, workflow_name: str, output_root: Path) -> None:
    state_key = "_av_toolbox_active_workflow"
    if st.session_state.get(state_key) == workflow_name:
        return
    defaults = default_form_values(output_root, workflow_name)
    st.session_state["tool_name"] = defaults["tool_name"]
    st.session_state["tool_display_name"] = tool_display_name(defaults["tool_name"])
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
          :root {{ --av-bg: #f6f7f4; --av-surface: #fbfcfa; --av-panel: #eef2ef; --av-border: #cfd8d3; --av-ink: #17201d; --av-muted: #5f6d68; --av-accent: #287c72; --av-warn: #9d641f; --av-ok: #2f7042; }}
          .stApp {{ background: var(--av-bg); color: var(--av-ink); }}
          [data-testid="stSidebar"] {{ background: var(--av-panel); border-right: 1px solid var(--av-border); }}
          [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.65rem; }}
          .block-container {{ padding-top: 1.1rem; max-width: 1500px; }}
          div[data-testid="stExpander"] {{ border: 1px solid var(--av-border); border-radius: 8px; background: var(--av-surface); }}
          div[data-testid="stMetric"] {{ border: 1px solid var(--av-border); border-radius: 8px; background: var(--av-panel); padding: 8px 10px; }}
          div[data-testid="stTabs"] button {{ font-weight: 700; }}
          .av-topbar {{ min-height: 68px; margin: -0.2rem 0 1rem; padding: 14px 16px; border: 1px solid var(--av-border); border-radius: 8px; background: var(--av-surface); display: flex; align-items: center; justify-content: space-between; gap: 16px; box-shadow: 0 1px 2px rgba(23, 32, 29, 0.05); }}
          .av-brand h1 {{ margin: 0; color: var(--av-ink); font-size: 24px; font-weight: 760; line-height: 1.18; letter-spacing: 0; }}
          .av-brand span {{ color: var(--av-muted); font-size: 12px; font-weight: 650; letter-spacing: 0; }}
          .av-chip {{ display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border: 1px solid var(--av-border); border-radius: 999px; background: var(--av-surface); color: var(--av-muted); font-size: 12px; font-weight: 750; }}
          .av-chip.ok {{ color: var(--av-ok); border-color: #b7d2bd; background: #eef7ef; }}
          .av-chip.warn {{ color: var(--av-warn); border-color: #dec8aa; background: #fff7ed; }}
          .av-run-progress {{ position: relative; height: 16px; margin: 0 0 0.8rem; border-radius: 999px; overflow: hidden; background: rgba(40, 124, 114, 0.16); color: #ffffff; font-size: 10px; font-weight: 800; line-height: 16px; }}
          .av-run-progress em {{ position: absolute; right: 8px; top: 0; z-index: 2; font-style: normal; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35); }}
          .av-run-progress.done em {{ color: #ffffff; }}
          .av-run-progress.failed .av-progress-fill {{ background: #b36a54; }}
          .av-output-progress {{ margin: 0.35rem 0 0.75rem; padding: 12px 12px 14px; border: 1px solid #a7c9c2; border-radius: 8px; background: #eef8f5; color: var(--av-ink); display: grid; gap: 8px; }}
          .av-output-progress span {{ display: block; font-size: 12px; color: var(--av-muted); }}
          .av-progress-fill {{ position: absolute; inset: 0 auto 0 0; width: 0%; border-radius: inherit; background: linear-gradient(90deg, var(--av-accent), #3fa093); transition: width 240ms ease; }}
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
