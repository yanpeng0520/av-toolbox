"""Dependency-light local web UI for av-toolbox."""

from __future__ import annotations

import html
import json
import mimetypes
import time
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import av_toolbox
from av_toolbox.public_demo import (
    DEFAULT_LOCAL_PAGE_TITLE,
    DEFAULT_PUBLIC_MAX_SECONDS,
    DEFAULT_PUBLIC_MAX_UPLOAD_MB,
    DEFAULT_PUBLIC_PAGE_TITLE,
    default_public_workflow_name,
    page_title_from_env,
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
    default_workflow_name,
    ensure_writable_output_dir,
    parameter_defaults_from_callable,
    parse_field_value,
    project_root,
    resolve_existing_input_path,
    workflow_names,
)


MEDIA_EXTENSIONS = ".mp4,.mov,.mkv,.webm,.avi,.wav,.mp3,.flac,.m4a,.aac,.ogg"
CHECKBOX_FIELDS = {
    name
    for name, spec in {**PARAMETER_SPECS, **RUNTIME_SPECS, "override_defaults": {"kind": "bool"}}.items()
    if spec["kind"] == "bool"
}
ARTIFACT_FIELDS = [
    ("overlay_path", "Overlay MP4"),
    ("timeline_json", "Timeline JSON"),
    ("csv_path", "Features CSV"),
    ("report_html", "Report HTML"),
    ("config_path", "Config YAML"),
    ("log_path", "Run Log"),
]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}

APP_CHROME_CSS = """
    :root {
      color-scheme: light;
      --bg: #f3f5f4;
      --surface: #ffffff;
      --panel: #eef2f1;
      --panel-strong: #e3e9e7;
      --border: #cfd8d5;
      --ink: #17201d;
      --muted: #63716c;
      --accent: #287c72;
      --accent-strong: #1f655d;
      --warn: #b26b24;
      --ok: #2f7a45;
      --code: #111817;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    .top-progress { position: fixed; inset: 0 0 auto 0; height: 3px; z-index: 1000; pointer-events: none; overflow: hidden; opacity: 0; background: rgba(40, 124, 114, 0.12); transition: opacity 120ms ease; }
    .top-progress::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 42%; background: var(--accent); transform: translateX(-105%); }
    body.is-running .top-progress { opacity: 1; }
    body.is-running .top-progress::before { animation: top-progress-run 1.08s ease-in-out infinite; }
    @keyframes top-progress-run { 0% { transform: translateX(-105%); } 55% { transform: translateX(72vw); } 100% { transform: translateX(105vw); } }
    header { min-height: 64px; padding: 0 24px; border-bottom: 1px solid var(--border); background: var(--surface); display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    h1 { margin: 0; font-size: 20px; font-weight: 720; }
    h2 { margin: 0 0 12px; font-size: 15px; font-weight: 700; }
    h3 { margin: 10px 0 4px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; }
    main { display: grid; grid-template-columns: minmax(320px, 408px) minmax(0, 1fr); min-height: calc(100vh - 64px); }
    form { border-right: 1px solid var(--border); padding: 16px; background: var(--panel); overflow: auto; }
    fieldset { border: 0; padding: 0; margin: 0 0 16px; display: grid; gap: 10px; }
    legend { font-size: 12px; font-weight: 760; color: var(--muted); text-transform: uppercase; margin-bottom: 3px; letter-spacing: 0; }
    label { display: grid; gap: 5px; font-size: 13px; font-weight: 600; color: #23302c; }
    input, select { width: 100%; min-height: 36px; border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; font: inherit; background: var(--surface); color: var(--ink); }
    input:focus, select:focus { outline: 2px solid rgba(40, 124, 114, 0.22); border-color: var(--accent); }
    input[type="checkbox"] { width: 16px; min-height: 16px; accent-color: var(--accent); }
    button { width: 100%; min-height: 40px; border: 0; border-radius: 6px; background: var(--accent); color: #fff; font: inherit; font-weight: 760; cursor: pointer; }
    button:hover { background: var(--accent-strong); }
    details { margin: 0 0 16px; border-top: 1px solid var(--border); padding-top: 12px; }
    summary { cursor: pointer; color: var(--muted); font-size: 12px; font-weight: 760; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0; }
    video, audio { width: 100%; max-height: 68vh; background: #111; border-radius: 6px; border: 1px solid var(--border); }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f7f8f8; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-size: 12px; }
    .brand { display: grid; gap: 2px; }
    .brand span { color: var(--muted); font-size: 12px; font-weight: 600; }
    .header-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .chip { display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface); color: var(--muted); font-size: 12px; font-weight: 700; }
    .chip.ok { color: var(--ok); border-color: #b7d2bd; background: #eef7ef; }
    .chip.warn { color: var(--warn); border-color: #dec8aa; background: #fff7ed; }
    .node-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(142px, 1fr)); gap: 8px; margin: 0 0 14px; }
    .node-card { min-height: 58px; border: 1px solid var(--border); border-radius: 8px; padding: 9px 10px; background: var(--surface); display: grid; align-content: center; gap: 3px; }
    .node-card.active { border-color: #8fc5bd; background: #eff8f6; }
    .node-card b { font-size: 13px; overflow-wrap: anywhere; }
    .node-card span { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; }
    .workspace { display: grid; grid-template-rows: auto minmax(0, 1fr); overflow: hidden; }
    .service-row { padding: 14px 18px; border-bottom: 1px solid var(--border); background: var(--surface); }
    .panes { display: grid; grid-template-columns: minmax(260px, 0.9fr) minmax(320px, 1.1fr); overflow: hidden; }
    .result-stack { min-height: 0; overflow: auto; background: var(--surface); }
    .overlay-stage { background: var(--surface); }
    .overlay-stage video { max-height: 58vh; }
    .result-details { border-top: 1px solid var(--border); }
    .result-stack .pane { overflow: visible; }
    .pane { padding: 18px 22px; overflow: auto; background: var(--surface); }
    .pane + .pane { border-left: 1px solid var(--border); }
    .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .check-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .check { display: flex; align-items: center; gap: 8px; font-weight: 620; }
    .path { margin-top: 10px; color: #26332f; }
    .downloads { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }
    .downloads a { color: var(--accent-strong); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; text-decoration: none; background: var(--surface); font-weight: 650; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin: 0 0 14px; }
    .metric { border: 1px solid var(--border); border-radius: 8px; padding: 9px 10px; background: var(--panel); }
    .metric span { display: block; color: var(--muted); font-size: 11px; font-weight: 760; text-transform: uppercase; }
    .metric strong { display: block; margin-top: 3px; font-size: 15px; overflow-wrap: anywhere; }
    .error { margin-bottom: 14px; padding: 12px; border-radius: 8px; color: #7c1d1d; background: #fff0f0; border: 1px solid #f1b8b8; }
    .empty { color: var(--muted); border: 1px dashed var(--border); border-radius: 8px; padding: 18px; background: #fbfcfc; }
    @media (max-width: 1100px) { .panes { grid-template-columns: 1fr; } .pane + .pane { border-left: 0; border-top: 1px solid var(--border); } }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } form { border-right: 0; border-bottom: 1px solid var(--border); } }
"""


@dataclass(slots=True)
class FormField:
    name: str
    value: str = ""
    filename: str | None = None
    data: bytes = b""


FormData = dict[str, FormField]


def run_simple_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    public_demo: bool = False,
    public_max_seconds: float = DEFAULT_PUBLIC_MAX_SECONDS,
    public_max_upload_mb: int = DEFAULT_PUBLIC_MAX_UPLOAD_MB,
    public_enable_denseav: bool = False,
    page_title: str | None = None,
) -> int:
    """Start the stdlib local web UI and block until it exits."""
    output_path = Path(output_root).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    default_page_title = DEFAULT_PUBLIC_PAGE_TITLE if public_demo else DEFAULT_LOCAL_PAGE_TITLE
    page_title = page_title_from_env(page_title or default_page_title)
    handler = make_handler(
        output_path,
        public_demo=public_demo,
        public_max_seconds=public_max_seconds,
        public_max_upload_mb=public_max_upload_mb,
        public_enable_denseav=public_enable_denseav,
        page_title=page_title,
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"av-toolbox web UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def make_handler(
    output_root: Path,
    *,
    public_demo: bool = False,
    public_max_seconds: float = DEFAULT_PUBLIC_MAX_SECONDS,
    public_max_upload_mb: int = DEFAULT_PUBLIC_MAX_UPLOAD_MB,
    public_enable_denseav: bool = False,
    page_title: str = DEFAULT_LOCAL_PAGE_TITLE,
) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to one output root."""

    class AVToolboxHandler(BaseHTTPRequestHandler):
        server_version = "av-toolbox"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                if public_demo:
                    self._send_html(
                        render_public_page(
                            output_root=output_root,
                            public_enable_denseav=public_enable_denseav,
                            public_max_seconds=public_max_seconds,
                            page_title=page_title,
                        )
                    )
                else:
                    self._send_html(render_page(output_root=output_root, page_title=page_title))
                return
            if parsed.path == "/artifact":
                self._send_artifact(parsed.query, output_root)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/run":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            form_values: dict[str, Any] | None = None
            try:
                form = parse_multipart_form(self)
                if public_demo:
                    workflow_name = form_text(
                        form,
                        "workflow",
                        default_public_workflow_name(enable_denseav=public_enable_denseav),
                    )
                    input_path = resolve_public_input_path(form, output_root, public_max_upload_mb)
                    max_seconds = public_max_seconds_from_form(form, public_max_seconds)
                    export_overlay = form_bool(form, "export_overlay")
                    tool_name = public_tool_name(workflow_name, enable_denseav=public_enable_denseav)
                    result = av_toolbox.run_tool(
                        tool_name,
                        input_path=input_path,
                        output_dir=public_run_output_dir(output_root),
                        **public_run_kwargs(max_seconds=max_seconds, export_overlay=export_overlay),
                    )
                    self._send_html(
                        render_public_page(
                            output_root=output_root,
                            result=result.to_dict(),
                            form_values={"workflow": workflow_name, "input_mode": form_text(form, "input_mode", "sample"), "max_seconds": max_seconds, "export_overlay": export_overlay},
                            public_enable_denseav=public_enable_denseav,
                            public_max_seconds=public_max_seconds,
                            page_title=page_title,
                        )
                    )
                    return

                form_values = form_values_from_form(form, output_root)
                input_path = resolve_input_path(form, output_root)
                tool_name = form_text(form, "tool_name", "")
                output_dir = resolve_output_dir(form, output_root)
                form_values["output_dir"] = str(output_dir)
                result = av_toolbox.run_tool(
                    tool_name,
                    input_path=input_path,
                    output_dir=output_dir,
                    **form_run_kwargs(form),
                )
                self._send_html(
                    render_page(
                        output_root=output_root,
                        result=result.to_dict(),
                        form_values=form_values,
                        page_title=page_title,
                    )
                )
            except Exception as exc:  # pragma: no cover - exercised through browser use
                if public_demo:
                    self._send_html(
                        render_public_page(
                            output_root=output_root,
                            error=f"{type(exc).__name__}: {exc}",
                            public_enable_denseav=public_enable_denseav,
                            public_max_seconds=public_max_seconds,
                            page_title=page_title,
                        ),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                else:
                    self._send_html(
                        render_page(
                            output_root=output_root,
                            error=f"{type(exc).__name__}: {exc}",
                            form_values=form_values,
                            page_title=page_title,
                        ),
                        status=HTTPStatus.BAD_REQUEST,
                    )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_artifact(self, query: str, output_root: Path) -> None:
            params = parse_qs(query)
            raw_path = params.get("path", [""])[0]
            path = Path(unquote(raw_path)).expanduser()
            if not path.exists() or not is_allowed_path(path, output_root, public_demo=public_demo):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            inline = params.get("inline", ["0"])[0] == "1"
            payload = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            disposition = "inline" if inline else "attachment"
            self.send_header("Content-Disposition", f'{disposition}; filename="{path.name}"')
            self.end_headers()
            self.wfile.write(payload)

    return AVToolboxHandler


def parse_multipart_form(handler: BaseHTTPRequestHandler) -> FormData:
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0") or 0)
    body = handler.rfile.read(length)

    if content_type.startswith("application/x-www-form-urlencoded"):
        parsed = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
        return {name: FormField(name=name, value=values[0] if values else "") for name, values in parsed.items()}

    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=email_policy).parsebytes(header + body)
    fields: FormData = {}
    if not message.is_multipart():
        return fields

    for part in message.iter_parts():
        params = dict(part.get_params(header="content-disposition", unquote=True) or [])
        name = params.get("name")
        if not name:
            continue
        filename = params.get("filename")
        payload = part.get_payload(decode=True) or b""
        if filename:
            fields[name] = FormField(name=name, filename=filename, data=payload)
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = FormField(
                name=name,
                value=payload.decode(charset, errors="replace"),
            )
    return fields


def resolve_input_path(form: FormData, output_root: Path) -> Path:
    upload = form.get("upload")
    if upload is not None and upload.filename:
        upload_dir = output_root / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(str(upload.filename)).name.replace(" ", "_")
        target = upload_dir / f"{int(time.time())}_{safe_name}"
        target.write_bytes(upload.data)
        return target
    return resolve_existing_input_path(form_text(form, "media_path", ""))


def resolve_output_dir(form: FormData, output_root: Path) -> Path:
    return ensure_writable_output_dir(
        form_text(form, "output_dir", default_form_values(output_root)["output_dir"]),
        output_root,
    )


def resolve_public_input_path(form: FormData, output_root: Path, public_max_upload_mb: int) -> Path:
    upload = form.get("upload")
    input_mode = form_text(form, "input_mode", "upload" if upload and upload.filename else "sample")
    if input_mode == "sample":
        sample_path = resolve_existing_input_path(DEFAULT_MEDIA_PATH)
        if not sample_path.exists():
            raise ValueError("Sample media is not available")
        return sample_path
    if upload is None or not upload.filename:
        raise ValueError("Upload a media file first")
    if len(upload.data) > public_upload_bytes(public_max_upload_mb):
        raise ValueError(f"Upload is larger than {public_max_upload_mb} MB")
    suffix = Path(str(upload.filename)).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS and suffix not in AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported media extension: {suffix}")
    upload_dir = public_upload_dir(output_root)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(str(upload.filename)).name.replace(" ", "_")
    target = upload_dir / f"{int(time.time())}_{safe_name}"
    target.write_bytes(upload.data)
    return target


def public_max_seconds_from_form(form: FormData, public_max_seconds: float) -> float:
    try:
        requested = float(form_text(form, "max_seconds", str(public_max_seconds)))
    except ValueError:
        requested = public_max_seconds
    return min(max(0.1, requested), max(0.1, float(public_max_seconds)))


def form_values_from_form(form: FormData, output_root: Path) -> dict[str, Any]:
    workflow_name = form_text(form, "workflow", default_workflow_name())
    values = default_form_values(output_root, workflow_name)
    values["workflow"] = workflow_name
    for key in BASE_FORM_VALUES:
        if key in CHECKBOX_FIELDS:
            values[key] = form_bool(form, key)
        elif key in form:
            values[key] = form_text(form, key, "")
    for key in PARAMETER_SPECS:
        if key in CHECKBOX_FIELDS:
            values[key] = form_bool(form, key)
        elif key in form:
            values[key] = form_text(form, key, "")
    for key in RUNTIME_SPECS:
        if key in CHECKBOX_FIELDS:
            values[key] = form_bool(form, key)
        elif key in form:
            values[key] = form_text(form, key, "")
    if "output_dir" in form:
        values["output_dir"] = form_text(form, "output_dir", values["output_dir"])
    return values


def form_run_kwargs(form: FormData) -> dict[str, Any]:
    if not form_bool(form, "override_defaults"):
        return {}

    tool_defaults = tool_parameter_defaults(form_text(form, "tool_name", ""))
    kwargs: dict[str, Any] = {}
    for name, default_value in tool_defaults.items():
        if name in CHECKBOX_FIELDS:
            raw_value = form_bool(form, name) if name in form else default_value
        else:
            raw_value = form_text(form, name, str(default_value)) if name in form else default_value
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
        value = parse_field_value(name, form_bool(form, name) if name in CHECKBOX_FIELDS else form_text(form, name, ""))
        if value in (None, False):
            continue
        kwargs[name] = value
    return kwargs


def tool_parameter_defaults(tool_name: str) -> dict[str, Any]:
    if not tool_name:
        return {}
    return parameter_defaults_from_callable(av_toolbox.get_tool(tool_name)._run)


def all_tool_parameter_defaults() -> dict[str, dict[str, Any]]:
    return {
        tool["name"]: tool_parameter_defaults(tool["name"])
        for tool in av_toolbox.list_tools()
    }


def render_page(
    *,
    output_root: Path,
    page_title: str = DEFAULT_LOCAL_PAGE_TITLE,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    form_values: dict[str, Any] | None = None,
) -> str:
    values = form_values or default_form_values(output_root)
    selected_workflow = str(values.get("workflow") or default_workflow_name())
    tools = av_toolbox.list_tools()
    selected_tool = str(values.get("tool_name", ""))
    tool_defaults = tool_parameter_defaults(selected_tool)
    values = values_with_tool_defaults(values, tool_defaults)
    options = render_tool_options(tools, selected_tool)
    workflow_options = render_workflow_options(selected_workflow)
    workflow_payload = json.dumps(
        {name: {"tool_name": default_form_values(output_root, name)["tool_name"]} for name in workflow_names()},
        sort_keys=True,
    )
    tool_defaults_payload = json.dumps(all_tool_parameter_defaults(), sort_keys=True)
    error_html = f'<div class="error">{escape(error)}</div>' if error else ""
    input_path = str(result.get("input_path") if result else values.get("media_path", ""))
    input_preview = render_media_preview(input_path)
    output_placeholder = "" if result else '<div class="empty">No run output yet.</div>'
    workspace_html = render_workspace(
        result=result,
        input_preview=input_preview,
        error_html=error_html,
        output_placeholder=output_placeholder,
    )
    advanced_open = " open" if values.get("override_defaults") else ""
    parameter_rows = render_parameter_rows(values, tool_defaults)
    runtime_rows = render_runtime_rows(values)
    node_strip = render_node_strip(selected_tool=selected_tool)
    safe_page_title = html.escape(page_title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_page_title}</title>
  <style>
{APP_CHROME_CSS}
  </style>
</head>
<body>
  <div class="top-progress" role="progressbar" aria-label="Processing video input"></div>
  <header>
    <div class="brand">
      <h1>{safe_page_title}</h1>
      <span>av-toolbox</span>
    </div>
    <div class="header-meta">
      <span class="chip ok">local</span>
      <span class="chip">{escape(str(output_root))}</span>
    </div>
  </header>
  <main>
    <form action="/run" method="post" enctype="multipart/form-data">
      <fieldset>
        <legend>Dispatch</legend>
        <label>Workflow<select name="workflow">{workflow_options}</select></label>
        <label>Tool<select name="tool_name">{options}</select></label>
        <label>Upload<input type="file" name="upload" accept="{MEDIA_EXTENSIONS}"></label>
        <label>Media Path<input name="media_path" value="{field_value(values, "media_path")}"></label>
        <label>Output Directory<input name="output_dir" value="{field_value(values, "output_dir")}"></label>
      </fieldset>
      <details{advanced_open}>
        <summary>Advanced</summary>
        <fieldset>
          <label class="check"><input name="override_defaults" type="checkbox"{checked(values, "override_defaults")}> Override tool defaults</label>
          <h3>Tool Parameters</h3>
          <div class="grid-2" data-param-grid>{parameter_rows}</div>
          <h3>Runtime</h3>
          <div class="grid-2">{runtime_rows}</div>
        </fieldset>
      </details>
      <button type="submit">Run</button>
    </form>
    <section class="workspace">
      <div class="service-row">{node_strip}</div>
      {workspace_html}
    </section>
  </main>
  <script>
    const workflows = {workflow_payload};
    const toolDefaults = {tool_defaults_payload};
    function setField(form, name, value) {{
      const field = form.elements[name];
      if (!field) return;
      if (field.type === "checkbox") {{
        field.checked = Boolean(value);
        return;
      }}
      field.value = value ?? "";
    }}
    function applyToolDefaults(form, toolName) {{
      const defaults = toolDefaults[toolName] || {{}};
      document.querySelectorAll("[data-param]").forEach((row) => {{
        const name = row.getAttribute("data-param");
        const visible = Object.prototype.hasOwnProperty.call(defaults, name);
        row.style.display = visible ? "" : "none";
        if (visible) setField(form, name, defaults[name]);
      }});
    }}
    document.addEventListener("DOMContentLoaded", () => {{
      const form = document.querySelector("form");
      if (!form) return;
      const workflow = form.elements["workflow"];
      const tool = form.elements["tool_name"];
      workflow?.addEventListener("change", () => {{
        const values = workflows[workflow.value];
        if (values?.tool_name && tool) {{
          tool.value = values.tool_name;
          applyToolDefaults(form, tool.value);
        }}
      }});
      tool?.addEventListener("change", () => applyToolDefaults(form, tool.value));
      form.addEventListener("submit", () => document.body.classList.add("is-running"));
    }});
  </script>
</body>
</html>
"""


def render_public_page(
    *,
    output_root: Path,
    page_title: str = DEFAULT_PUBLIC_PAGE_TITLE,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    form_values: dict[str, Any] | None = None,
    public_enable_denseav: bool = False,
    public_max_seconds: float = DEFAULT_PUBLIC_MAX_SECONDS,
) -> str:
    values = form_values or {}
    selected_workflow = str(
        values.get("workflow")
        or default_public_workflow_name(enable_denseav=public_enable_denseav)
    )
    workflow_options = "\n".join(
        f'<option value="{escape(name)}"{selected_attr(name, selected_workflow)}>{escape(name)}</option>'
        for name in public_workflow_names(enable_denseav=public_enable_denseav)
    )
    input_mode = str(values.get("input_mode") or "sample")
    max_seconds = public_max_seconds_from_values(values, public_max_seconds)
    export_overlay = public_export_overlay_from_values(values)
    sample_selected = input_mode == "sample"
    upload_selected = "" if sample_selected else " selected"
    sample_selected_attr = " selected" if sample_selected else ""
    overlay_checked = " checked" if export_overlay else ""
    error_html = f'<div class="error">{escape(error)}</div>' if error else ""
    input_path = str(result.get("input_path") if result else resolve_existing_input_path(DEFAULT_MEDIA_PATH))
    input_preview = render_media_preview(input_path) if input_path else '<div class="empty">No media selected.</div>'
    output_placeholder = "" if result else '<div class="empty">No run output yet.</div>'
    workspace_html = render_workspace(
        result=result,
        input_preview=input_preview,
        error_html=error_html,
        output_placeholder=output_placeholder,
    )
    node_strip = render_node_strip(
        selected_tool=public_tool_name(selected_workflow, enable_denseav=public_enable_denseav),
        public_demo=True,
        public_enable_denseav=public_enable_denseav,
    )
    safe_page_title = html.escape(page_title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_page_title}</title>
  <style>
{APP_CHROME_CSS}
  </style>
</head>
<body>
  <div class="top-progress" role="progressbar" aria-label="Processing video input"></div>
  <header>
    <div class="brand">
      <h1>{safe_page_title}</h1>
      <span>av-toolbox</span>
    </div>
    <div class="header-meta">
      <span class="chip warn">public</span>
      <span class="chip">upload run</span>
    </div>
  </header>
  <main>
    <form action="/run" method="post" enctype="multipart/form-data">
      <fieldset>
        <legend>Dispatch</legend>
        <label>Workflow<select name="workflow">{workflow_options}</select></label>
        <label>Input<select name="input_mode"><option value="sample"{sample_selected_attr}>Sample video</option><option value="upload"{upload_selected}>Upload media</option></select></label>
        <label>Upload<input type="file" name="upload" accept="{MEDIA_EXTENSIONS}"></label>
        <label>Analyze Seconds<input name="max_seconds" type="number" min="0.1" max="{escape(public_max_seconds)}" step="1" value="{escape(max_seconds)}"></label>
        <label class="check"><input name="export_overlay" type="checkbox"{overlay_checked}> Overlay</label>
      </fieldset>
      <button type="submit">Run</button>
    </form>
    <section class="workspace">
      <div class="service-row">{node_strip}</div>
      {workspace_html}
    </section>
  </main>
  <script>
    document.addEventListener("DOMContentLoaded", () => {{
      document.querySelector("form")?.addEventListener("submit", () => document.body.classList.add("is-running"));
    }});
  </script>
</body>
</html>
"""


def public_max_seconds_from_values(values: dict[str, Any], public_max_seconds: float) -> str:
    try:
        requested = float(values.get("max_seconds", public_max_seconds))
    except (TypeError, ValueError):
        requested = public_max_seconds
    return str(min(max(0.1, requested), max(0.1, float(public_max_seconds))))


def public_export_overlay_from_values(values: dict[str, Any]) -> bool:
    value = values.get("export_overlay", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def render_node_strip(
    *,
    selected_tool: str,
    public_demo: bool = False,
    public_enable_denseav: bool = False,
) -> str:
    workflows = (
        public_workflow_names(enable_denseav=public_enable_denseav)
        if public_demo
        else workflow_names()
    )
    cards = []
    for workflow in workflows:
        tool_name = (
            public_tool_name(workflow, enable_denseav=public_enable_denseav)
            if public_demo
            else str(default_form_values(DEFAULT_OUTPUT_ROOT, workflow)["tool_name"])
        )
        active = " active" if tool_name == selected_tool else ""
        cards.append(
            "".join([
                f'<div class="node-card{active}">',
                f'<b>{escape(workflow)}</b>',
                f'<span>{escape(tool_name)}</span>',
                "</div>",
            ])
        )
    return f'<div class="node-strip">{"".join(cards)}</div>'


def values_with_tool_defaults(values: dict[str, Any], tool_defaults: dict[str, Any]) -> dict[str, Any]:
    merged = dict(values)
    for name, value in tool_defaults.items():
        merged.setdefault(name, value)
    for name in RUNTIME_SPECS:
        merged.setdefault(name, "")
    return merged


def render_tool_options(tools: list[dict[str, str]], selected: str) -> str:
    return "\n".join(
        f'<option value="{escape(tool["name"])}"{selected_attr(tool["name"], selected)}>{escape(tool["name"])}</option>'
        for tool in tools
    )


def render_workflow_options(selected: str) -> str:
    return "\n".join(
        f'<option value="{escape(name)}"{selected_attr(name, selected)}>{escape(name)}</option>'
        for name in workflow_names()
    )


def render_parameter_rows(values: dict[str, Any], tool_defaults: dict[str, Any]) -> str:
    return "\n".join(
        render_field(name, PARAMETER_SPECS[name], values, visible=name in tool_defaults, param=True)
        for name in PARAMETER_SPECS
    )


def render_runtime_rows(values: dict[str, Any]) -> str:
    return "\n".join(
        render_field(name, spec, values, visible=True, param=False)
        for name, spec in RUNTIME_SPECS.items()
    )


def render_field(name: str, spec: dict[str, Any], values: dict[str, Any], *, visible: bool, param: bool) -> str:
    style = "" if visible else ' style="display:none"'
    attr = f' data-param="{escape(name)}"' if param else ""
    label = escape(spec["label"])
    if spec["kind"] == "bool":
        return f'<label class="check"{attr}{style}><input name="{escape(name)}" type="checkbox"{checked(values, name)}> {label}</label>'
    if spec["kind"] == "choice":
        options = [""] + list(spec.get("choices", []))
        rendered = "".join(
            f'<option value="{escape(option)}"{selected_attr(option, str(values.get(name, "")))}>{escape(option)}</option>'
            for option in options
        )
        return f'<label{attr}{style}>{label}<select name="{escape(name)}">{rendered}</select></label>'
    input_type = "number" if spec["kind"] in {"int", "float"} else "text"
    step = f' step="{escape(spec.get("step", "any"))}"' if input_type == "number" else ""
    min_attr = ' min="0"' if input_type == "number" else ""
    return f'<label{attr}{style}>{label}<input name="{escape(name)}" type="{input_type}"{min_attr}{step} value="{field_value(values, name)}"></label>'


def render_media_preview(path_value: str | Path | None) -> str:
    if not path_value:
        return '<div class="empty">No media selected.</div>'
    path = resolve_existing_input_path(path_value)
    path_label = f'<pre class="path">{escape(path)}</pre>'
    if not path.exists():
        return f'<div class="empty">Media not found.</div>{path_label}'
    src = f"/artifact?path={quote(str(path))}&inline=1"
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return f'<video controls src="{src}"></video>{path_label}'
    if suffix in AUDIO_EXTENSIONS:
        return f'<audio controls src="{src}"></audio>{path_label}'
    return path_label


def render_workspace(
    *,
    result: dict[str, Any] | None,
    input_preview: str,
    error_html: str,
    output_placeholder: str,
) -> str:
    if not result:
        return f"""
      <div class="panes">
        <div class="pane"><h2>Input</h2>{input_preview}</div>
        <div class="pane"><h2>Output</h2>{error_html}{output_placeholder}</div>
      </div>
        """
    return f"""
      <div class="result-stack">
        {error_html}
        <div class="panes">
          <div class="pane"><h2>Input</h2>{input_preview}</div>
          {render_overlay_stage(result)}
        </div>
        <div class="pane result-details"><h2>Results</h2>{render_result_details(result)}</div>
      </div>
    """


def render_overlay_stage(result: dict[str, Any]) -> str:
    overlay = result.get("overlay_path")
    preview = render_media_preview(overlay) if overlay else ""
    if not preview:
        preview = '<div class="empty">No overlay video was generated.</div>'
    return f'<div class="pane overlay-stage"><h2>Output</h2>{preview}</div>'


def render_result_details(result: dict[str, Any]) -> str:
    summary = render_summary(result)
    downloads = "\n".join(
        f'<a href="/artifact?path={quote(str(path))}">{escape(label)}</a>'
        for label, path in artifact_items(result)
    )
    downloads_html = f'<div class="downloads">{downloads}</div>' if downloads else ""
    return f"""
      {summary}
      {downloads_html}
      <pre>{escape(json.dumps(result, indent=2))}</pre>
    """


def render_summary(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    summary = metadata.get("summary") if isinstance(metadata, dict) else None
    if not isinstance(summary, dict) or not summary:
        return ""
    items = list(summary.items())[:8]
    metrics = "\n".join(
        f'<div class="metric"><span>{escape(key)}</span><strong>{escape(value)}</strong></div>'
        for key, value in items
    )
    return f'<div class="metrics">{metrics}</div>'


def artifact_items(result: dict[str, Any]) -> list[tuple[str, Path]]:
    items = []
    for field, label in ARTIFACT_FIELDS:
        value = result.get(field)
        if not value:
            continue
        path = Path(value)
        if path.exists():
            items.append((label, path))
    return items


def is_allowed_path(path: Path, output_root: Path, *, public_demo: bool = False) -> bool:
    resolved = path.resolve()
    if public_demo:
        sample_path = resolve_existing_input_path(DEFAULT_MEDIA_PATH).resolve()
        allowed_roots = [output_root.resolve()]
        return (
            resolved == sample_path
            or any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots)
        )
    allowed_roots = [Path.cwd().resolve(), project_root().resolve(), output_root.resolve()]
    return any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots)


def field_value(values: dict[str, Any], key: str) -> str:
    return escape(values.get(key, ""))


def checked(values: dict[str, Any], key: str) -> str:
    return " checked" if bool(values.get(key)) else ""


def selected_attr(value: str, selected: str) -> str:
    return " selected" if value == selected else ""


def form_text(form: FormData, name: str, default: str = "") -> str:
    field = form.get(name)
    if field is None:
        return default
    return str(field.value)


def form_bool(form: FormData, name: str) -> bool:
    return name in form


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)
