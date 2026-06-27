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


MEDIA_EXTENSIONS = ".mp4,.mov,.mkv,.webm,.avi,.wav,.mp3,.flac,.m4a,.aac,.ogg"

@dataclass(slots=True)
class FormField:
    name: str
    value: str = ""
    filename: str | None = None
    data: bytes = b""


FormData = dict[str, FormField]


ARTIFACT_FIELDS = [
    ("overlay_path", "Overlay MP4"),
    ("timeline_json", "Timeline JSON"),
    ("csv_path", "Features CSV"),
    ("report_html", "Report HTML"),
    ("config_path", "Config YAML"),
    ("log_path", "Run Log"),
]


def run_simple_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    output_root: str | Path = "outputs/web_runs",
) -> int:
    """Start the stdlib local web UI and block until it exits."""
    output_path = Path(output_root).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    handler = make_handler(output_path)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"av-toolbox web UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def make_handler(output_root: Path) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to one output root."""

    class AVToolboxHandler(BaseHTTPRequestHandler):
        server_version = "av-toolbox"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_page(output_root=output_root))
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
            try:
                form = parse_multipart_form(self)
                input_path = resolve_input_path(form, output_root)
                tool_name = form_text(form, "tool_name", "")
                output_dir = Path(form_text(form, "output_dir", str(output_root / "latest"))).expanduser()
                result = av_toolbox.run_tool(
                    tool_name,
                    input_path=input_path,
                    output_dir=output_dir,
                    **form_run_kwargs(form),
                )
                self._send_html(render_page(output_root=output_root, result=result.to_dict()))
            except Exception as exc:  # pragma: no cover - exercised through browser use
                self._send_html(
                    render_page(
                        output_root=output_root,
                        error=f"{type(exc).__name__}: {exc}",
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
            if not path.exists() or not is_allowed_path(path, output_root):
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
    return Path(form_text(form, "media_path", "")).expanduser()


def form_run_kwargs(form: FormData) -> dict[str, Any]:
    return {
        "sample_fps": form_float(form, "sample_fps", 5.0),
        "max_seconds": form_optional_float(form, "max_seconds"),
        "sample_rate": form_optional_int(form, "sample_rate"),
        "hop_length": form_optional_int(form, "hop_length"),
        "window_sec": form_optional_float(form, "window_sec"),
        "overlay_fps": form_optional_float(form, "overlay_fps"),
        "model_name": form_optional_text(form, "model_name"),
        "checkpoint": form_optional_text(form, "checkpoint"),
        "offline": form_bool(form, "offline"),
        "include_sim_matrix": form_bool(form, "include_sim_matrix"),
        "load_size": form_optional_int(form, "load_size"),
        "plot_size": form_optional_int(form, "plot_size"),
        "device": form_optional_text(form, "device"),
        "batch_size": form_optional_int(form, "batch_size"),
        "fp16": form_bool(form, "fp16"),
        "cache_dir": form_optional_text(form, "cache_dir"),
        "workspace_dir": form_optional_text(form, "workspace_dir"),
        "keep_workspace": form_bool(form, "keep_workspace"),
        "export_json": form_bool(form, "export_json"),
        "export_csv": form_bool(form, "export_csv"),
        "export_report": form_bool(form, "export_report"),
        "export_overlay": form_bool(form, "export_overlay"),
    }


def render_page(
    *,
    output_root: Path,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    tools = av_toolbox.list_tools()
    options = "\n".join(
        f'<option value="{escape(tool["name"])}">{escape(tool["name"])}</option>'
        for tool in tools
    )
    result_html = render_result(result) if result else ""
    error_html = f'<div class="error">{escape(error)}</div>' if error else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>av-toolbox</title>
  <style>
    :root {{ color-scheme: light; --border: #d6d9df; --ink: #151821; --muted: #657084; --panel: #f7f8fa; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #ffffff; }}
    header {{ padding: 18px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 650; }}
    main {{ display: grid; grid-template-columns: minmax(320px, 420px) minmax(0, 1fr); min-height: calc(100vh - 62px); }}
    form {{ border-right: 1px solid var(--border); padding: 18px; background: var(--panel); overflow: auto; }}
    section {{ padding: 18px 24px; overflow: auto; }}
    fieldset {{ border: 0; padding: 0; margin: 0 0 18px; display: grid; gap: 10px; }}
    legend {{ font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }}
    label {{ display: grid; gap: 5px; font-size: 13px; font-weight: 560; }}
    input, select {{ width: 100%; min-height: 36px; border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; font: inherit; background: #fff; }}
    input[type="checkbox"] {{ width: 16px; min-height: 16px; }}
    .check-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    .check {{ display: flex; align-items: center; gap: 8px; font-weight: 560; }}
    button {{ min-height: 38px; border: 0; border-radius: 6px; background: #222a35; color: #fff; font: inherit; font-weight: 650; cursor: pointer; }}
    button:hover {{ background: #111820; }}
    video, audio {{ width: 100%; max-height: 72vh; background: #111; border-radius: 6px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f1f3f6; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-size: 12px; }}
    .downloads {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }}
    .downloads a {{ color: #1b4f9c; border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; text-decoration: none; background: #fff; }}
    .error {{ margin-bottom: 14px; padding: 12px; border-radius: 6px; color: #7c1d1d; background: #fff0f0; border: 1px solid #f1b8b8; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} form {{ border-right: 0; border-bottom: 1px solid var(--border); }} }}
  </style>
</head>
<body>
  <header>
    <h1>av-toolbox</h1>
    <span>{escape(str(output_root))}</span>
  </header>
  <main>
    <form action="/run" method="post" enctype="multipart/form-data">
      <fieldset>
        <legend>Run</legend>
        <label>Tool<select name="tool_name">{options}</select></label>
        <label>Upload<input type="file" name="upload" accept="{MEDIA_EXTENSIONS}"></label>
        <label>Media Path<input name="media_path" value="data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4"></label>
        <label>Output Directory<input name="output_dir" value="{escape(str(output_root / "latest"))}"></label>
      </fieldset>
      <fieldset>
        <legend>Hardware</legend>
        <label>Device<input name="device" value="auto"></label>
        <label>Batch Size<input name="batch_size" type="number" min="0" step="1" value="0"></label>
        <label class="check"><input name="fp16" type="checkbox"> FP16</label>
        <label>Cache Dir<input name="cache_dir"></label>
        <label>Workspace Dir<input name="workspace_dir"></label>
        <label class="check"><input name="keep_workspace" type="checkbox"> Keep Workspace</label>
      </fieldset>
      <fieldset>
        <legend>Processing</legend>
        <label>Sample FPS<input name="sample_fps" type="number" min="0.1" step="0.5" value="5"></label>
        <label>Max Seconds<input name="max_seconds" type="number" min="0" step="1" value="0"></label>
        <label>Sample Rate<input name="sample_rate" type="number" min="0" step="1000" value="0"></label>
        <label>Hop Length<input name="hop_length" type="number" min="0" step="128" value="0"></label>
        <label>Window Seconds<input name="window_sec" type="number" min="0" step="1" value="0"></label>
        <label>Overlay FPS<input name="overlay_fps" type="number" min="0" step="1" value="0"></label>
      </fieldset>
      <fieldset>
        <legend>DenseAV</legend>
        <label>Model<select name="model_name"><option value=""></option><option value="sound_and_language">sound_and_language</option><option value="sound">sound</option></select></label>
        <label>Checkpoint<input name="checkpoint"></label>
        <label>Load Size<input name="load_size" type="number" min="0" step="32" value="0"></label>
        <label>Plot Size<input name="plot_size" type="number" min="0" step="32" value="0"></label>
        <label class="check"><input name="offline" type="checkbox"> Offline</label>
        <label class="check"><input name="include_sim_matrix" type="checkbox"> Sim Matrix</label>
      </fieldset>
      <fieldset>
        <legend>Artifacts</legend>
        <div class="check-grid">
          <label class="check"><input name="export_overlay" type="checkbox" checked> Overlay</label>
          <label class="check"><input name="export_json" type="checkbox" checked> JSON</label>
          <label class="check"><input name="export_csv" type="checkbox" checked> CSV</label>
          <label class="check"><input name="export_report" type="checkbox" checked> HTML</label>
        </div>
      </fieldset>
      <button type="submit">Run</button>
    </form>
    <section>
      {error_html}
      {result_html}
    </section>
  </main>
</body>
</html>
"""


def render_result(result: dict[str, Any]) -> str:
    overlay = result.get("overlay_path")
    preview = ""
    if overlay and Path(overlay).exists():
        preview = f'<video controls src="/artifact?path={quote(str(overlay))}&inline=1"></video>'
    downloads = "\n".join(
        f'<a href="/artifact?path={quote(str(path))}">{escape(label)}</a>'
        for label, path in artifact_items(result)
    )
    return f"""
      {preview}
      <div class="downloads">{downloads}</div>
      <pre>{escape(json.dumps(result, indent=2))}</pre>
    """


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


def is_allowed_path(path: Path, output_root: Path) -> bool:
    resolved = path.resolve()
    allowed_roots = [Path.cwd().resolve(), output_root.resolve()]
    return any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots)


def form_text(form: FormData, name: str, default: str = "") -> str:
    field = form.get(name)
    if field is None:
        return default
    return str(field.value)


def form_optional_text(form: FormData, name: str) -> str | None:
    value = form_text(form, name, "").strip()
    return value or None


def form_bool(form: FormData, name: str) -> bool:
    return name in form


def form_float(form: FormData, name: str, default: float) -> float:
    value = form_text(form, name, "")
    try:
        return float(value)
    except ValueError:
        return default


def form_optional_float(form: FormData, name: str) -> float | None:
    value = form_float(form, name, 0.0)
    return value or None


def form_optional_int(form: FormData, name: str) -> int | None:
    value = form_text(form, name, "")
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed or None


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)
