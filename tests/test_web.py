from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from av_toolbox.core.result import AVResult
from av_toolbox.public_demo import public_run_kwargs, public_tool_name, public_workflow_names
from av_toolbox.ui_defaults import default_form_values, default_workflow_name, resolve_existing_input_path
from av_toolbox.web import build_streamlit_command, web_app_path
from av_toolbox.web_app import artifact_items, build_run_kwargs, tool_choices, tool_parameter_defaults
from av_toolbox.web_server import FormField
from av_toolbox.web_server import artifact_items as server_artifact_items
from av_toolbox.web_server import (
    is_allowed_path,
    make_handler,
    render_page,
    render_public_page,
    resolve_output_dir,
    resolve_public_input_path,
)


def test_streamlit_command_targets_packaged_web_app() -> None:
    command = build_streamlit_command(
        host="0.0.0.0",
        port=8600,
        output_root="outputs/ui",
    )

    assert command[1:4] == ["-m", "streamlit", "run"]
    assert command[4] == str(web_app_path())
    assert "--server.address" in command
    assert "0.0.0.0" in command
    assert "--server.port" in command
    assert "8600" in command
    assert command[-2:] == ["--output-root", "outputs/ui"]


def test_public_streamlit_command_enforces_demo_args() -> None:
    command = build_streamlit_command(
        host="127.0.0.1",
        port=8501,
        output_root="/srv/demo",
        public_demo=True,
        public_max_seconds=12,
        public_max_upload_mb=50,
        public_enable_denseav=True,
        page_title="Demo Lab",
    )

    assert command[command.index("--server.maxUploadSize") + 1] == "50"
    assert command.index("--server.maxUploadSize") < command.index("--")
    assert command[command.index("--page-title") + 1] == "Demo Lab"
    assert "--public-demo" in command
    assert command[command.index("--public-max-seconds") + 1] == "12"
    assert command[command.index("--public-max-upload-mb") + 1] == "50"
    assert "--public-enable-denseav" in command


def test_public_workflows_keep_denseav_opt_in() -> None:
    assert "DenseAV" not in public_workflow_names()
    assert public_tool_name("AV Sync") == "av.sync_correspondence"
    try:
        public_tool_name("DenseAV")
    except ValueError:
        pass
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("DenseAV should be disabled by default")

    assert "DenseAV" in public_workflow_names(enable_denseav=True)
    assert public_tool_name("DenseAV", enable_denseav=True) == "av.denseav"
    assert public_run_kwargs(max_seconds=9)["max_seconds"] == 9


def test_web_tool_choices_use_registry() -> None:
    names = tool_choices()

    assert "audio.beat_detection" in names
    assert "av.denseav" in names
    assert "video.motion" in names


def test_build_run_kwargs_keeps_tool_defaults_until_overridden() -> None:
    assert build_run_kwargs({"sample_fps": 25.0}) == {}


def test_build_run_kwargs_only_sends_explicit_overrides() -> None:
    kwargs = build_run_kwargs({
        "tool_name": "av.denseav",
        "override_defaults": True,
        "sample_fps": "25.0",
        "max_seconds": "",
        "model_name": "sound",
        "checkpoint": "",
        "offline": False,
        "include_sim_matrix": False,
        "audio_sample_rate": "",
        "load_size": "224",
        "plot_size": "720",
        "expected_sha256": "",
        "device": "",
        "batch_size": "",
        "fp16": True,
        "cache_dir": "",
        "workspace_dir": "",
        "keep_workspace": False,
        "export_json": True,
        "export_csv": False,
        "export_report": True,
        "export_overlay": True,
    })

    assert kwargs == {
        "sample_fps": 25.0,
        "model_name": "sound",
        "load_size": 224,
        "plot_size": 720,
        "fp16": True,
        "export_csv": False,
    }


def test_ui_defaults_point_at_bundled_demo_and_latest_output(tmp_path) -> None:
    values = default_form_values(tmp_path)

    assert default_workflow_name() == "Motion"
    assert values["tool_name"] == "video.motion"
    assert values["media_path"] == "data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4"
    assert Path(values["output_dir"]).parent == tmp_path
    assert Path(values["output_dir"]).name.startswith("run_")
    assert values["device"] == ""
    assert values["override_defaults"] is False


def test_ui_resolves_bundled_default_media_from_other_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = resolve_existing_input_path(default_form_values(tmp_path)["media_path"])

    assert path.exists()
    assert path.name == "Clever_Cat_Outsmarts_Warrior_square.mp4"


def test_denseav_workflow_uses_tool_defaults(tmp_path) -> None:
    values = default_form_values(tmp_path, "DenseAV")
    kwargs = build_run_kwargs(values)

    assert values["tool_name"] == "av.denseav"
    assert kwargs == {}


def test_tool_parameter_defaults_follow_selected_tool_signature() -> None:
    motion = tool_parameter_defaults("video.motion")
    denseav = tool_parameter_defaults("av.denseav")

    assert motion["sample_fps"] == "5.0"
    assert motion["threshold"] == "15.0"
    assert motion["downscale_width"] == "512"
    assert "model_name" not in motion
    assert denseav["model_name"] == "sound_and_language"
    assert denseav["load_size"] == "224"
    assert denseav["plot_size"] == "720"


def test_artifact_items_only_returns_existing_paths(tmp_path) -> None:
    timeline = tmp_path / "timeline.json"
    timeline.write_text("{}")
    result = AVResult(
        tool_name="test.tool",
        timeline_json=timeline,
        overlay_path=tmp_path / "missing.mp4",
    )

    assert artifact_items(result) == [("Timeline JSON", timeline)]


def test_builtin_web_page_renders_registered_tools(tmp_path) -> None:
    html = render_page(output_root=tmp_path)

    assert "av-toolbox" in html
    assert "audio.beat_detection" in html
    assert "av.denseav" in html
    assert "video.motion" in html
    assert "Workflow" in html
    assert "Motion" in html
    assert "Threshold" in html
    assert 'value="15.0"' in html
    assert "data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4" in html
    assert 'class="top-progress"' in html
    assert 'aria-label="Processing video input"' in html
    assert 'document.body.classList.add("is-running")' in html


def test_builtin_web_result_keeps_input_left_output_right_results_underneath(tmp_path) -> None:
    overlay = tmp_path / "overlay.mp4"
    overlay.write_bytes(b"mp4")
    timeline = tmp_path / "timeline.json"
    timeline.write_text("{}")
    result = {
        "tool_name": "video.motion",
        "input_path": str(overlay),
        "overlay_path": str(overlay),
        "timeline_json": str(timeline),
        "metadata": {"summary": {"sample_count": 1}},
    }

    html = render_page(output_root=tmp_path, result=result)

    assert "result-stack" in html
    assert "result-details" in html
    assert "<h2>Input</h2>" in html
    assert "<h2>Output</h2>" in html
    assert "<h2>Results</h2>" in html
    assert html.index("<h2>Input</h2>") < html.index("<h2>Output</h2>")
    assert html.index("<h2>Output</h2>") < html.index("<h2>Results</h2>")
    assert html.index("Overlay MP4") > html.index("<h2>Results</h2>")


def test_public_web_page_hides_local_filesystem_controls(tmp_path) -> None:
    html = render_public_page(output_root=tmp_path, page_title="Demo Lab")

    assert "<title>Demo Lab</title>" in html
    assert "<h1>Demo Lab</h1>" in html
    assert "av-toolbox" in html
    assert "Workflow" in html
    assert "AV Sync" in html
    assert "Sample video" in html
    assert "Upload" in html
    assert "Analyze Seconds" in html
    assert "Overlay" in html
    assert "Media Path" not in html
    assert "Output Directory" not in html
    assert "Tool" not in html
    assert "DenseAV" not in html
    assert 'class="top-progress"' in html
    assert 'document.body.classList.add("is-running")' in html

    dense_html = render_public_page(output_root=tmp_path, public_enable_denseav=True)
    assert "DenseAV" in dense_html


def test_builtin_web_falls_back_from_unusable_output_dir(tmp_path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    output_root = tmp_path / "web"
    form = {"output_dir": FormField(name="output_dir", value=str(blocked))}

    resolved = resolve_output_dir(form, output_root)

    assert resolved.parent == output_root
    assert resolved.exists()
    assert resolved != blocked


def test_public_upload_resolution_caps_and_sanitizes(tmp_path) -> None:
    sample = resolve_public_input_path({"input_mode": FormField(name="input_mode", value="sample")}, tmp_path, public_max_upload_mb=1)
    assert sample.name == "Clever_Cat_Outsmarts_Warrior_square.mp4"

    form = {"input_mode": FormField(name="input_mode", value="upload"), "upload": FormField(name="upload", filename="my clip.mp4", data=b"abc")}
    path = resolve_public_input_path(form, tmp_path, public_max_upload_mb=1)

    assert path.parent == tmp_path / "public_uploads"
    assert path.name.endswith("my_clip.mp4")
    assert path.read_bytes() == b"abc"

    too_large = {"input_mode": FormField(name="input_mode", value="upload"), "upload": FormField(name="upload", filename="clip.mp4", data=b"x" * (1024 * 1024 + 1))}
    try:
        resolve_public_input_path(too_large, tmp_path, public_max_upload_mb=1)
    except ValueError as exc:
        assert "larger than 1 MB" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("oversized upload should fail")

    bad_ext = {"input_mode": FormField(name="input_mode", value="upload"), "upload": FormField(name="upload", filename="clip.txt", data=b"abc")}
    try:
        resolve_public_input_path(bad_ext, tmp_path, public_max_upload_mb=1)
    except ValueError as exc:
        assert "Unsupported media extension" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("unsupported upload should fail")


def test_builtin_web_artifact_safety_and_listing(tmp_path) -> None:
    artifact = tmp_path / "run" / "timeline.json"
    artifact.parent.mkdir()
    artifact.write_text("{}")
    result = {"timeline_json": str(artifact), "overlay_path": str(tmp_path / "missing.mp4")}

    assert server_artifact_items(result) == [("Timeline JSON", artifact)]
    assert is_allowed_path(artifact, tmp_path) is True
    assert is_allowed_path(Path("/etc/passwd"), tmp_path) is False
    assert is_allowed_path(Path.cwd() / "README.md", tmp_path) is True
    assert is_allowed_path(Path.cwd() / "README.md", tmp_path, public_demo=True) is False


def test_builtin_web_run_endpoint_executes_registry_tool(tmp_path) -> None:
    output_root = tmp_path / "web"
    output_root.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(output_root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    root = Path(__file__).resolve().parents[1]
    demo = root / "data_segments" / "Clever_Cat_Outsmarts_Warrior_square.mp4"
    run_output = tmp_path / "run"
    form = urlencode({
        "tool_name": "video.motion",
        "media_path": str(demo),
        "output_dir": str(run_output),
        "export_json": "on",
        "export_csv": "on",
        "export_report": "on",
    }).encode("utf-8")

    try:
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}/run",
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
            assert response.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert "video.motion" in body
    timeline_paths = sorted(run_output.glob("*_timeline.json"))
    assert timeline_paths
    payload = json.loads(timeline_paths[0].read_text())
    assert payload["tool_name"] == "video.motion"
    assert payload["summary"]["sample_count"] > 0
