from __future__ import annotations

import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from av_toolbox.core.result import AVResult
from av_toolbox.public_demo import public_run_kwargs, public_tool_name, public_workflow_names
from av_toolbox.ui_defaults import category_from_label, default_form_values, default_workflow_name, resolve_existing_input_path, tool_names_for_category, tool_type_labels
from av_toolbox.web import build_streamlit_command, web_app_path
from av_toolbox.web_app import artifact_items, build_run_kwargs, tool_choices, tool_parameter_defaults, _normalize_streamlit_sys_path, _render_overlay_panel, _render_run_complete, _render_run_progress, _render_running_output_panel, _run_local, _transcript_text_from_result
from av_toolbox.web_server import FormField
from av_toolbox.web_server import artifact_items as server_artifact_items
from av_toolbox.web_server import (
    is_allowed_path,
    make_handler,
    render_page,
    render_public_page,
    render_result_details,
    resolve_output_dir,
    resolve_public_input_path,
)


def test_streamlit_sys_path_normalizer_prevents_av_shadowing() -> None:
    package_dir = web_app_path().parent
    src_dir = package_dir.parent
    original_path = list(sys.path)
    try:
        sys.path[:] = [str(package_dir), str(src_dir), *original_path]
        _normalize_streamlit_sys_path()

        assert str(package_dir) not in sys.path
        assert str(src_dir) in sys.path
    finally:
        sys.path[:] = original_path


def test_transcription_result_extracts_timestamped_transcript(tmp_path) -> None:
    timeline = tmp_path / "transcription_timeline.json"
    timeline.write_text(json.dumps({
        "segments": [
            {"start": 0.0, "end": 1.25, "text": " Hello"},
            {"start": 61.5, "end": 62.75, "text": "world"},
        ]
    }))

    transcript = _transcript_text_from_result({
        "tool_name": "audio.transcription",
        "timeline_json": str(timeline),
    })

    assert transcript == "00:00.000 - 00:01.250  Hello\n01:01.500 - 01:02.750  world"


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


def test_public_streamlit_command_defaults_to_10mb_origin_upload_cap() -> None:
    command = build_streamlit_command(
        host="127.0.0.1",
        port=8501,
        output_root="/srv/demo",
        public_demo=True,
    )

    assert command[command.index("--server.maxUploadSize") + 1] == "10"
    assert command[command.index("--public-max-upload-mb") + 1] == "10"


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
    expected_public_names = {
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
    }
    assert set(public_workflow_names()) == expected_public_names
    assert "DenseAV" not in public_workflow_names()
    assert public_tool_name("Image Quality") == "video.image_quality"
    assert public_tool_name("Blur Exposure") == "video.blur_exposure"
    assert public_tool_name("Obstruction") == "video.obstruction"
    try:
        public_tool_name("DenseAV")
    except ValueError:
        pass
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("DenseAV should be disabled by default")

    assert "DenseAV" in public_workflow_names(enable_denseav=True)
    assert public_tool_name("DenseAV", enable_denseav=True) == "av.denseav"
    assert public_run_kwargs(max_seconds=9)["max_seconds"] == 9


def test_shot_boundaries_workflow_uses_transnetv2_cut_detection() -> None:
    values = default_form_values(workflow_name="Shot Boundaries")

    assert values["tool_name"] == "video.cut_detection"
    assert values["backend"] == "transnetv2"
    assert public_tool_name("Shot Boundaries") == "video.cut_detection"
    assert tool_parameter_defaults("video.cut_detection")["backend"] == "transnetv2"


def test_web_tool_choices_use_registry() -> None:
    names = tool_choices()

    assert "audio.beat_detection" in names
    assert "av.denseav" in names
    assert "video.motion" in names


def test_tool_type_grouping_uses_registry_categories() -> None:
    records = [
        {"name": "video.motion", "category": "video", "description": ""},
        {"name": "audio.energy", "category": "audio", "description": ""},
        {"name": "av.denseav", "category": "av", "description": ""},
    ]

    assert tool_type_labels(records) == ["Video", "Audio", "Audio-Visual"]
    assert category_from_label("Audio-Visual") == "av"
    assert tool_names_for_category(records, "video") == ["video.motion"]
    assert tool_names_for_category(records, "audio") == ["audio.energy"]
    assert tool_names_for_category(records, "av") == ["av.denseav"]


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


def test_streamlit_local_run_does_not_mutate_output_dir_widget_key(tmp_path, monkeypatch) -> None:
    class GuardedSessionState(dict):
        def __setitem__(self, key, value):
            if key == "output_dir":
                raise AssertionError("output_dir widget key was mutated")
            super().__setitem__(key, value)

    class FakeStatus:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeStreamlit:
        def __init__(self):
            self.session_state = GuardedSessionState({"output_dir": "widget-owned"})

        def status(self, *_args, **_kwargs):
            return FakeStatus()

        def success(self, message):
            self.success_message = message

        def error(self, message):
            raise AssertionError(message)

    def fake_run_tool(tool_name, *, input_path, output_dir, **_kwargs):
        timeline = output_dir / "timeline.json"
        timeline.write_text("{}")
        return AVResult(tool_name=tool_name, input_path=input_path, output_dir=output_dir, timeline_json=timeline)

    st = FakeStreamlit()
    monkeypatch.setattr("av_toolbox.web_app.av_toolbox.run_tool", fake_run_tool)

    ok = _run_local(
        st,
        "video.motion",
        tmp_path / "input.mp4",
        str(tmp_path / "runs" / "motion"),
        tmp_path / "runs",
    )

    assert ok is True
    assert st.session_state["output_dir"] == "widget-owned"
    assert st.session_state["_last_output_dir"].endswith("motion")
    assert st.session_state["last_result"]["tool_name"] == "video.motion"
    assert st.success_message == "Done"


def test_streamlit_progress_banner_renders_visible_status() -> None:
    class FakeSlot:
        def markdown(self, body, *, unsafe_allow_html):
            self.body = body
            self.unsafe_allow_html = unsafe_allow_html

    slot = FakeSlot()

    _render_run_progress(slot, "video.motion")

    assert slot.unsafe_allow_html is True
    assert "av-run-progress" in slot.body
    assert 'aria-valuemin="0"' in slot.body
    assert 'aria-valuemax="100"' in slot.body
    assert "0%" in slot.body
    assert "100%" not in slot.body
    assert "Running video.motion" not in slot.body

    _render_run_complete(slot, "video.motion", success=True)
    assert 'aria-valuenow="100"' in slot.body
    assert "100%" in slot.body
    assert "Done: video.motion" not in slot.body


def test_streamlit_running_output_panel_keeps_input_context_visible() -> None:
    class FakeStreamlit:
        def __init__(self):
            self.calls = []

        def subheader(self, value):
            self.calls.append(("subheader", value))

        def markdown(self, value, *, unsafe_allow_html):
            self.calls.append(("markdown", value, unsafe_allow_html))

    st = FakeStreamlit()

    _render_running_output_panel(st, "video.motion")

    assert ("subheader", "Output") in st.calls
    markdown_calls = [call for call in st.calls if call[0] == "markdown"]
    assert markdown_calls
    body = markdown_calls[0][1]
    assert markdown_calls[0][2] is True
    assert "av-output-progress" in body
    assert "av-progress-track" not in body
    assert "av-run-progress" not in body
    assert "Output will appear here when ready." in body
    assert "Running video.motion" not in body
    assert "input preview stays visible" not in body


def test_streamlit_output_panel_falls_back_to_source_video_when_no_overlay(tmp_path) -> None:
    class FakeStreamlit:
        def __init__(self):
            self.calls = []

        def subheader(self, value):
            self.calls.append(("subheader", value))

        def video(self, value):
            self.calls.append(("video", value))

        def audio(self, value):
            self.calls.append(("audio", value))

        def caption(self, value):
            self.calls.append(("caption", value))

        def code(self, value):
            self.calls.append(("code", value))

        def info(self, value):
            self.calls.append(("info", value))

    source = tmp_path / "source.mp4"
    source.write_bytes(b"not a real mp4, only path existence matters here")
    st = FakeStreamlit()

    _render_overlay_panel(
        st,
        {"tool_name": "video.motion", "input_path": str(source), "overlay_path": None},
    )

    assert ("video", str(source)) in st.calls
    assert any(call[0] == "caption" and "did not produce an overlay MP4" in call[1] for call in st.calls)


def test_ui_defaults_keep_default_media_path_and_latest_output(tmp_path) -> None:
    values = default_form_values(tmp_path)

    assert default_workflow_name() == "Motion"
    assert values["tool_name"] == "video.motion"
    assert values["media_path"] == "data_segments/CatFu.mp4"
    assert Path(values["output_dir"]).parent == tmp_path
    assert Path(values["output_dir"]).name.startswith("run_")
    assert values["device"] == ""
    assert values["override_defaults"] is False


def test_ui_resolves_default_media_path_from_other_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = resolve_existing_input_path(default_form_values(tmp_path)["media_path"])

    assert path.name == "CatFu.mp4"
    assert path.parent.name == "data_segments"


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


def test_artifact_items_only_returns_existing_non_log_paths(tmp_path) -> None:
    timeline = tmp_path / "timeline.json"
    timeline.write_text("{}")
    log = tmp_path / "run.log"
    log.write_text("debug details")
    result = AVResult(
        tool_name="test.tool",
        timeline_json=timeline,
        overlay_path=tmp_path / "missing.mp4",
        log_path=log,
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
    assert "data_segments/CatFu.mp4" in html
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
    assert sample.suffix == ".mp4"
    assert sample.exists()

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


def test_public_sample_falls_back_to_generated_media(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("av_toolbox.web_server.DEFAULT_MEDIA_PATH", "missing-demo.mp4")

    sample = resolve_public_input_path({"input_mode": FormField(name="input_mode", value="sample")}, tmp_path, public_max_upload_mb=1)

    assert sample.parent == tmp_path / "public_sample"
    assert sample.name == "av_toolbox_public_sample.mp4"
    assert sample.exists()


def test_public_sample_falls_back_when_lfs_pointer_is_unpulled(tmp_path, monkeypatch) -> None:
    pointer = tmp_path / "sample.mp4"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc123\n"
        "size 3770566\n"
    )
    monkeypatch.setattr("av_toolbox.web_server.DEFAULT_MEDIA_PATH", str(pointer))

    sample = resolve_public_input_path({"input_mode": FormField(name="input_mode", value="sample")}, tmp_path, public_max_upload_mb=1)

    assert sample != pointer
    assert sample.parent == tmp_path / "public_sample"
    assert sample.exists()


def test_builtin_web_result_json_hides_log_path(tmp_path) -> None:
    log = tmp_path / "run.log"
    log.write_text("debug details")
    html = render_result_details({"tool_name": "test.tool", "log_path": str(log)})

    assert "log_path" not in html
    assert str(log) not in html


def test_builtin_web_artifact_safety_and_listing(tmp_path) -> None:
    artifact = tmp_path / "run" / "timeline.json"
    artifact.parent.mkdir()
    artifact.write_text("{}")
    log = tmp_path / "run" / "run.log"
    log.write_text("debug details")
    result = {
        "timeline_json": str(artifact),
        "overlay_path": str(tmp_path / "missing.mp4"),
        "log_path": str(log),
    }

    assert server_artifact_items(result) == [("Timeline JSON", artifact)]
    assert is_allowed_path(artifact, tmp_path) is True
    assert is_allowed_path(Path("/etc/passwd"), tmp_path) is False
    assert is_allowed_path(Path.cwd() / "README.md", tmp_path) is True
    assert is_allowed_path(Path.cwd() / "README.md", tmp_path, public_demo=True) is False


def test_builtin_web_run_endpoint_executes_registry_tool(tmp_path, demo_video_path: Path) -> None:
    output_root = tmp_path / "web"
    output_root.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(output_root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    run_output = tmp_path / "run"
    form = urlencode({
        "tool_name": "video.motion",
        "media_path": str(demo_video_path),
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
