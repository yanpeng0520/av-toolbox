from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from av_toolbox.core.result import AVResult
from av_toolbox.web import build_streamlit_command, web_app_path
from av_toolbox.web_app import artifact_items, build_run_kwargs, tool_choices
from av_toolbox.web_server import artifact_items as server_artifact_items
from av_toolbox.web_server import is_allowed_path, make_handler, render_page


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


def test_web_tool_choices_use_registry() -> None:
    names = tool_choices()

    assert "audio.beat_detection" in names
    assert "av.denseav" in names
    assert "video.motion" in names


def test_build_run_kwargs_normalizes_empty_values() -> None:
    kwargs = build_run_kwargs({
        "sample_fps": 25.0,
        "max_seconds": 0.0,
        "sample_rate": 0,
        "hop_length": 0,
        "window_sec": 0.0,
        "overlay_fps": 0.0,
        "model_name": "",
        "checkpoint": "",
        "offline": False,
        "include_sim_matrix": False,
        "load_size": 224,
        "plot_size": 720,
        "device": "auto",
        "batch_size": 0,
        "fp16": True,
        "cache_dir": "",
        "workspace_dir": "",
        "keep_workspace": False,
        "export_json": True,
        "export_csv": True,
        "export_report": True,
        "export_overlay": True,
    })

    assert kwargs["sample_fps"] == 25.0
    assert kwargs["max_seconds"] is None
    assert kwargs["sample_rate"] is None
    assert kwargs["checkpoint"] is None
    assert kwargs["batch_size"] is None
    assert kwargs["plot_size"] == 720
    assert kwargs["fp16"] is True


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


def test_builtin_web_artifact_safety_and_listing(tmp_path) -> None:
    artifact = tmp_path / "run" / "timeline.json"
    artifact.parent.mkdir()
    artifact.write_text("{}")
    result = {"timeline_json": str(artifact), "overlay_path": str(tmp_path / "missing.mp4")}

    assert server_artifact_items(result) == [("Timeline JSON", artifact)]
    assert is_allowed_path(artifact, tmp_path) is True
    assert is_allowed_path(Path("/etc/passwd"), tmp_path) is False


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
        "sample_fps": "4",
        "max_seconds": "1",
        "device": "cpu",
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
