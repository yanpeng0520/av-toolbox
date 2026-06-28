"""Local web UI launcher for av-toolbox."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from av_toolbox.public_demo import (
    DEFAULT_LOCAL_PAGE_TITLE,
    DEFAULT_PUBLIC_MAX_SECONDS,
    DEFAULT_PUBLIC_MAX_UPLOAD_MB,
    DEFAULT_PUBLIC_PAGE_TITLE,
    page_title_from_env,
    public_enable_denseav_from_env,
    public_max_seconds_from_env,
    public_max_upload_mb_from_env,
    public_mode_from_env,
)
from av_toolbox.ui_defaults import DEFAULT_OUTPUT_ROOT


def web_app_path() -> Path:
    """Return the Streamlit app entry file."""
    return Path(__file__).with_name("web_app.py")


def build_streamlit_command(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    public_demo: bool = False,
    public_max_seconds: float = DEFAULT_PUBLIC_MAX_SECONDS,
    public_max_upload_mb: int = DEFAULT_PUBLIC_MAX_UPLOAD_MB,
    public_enable_denseav: bool = False,
    page_title: str | None = None,
) -> list[str]:
    """Build the command used by ``av-toolbox serve``."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(web_app_path()),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--",
        "--output-root",
        str(output_root),
    ]
    if page_title:
        command.extend(["--page-title", str(page_title)])
    if public_demo:
        app_arg_index = command.index("--")
        command[app_arg_index:app_arg_index] = ["--server.maxUploadSize", str(public_max_upload_mb)]
        command.extend([
            "--public-demo",
            "--public-max-seconds",
            str(public_max_seconds),
            "--public-max-upload-mb",
            str(public_max_upload_mb),
        ])
        if public_enable_denseav:
            command.append("--public-enable-denseav")
    return command


def serve(
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
    """Start the local web UI and block until it exits."""
    public_demo = public_demo or public_mode_from_env()
    public_max_seconds = public_max_seconds_from_env(public_max_seconds)
    public_max_upload_mb = public_max_upload_mb_from_env(public_max_upload_mb)
    public_enable_denseav = public_enable_denseav or public_enable_denseav_from_env()
    default_page_title = DEFAULT_PUBLIC_PAGE_TITLE if public_demo else DEFAULT_LOCAL_PAGE_TITLE
    page_title = page_title_from_env(page_title or default_page_title)

    if importlib.util.find_spec("streamlit") is None:
        from av_toolbox.web_server import run_simple_server

        return run_simple_server(
            host=host,
            port=port,
            output_root=output_root,
            public_demo=public_demo,
            public_max_seconds=public_max_seconds,
            public_max_upload_mb=public_max_upload_mb,
            public_enable_denseav=public_enable_denseav,
            page_title=page_title,
        )

    env = dict(os.environ)
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    command = build_streamlit_command(
        host=host,
        port=port,
        output_root=output_root,
        public_demo=public_demo,
        public_max_seconds=public_max_seconds,
        public_max_upload_mb=public_max_upload_mb,
        public_enable_denseav=public_enable_denseav,
        page_title=page_title,
    )
    return subprocess.run(command, env=env, check=False).returncode
