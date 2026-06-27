"""Local web UI launcher for av-toolbox."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def web_app_path() -> Path:
    """Return the Streamlit app entry file."""
    return Path(__file__).with_name("web_app.py")


def build_streamlit_command(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    output_root: str | Path = "outputs/web_runs",
) -> list[str]:
    """Build the command used by ``av-toolbox serve``."""
    return [
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


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    output_root: str | Path = "outputs/web_runs",
) -> int:
    """Start the local web UI and block until it exits."""
    if importlib.util.find_spec("streamlit") is None:
        from av_toolbox.web_server import run_simple_server

        return run_simple_server(
            host=host,
            port=port,
            output_root=output_root,
        )

    env = dict(os.environ)
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    command = build_streamlit_command(
        host=host,
        port=port,
        output_root=output_root,
    )
    return subprocess.run(command, env=env, check=False).returncode
