"""Small helpers for tools that emit timeline/CSV/report artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from av_toolbox.core.base_tool import ToolRunContext
from av_toolbox.core.outputs import ArtifactPaths, make_artifact_paths
from av_toolbox.core.result import AVResult


def json_default(value: Any) -> Any:
    """Best-effort JSON conversion for NumPy/PyTorch scalar-ish values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def html_report(title: str, payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "<!doctype html>",
        "<html><head>",
        f"<title>{title}</title>",
        "</head><body>",
        f"<h1>{title}</h1>",
        f"<p>Input: {payload.get('input_path', '')}</p>",
        "<ul>",
    ]
    for key, value in summary.items():
        lines.append(f"<li>{key}: {value}</li>")
    lines.extend(["</ul>", "</body></html>"])
    return "\n".join(lines)


def write_standard_artifacts(
    *,
    tool_name: str,
    input_path: Path,
    context: ToolRunContext,
    config: dict[str, Any],
    timeline_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    csv_fields: list[str],
    export_json: bool,
    export_csv: bool,
    export_report: bool,
    log_lines: list[str],
) -> tuple[ArtifactPaths, AVResult]:
    artifacts = make_artifact_paths(
        input_path=input_path,
        output_dir=context.output_dir,
        tool_name=tool_name,
    )
    artifacts.config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    if export_json:
        artifacts.timeline_json.write_text(
            json.dumps(timeline_payload, indent=2, default=json_default)
        )
    if export_csv:
        write_csv(artifacts.csv_path, rows, csv_fields)
    if export_report:
        artifacts.report_html.write_text(html_report(f"{tool_name} report", timeline_payload))
    artifacts.log_path.write_text("\n".join(log_lines) + "\n")

    result = AVResult(
        tool_name=tool_name,
        input_path=input_path,
        output_dir=context.output_dir,
        timeline_json=artifacts.timeline_json if export_json else None,
        csv_path=artifacts.csv_path if export_csv else None,
        report_html=artifacts.report_html if export_report else None,
        config_path=artifacts.config_path,
        log_path=artifacts.log_path,
        metadata={
            "media": timeline_payload.get("media", {}),
            "summary": timeline_payload.get("summary", {}),
            "events": len(timeline_payload.get("events", [])),
        },
    )
    return artifacts, result


def resize_to_width(frame: Any, width: int, cv2: Any) -> Any:
    if width <= 0:
        return frame
    height, original_width = frame.shape[:2]
    if original_width <= width:
        return frame
    scale = width / float(original_width)
    size = (width, max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
