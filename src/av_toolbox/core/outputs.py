"""Output artifact naming helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    prefix: Path
    overlay_path: Path
    timeline_json: Path
    csv_path: Path
    report_html: Path
    config_path: Path
    log_path: Path


def artifact_stem(input_path: str | Path, tool_name: str) -> str:
    stem = Path(input_path).stem
    tool_slug = tool_name.replace(".", "_").replace("-", "_")
    return f"{stem}_{tool_slug}"


def make_artifact_paths(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    tool_name: str,
) -> ArtifactPaths:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / artifact_stem(input_path, tool_name)
    return ArtifactPaths(
        prefix=prefix,
        overlay_path=prefix.with_name(prefix.name + "_overlay.mp4"),
        timeline_json=prefix.with_name(prefix.name + "_timeline.json"),
        csv_path=prefix.with_name(prefix.name + "_features.csv"),
        report_html=prefix.with_name(prefix.name + "_report.html"),
        config_path=prefix.with_name(prefix.name + "_config.yaml"),
        log_path=prefix.with_name(prefix.name + "_log.txt"),
    )

