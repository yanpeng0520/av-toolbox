"""Standard result object returned by av-toolbox tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _path_to_str(value: str | Path | None) -> str | None:
    return str(value) if value is not None else None


@dataclass(slots=True)
class AVResult:
    """Declared artifacts and metadata produced by one tool run."""

    tool_name: str
    input_path: str | Path | None = None
    output_dir: str | Path | None = None
    overlay_path: str | Path | None = None
    timeline_json: str | Path | None = None
    csv_path: str | Path | None = None
    report_html: str | Path | None = None
    config_path: str | Path | None = None
    log_path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "tool_name": self.tool_name,
            "input_path": _path_to_str(self.input_path),
            "output_dir": _path_to_str(self.output_dir),
            "overlay_path": _path_to_str(self.overlay_path),
            "timeline_json": _path_to_str(self.timeline_json),
            "csv_path": _path_to_str(self.csv_path),
            "report_html": _path_to_str(self.report_html),
            "config_path": _path_to_str(self.config_path),
            "log_path": _path_to_str(self.log_path),
            "metadata": dict(self.metadata),
        }

