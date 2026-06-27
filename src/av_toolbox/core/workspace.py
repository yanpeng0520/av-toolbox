"""Temporary workspace lifecycle for tool runs."""

from __future__ import annotations

import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class WorkspaceInfo:
    path: Path
    kept: bool
    cleaned: bool = False


class WorkspaceManager:
    """Create and clean an isolated per-run temporary workspace."""

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        *,
        keep_workspace: bool = False,
    ) -> None:
        self.root = Path(workspace_dir).expanduser() if workspace_dir else Path(tempfile.gettempdir()) / "av_toolbox"
        self.keep_workspace = keep_workspace
        self.info: WorkspaceInfo | None = None

    def __enter__(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"run-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        self.info = WorkspaceInfo(path=path, kept=self.keep_workspace)
        return path

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.info is None:
            return False
        if self.keep_workspace:
            self.info.cleaned = False
            return False
        shutil.rmtree(self.info.path, ignore_errors=True)
        self.info.cleaned = True
        return False

    def metadata(self) -> dict[str, object]:
        if self.info is None:
            return {"workspace_path": None, "workspace_kept": self.keep_workspace, "workspace_cleaned": False}
        return {
            "workspace_path": str(self.info.path),
            "workspace_kept": self.info.kept,
            "workspace_cleaned": self.info.cleaned,
        }

