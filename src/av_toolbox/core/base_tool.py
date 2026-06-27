"""Base tool interface for av-toolbox."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from av_toolbox.core.cache import ModelCache
from av_toolbox.core.hardware import HardwareConfig
from av_toolbox.core.result import AVResult
from av_toolbox.core.workspace import WorkspaceManager


@dataclass(slots=True)
class ToolRunContext:
    output_dir: Path
    hardware: HardwareConfig
    cache: ModelCache
    workspace: Path


class BaseTool(ABC):
    """Common lifecycle for video, audio, and audio-visual tools."""

    name: str
    category: str
    description: str = ""

    def run(
        self,
        *,
        input_path: str | Path | None = None,
        output_dir: str | Path = "outputs",
        hardware: HardwareConfig | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        fp16: bool | None = None,
        cache_dir: str | Path | None = None,
        workspace_dir: str | Path | None = None,
        keep_workspace: bool = False,
        **kwargs: Any,
    ) -> AVResult:
        """Run a tool with standardized output, cache, and workspace handling."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        hardware_config = HardwareConfig.from_values(
            hardware=hardware,
            device=device,
            batch_size=batch_size,
            fp16=fp16,
        )
        cache = ModelCache(cache_dir)
        workspace_manager = WorkspaceManager(
            workspace_dir,
            keep_workspace=keep_workspace,
        )

        result: AVResult | None = None
        with workspace_manager as workspace:
            context = ToolRunContext(
                output_dir=output_path,
                hardware=hardware_config,
                cache=cache,
                workspace=workspace,
            )
            result = self._run(
                input_path=Path(input_path) if input_path is not None else None,
                context=context,
                **kwargs,
            )

        result.output_dir = result.output_dir or output_path
        result.input_path = result.input_path or input_path
        result.metadata.setdefault("hardware", hardware_config.to_dict())
        result.metadata.setdefault(
            "cache",
            {
                "cache_dir": str(cache.cache_dir),
                "weights_dir": str(cache.weights_dir),
            },
        )
        result.metadata.setdefault("workspace", workspace_manager.metadata())
        return result

    @abstractmethod
    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        **kwargs: Any,
    ) -> AVResult:
        """Tool-specific implementation."""

