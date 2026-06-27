"""Tool registry shared by the Python API, CLI, and future web UI."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from av_toolbox.core.base_tool import BaseTool
from av_toolbox.core.result import AVResult


class ToolRegistry:
    """Map stable tool names such as ``video.blur_exposure`` to tool objects."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> BaseTool:
        if not getattr(tool, "name", None):
            raise ValueError("Tool must define a non-empty name")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            known = ", ".join(self.list_names()) or "no tools registered"
            raise KeyError(f"Unknown tool {name!r}; known tools: {known}") from exc

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {
                "name": tool.name,
                "category": getattr(tool, "category", ""),
                "description": getattr(tool, "description", ""),
            }
            for tool in (self._tools[name] for name in self.list_names())
        ]

    def run(self, name: str, **kwargs: Any) -> AVResult:
        return self.get(name).run(**kwargs)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __iter__(self) -> Iterable[BaseTool]:
        for name in self.list_names():
            yield self._tools[name]


default_registry = ToolRegistry()


def register_tool(tool: BaseTool) -> BaseTool:
    return default_registry.register(tool)


def get_tool(name: str) -> BaseTool:
    return default_registry.get(name)


def list_tools() -> list[dict[str, str]]:
    return default_registry.list_tools()


def run_tool(name: str, **kwargs: Any) -> AVResult:
    return default_registry.run(name, **kwargs)

