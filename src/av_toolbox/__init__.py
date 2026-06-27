"""Public API for av-toolbox."""

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.cache import ModelCache
from av_toolbox.core.hardware import HardwareConfig
from av_toolbox.core.registry import (
    ToolRegistry,
    default_registry,
    get_tool,
    list_tools,
    register_tool,
    run_tool,
)
from av_toolbox.core.result import AVResult
from av_toolbox.core.workspace import WorkspaceManager

from av_toolbox.builtins import register_builtin_tools

register_builtin_tools()

__all__ = [
    "AVResult",
    "BaseTool",
    "HardwareConfig",
    "ModelCache",
    "ToolRegistry",
    "ToolRunContext",
    "WorkspaceManager",
    "default_registry",
    "get_tool",
    "list_tools",
    "register_builtin_tools",
    "register_tool",
    "run_tool",
]
