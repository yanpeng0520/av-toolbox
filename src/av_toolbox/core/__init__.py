"""Core abstractions shared by all av-toolbox tools."""

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.cache import ModelCache
from av_toolbox.core.hardware import HardwareConfig
from av_toolbox.core.media_io import VideoMetadata, read_video_metadata
from av_toolbox.core.outputs import ArtifactPaths, make_artifact_paths
from av_toolbox.core.registry import ToolRegistry
from av_toolbox.core.result import AVResult
from av_toolbox.core.workspace import WorkspaceManager

__all__ = [
    "AVResult",
    "BaseTool",
    "HardwareConfig",
    "ModelCache",
    "ArtifactPaths",
    "ToolRegistry",
    "ToolRunContext",
    "VideoMetadata",
    "WorkspaceManager",
    "make_artifact_paths",
    "read_video_metadata",
]
