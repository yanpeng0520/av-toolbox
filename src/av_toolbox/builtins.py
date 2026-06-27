"""Built-in tool registration."""

from __future__ import annotations

from av_toolbox.audio import AudioEventDetectionTool, BeatDetectionTool, MusicPhaseTool
from av_toolbox.av import DenseAVTool, SyncCorrespondenceTool
from av_toolbox.core.registry import ToolRegistry, default_registry
from av_toolbox.video import BlurExposureTool, MotionTool, ShotBoundaryTool


def register_builtin_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Register built-in tools once and return the registry."""
    target = registry or default_registry
    if "audio.beat_detection" not in target:
        target.register(BeatDetectionTool())
    if "audio.event_detection" not in target:
        target.register(AudioEventDetectionTool())
    if "audio.music_phase" not in target:
        target.register(MusicPhaseTool())
    if "av.denseav" not in target:
        target.register(DenseAVTool())
    if "av.sync_correspondence" not in target:
        target.register(SyncCorrespondenceTool())
    if "video.blur_exposure" not in target:
        target.register(BlurExposureTool())
    if "video.motion" not in target:
        target.register(MotionTool())
    if "video.shot_boundary" not in target:
        target.register(ShotBoundaryTool())
    return target
