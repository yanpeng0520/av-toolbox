"""Built-in tool registration."""

from __future__ import annotations

from av_toolbox.audio import (
    AudioEnergyTool,
    AudioEventDetectionTool,
    BeatDetectionTool,
    MusicPhaseTool,
    TranscriptionTool,
)
from av_toolbox.av import DenseAVTool, SyncCorrespondenceTool
from av_toolbox.core.registry import ToolRegistry, default_registry
from av_toolbox.video import (
    ActionRecognitionTool,
    CameraShakeTool,
    CutDetectionTool,
    ForegroundMotionTool,
    ImageQualityTool,
    MotionTool,
    ObjectDetectionTool,
    OpticalFlowTool,
    PoseTool,
    SegmentationTool,
    ShotTypeTool,
    STActionTool,
)


def register_builtin_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Register built-in tools once and return the registry."""
    target = registry or default_registry
    tools = [
        BeatDetectionTool(),
        AudioEnergyTool(),
        AudioEventDetectionTool(),
        MusicPhaseTool(),
        TranscriptionTool(),
        DenseAVTool(),
        SyncCorrespondenceTool(),
        ImageQualityTool(),
        CameraShakeTool(),
        CutDetectionTool(),
        ForegroundMotionTool(),
        MotionTool(),
        ObjectDetectionTool(),
        OpticalFlowTool(),
        PoseTool(),
        SegmentationTool(),
        ShotTypeTool(),
        ActionRecognitionTool(),
        STActionTool(),
    ]
    for tool in tools:
        if tool.name not in target:
            target.register(tool)
    return target
