"""Audio analysis tools."""

from av_toolbox.audio.event_detection import AudioEventDetectionTool
from av_toolbox.audio.beat_detection import BeatDetectionTool
from av_toolbox.audio.music_phase import MusicPhaseTool

__all__ = ["AudioEventDetectionTool", "BeatDetectionTool", "MusicPhaseTool"]
