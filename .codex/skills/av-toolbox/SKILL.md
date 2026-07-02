---
name: av-toolbox
description: Build and maintain the local AV Toolbox Python project at /home/yanp/projects/av_toolbox. Use when Codex works on av_toolbox or av-toolbox tasks involving video, audio, audio-visual analysis tools, generated overlay MP4s that must preserve full source duration and source audio, tool registry/API/CLI/web integration, media metadata, FFmpeg exports, overlays, timeline JSON/CSV outputs, tests, docs, Docker, DenseAV, or migrating older AV analysis code into the installable package.
---

# AV Toolbox

## Overview

Use this skill to make focused, testable changes in the local AV Toolbox repo. Treat it as an installable Python package whose public surfaces are the Python API, `av-toolbox` CLI, and local web UI over the same backend registry.

## Project Facts

- Repo: `/home/yanp/projects/av_toolbox`
- Package: `src/av_toolbox`
- CLI: `av-toolbox`
- Default demo video: `/home/yanp/projects/av_toolbox/data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4`
- Current foundation includes `AVResult`, `BaseTool`, `ToolRegistry`, built-in category modules, CLI commands, demo media, and tests.

## Core Workflow

1. Inspect the repo before coding. Use `rg --files`, targeted `rg`, and existing tests to understand the current structure and avoid duplicate abstractions.
2. Keep changes small. Preserve existing package layout and avoid broad refactors unless the user asks for one.
3. Put reusable logic in `src/av_toolbox/core/`; put tools in `src/av_toolbox/video/`, `src/av_toolbox/audio/`, or `src/av_toolbox/av/`.
4. Route Python API, CLI, and web UI through the registry and `av_toolbox.run_tool`; do not duplicate algorithm logic in UI or CLI code.
5. Return `AVResult` and predictable artifact paths for user-facing outputs: overlay MP4, timeline JSON, feature CSV, report HTML, config YAML, and log text when applicable.
6. Add or update tests for changed public behavior, especially registry names, CLI commands, JSON schemas, artifact naming, and media export properties.
7. Validate with the narrowest meaningful commands, then broaden to `pytest` or CLI smoke tests when shared behavior changed.

## Standards

- Use dot-style tool names such as `video.blur_exposure`, `video.cut_detection`, `video.motion`, `audio.beat_detection`, `audio.music_phase`, `audio.event_detection`, `av.denseav`, and `av.sync_correspondence`.
- Accept video paths and audio paths where appropriate; extract audio internally for audio tools that receive video.
- Use seconds as the standard time unit in JSON/CSV outputs.
- Keep heavy dependencies optional. DenseAV and torch-backed behavior should fail gracefully or skip tests when optional dependencies are missing.
- Use FFmpeg/ffprobe for robust user-facing media export checks. Preview MP4s should be H.264, yuv420p, AAC when audio is present, and fast-start when feasible.
- Make overlays readable at preview sizes and reuse shared rendering helpers before introducing per-tool rendering code.
- Generated video overlays for demos/reviews must cover the whole source duration and preserve source audio when the input has audio. Never hand over silent or shortened MP4s unless the user explicitly asks for muted or clipped output.
- Video-analysis overlay style contract ("dark-slate" house style): keep the raw video unobstructed except for a small translucent top-left card/badge; put a dark-slate timeline panel under the video; draw a single amber vertical playhead only inside the panel; use thin line charts (no gridlines, no filled/shaded areas) for time-varying values with the threshold as a short axis tick. Scalar status values (shakiness, blur, luma, contrast, obstruction, confidence) show compact threshold-aware bars: green while inside/approaching the threshold, red when the value crosses/fails it. Each tool owns a dedicated `_render_<tool>_overlay`; do not restyle the shared `source_overlay.render_source_video_overlay` (box/pose/mask tools still use it). Reference tools: `video.camera_shake`, `video.image_quality`, `video.foreground_motion`, `video.cut_detection`. See the playbook's "Video Analysis Overlay Style Contract" for the exact palette and layout.
- Verify media details before blaming models: input existence, FPS, duration, sample rate, channels, timestamps, tensor shapes, and output alignment.

## Detailed Playbook

Read [references/av-toolbox-playbook.md](references/av-toolbox-playbook.md) when a task touches architecture, a new tool, CLI/web behavior, output formats, migration, Docker/CI/docs, generated overlay MP4s, or unclear AV analysis workflow choices. The reference adapts the pasted AGENT_SKILLS-style guidance to the current `av_toolbox` repo.

## Useful Commands

```bash
cd /home/yanp/projects/av_toolbox
rg --files | sort
python -c "import av_toolbox; print(av_toolbox.__file__)"
python -c "import av_toolbox; print(av_toolbox.list_tools())"
av-toolbox list-tools
pytest
```

For a focused smoke test, prefer a command that matches the changed area, for example:

```bash
av-toolbox video blur-exposure \
  /home/yanp/projects/av_toolbox/data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output /tmp/av_toolbox_test/video_blur
```

## Completion Report

When finishing AV Toolbox work, report changed files, new files, tests added, commands run, whether tests passed, whether CLI smoke tests passed when relevant, known limitations, and the next recommended step.
