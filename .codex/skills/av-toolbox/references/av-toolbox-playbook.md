# AV Toolbox Playbook

This reference adapts the provided AGENT_SKILLS-style guidance for `/home/yanp/projects/av_toolbox`, whose package is `av_toolbox` and CLI is `av-toolbox`.

## 1. Repo Inspection

Purpose: understand the existing repository before making changes.

Use before coding, especially when moving older AV analysis code into the package.

Actions:

- Inspect the repo tree with targeted commands.
- Identify current package structure, scripts, tests, configs, data, notebooks, and demos.
- Check `pyproject.toml`, README, CLI entry points, optional dependencies, and tests.
- Identify conflicts with any production or previously installed `av_tools`/`av_toolbox` packages.
- Summarize what should be preserved, moved, wrapped, or removed.

Useful commands:

```bash
cd /home/yanp/projects/av_toolbox
rg --files | sort
rg "import av_toolbox|from av_toolbox" -n .
rg "import av_tools|from av_tools" -n .
```

Output a short diagnosis before implementation when the task is broad or migration-shaped.

## 2. Package Structure

Preferred layout:

```text
av_toolbox/
  pyproject.toml
  README.md
  src/
    av_toolbox/
      __init__.py
      core/
      video/
      audio/
      av/
      web.py
      web_app.py
      cli.py
  tests/
  docs/
  data_segments/
  outputs/
```

Rules:

- Prefer the existing `src/av_toolbox/` layout.
- Keep reusable logic in `src/av_toolbox/core/`.
- Keep category-specific tools in `video/`, `audio/`, and `av/`.
- Keep CLI and web thin; both should call the same backend API/registry.
- Do not scatter reusable tool code into scripts or notebooks.

Acceptance checks:

```bash
python -c "import av_toolbox; print(av_toolbox.__file__)"
pytest
```

## 3. Unified Tool Interface

Every analysis tool should behave consistently:

```python
import av_toolbox

result = av_toolbox.run_tool(
    "audio.beat_detection",
    input_path="input.mp4",
    output_dir="outputs/",
    export_overlay=True,
    export_json=True,
    export_csv=True,
)
```

Use the shared `AVResult` shape from `src/av_toolbox/core/result.py` for artifacts:

```text
tool_name, input_path, output_dir, overlay_path, timeline_json,
csv_path, report_html, config_path, log_path, metadata
```

Rules:

- Do not invent separate return formats per tool.
- Keep output paths predictable with `core.outputs.make_artifact_paths` where possible.
- Add or update tests when changing result behavior.

## 4. Tool Registry

Python API, CLI, and web UI should use one central registry.

Expected public API:

```python
av_toolbox.list_tools()
av_toolbox.get_tool("video.blur_exposure")
av_toolbox.run_tool("video.blur_exposure", input_path="input.mp4")
```

Current registry names include:

```text
video.blur_exposure
video.cut_detection
video.motion

audio.beat_detection
audio.music_phase
audio.event_detection

av.denseav
av.sync_correspondence
```

Rules:

- Register tools in one central registry.
- Let CLI and web discover tools from the registry.
- Avoid hardcoding separate tool lists in multiple places.

Acceptance checks:

```bash
python -c "import av_toolbox; print(av_toolbox.list_tools())"
av-toolbox list-tools
```

## 5. Media Metadata

Extract reliable media information before analysis.

Video metadata should include duration, FPS, width, height, frame count, codec/pixel format when available, audio presence, audio sample rate, and audio channels when relevant.

Audio metadata should include duration, sample rate, channels, and codec when available.

Recommended utilities:

```python
get_media_info(path)
get_video_info(path)
get_audio_info(path)
has_audio_stream(path)
```

Rules:

- Use ffprobe for robust stream metadata when codec/audio fields matter.
- Validate that input files exist.
- Give useful errors when media cannot be read.
- Preserve existing OpenCV helpers where they are sufficient for lightweight video sampling.

## 6. Standard MP4 Export

User-facing overlay videos should open in VS Code, browsers, and normal players.

Standard preview format:

```text
Container: MP4
Video codec: H.264 / libx264
Pixel format: yuv420p
Audio codec: AAC when audio is present
Fast start: enabled
```

Recommended ffmpeg pattern:

```bash
ffmpeg -y -i input.mp4 \
  -vf "format=yuv420p" \
  -c:v libx264 \
  -preset medium \
  -crf 18 \
  -pix_fmt yuv420p \
  -c:a aac \
  -b:a 128k \
  -movflags +faststart \
  output_preview.mp4
```

Validate video outputs with:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt \
  -of default=noprint_wrappers=1 \
  output_preview.mp4
```

Expected:

```text
codec_name=h264
pix_fmt=yuv420p
```

## 7. Audio Extraction

Audio tools should analyze `.wav` files and video files.

Recommended utility:

```python
extract_audio(input_path, output_wav_path, sample_rate=44100, mono=True)
```

Recommended ffmpeg pattern:

```bash
ffmpeg -y -i input.mp4 -vn -ac 1 -ar 44100 output.wav
```

Rules:

- Extract audio internally when an audio tool receives video.
- Store extracted audio in the run output directory or workspace/temp directory.
- Clean up temporary files unless `keep_workspace` is requested.

## 8. Overlay Rendering

Overlays are for debugging and demos, so they must be readable.

Common overlays:

- Video: shot boundaries, blur/exposure scores, motion vectors, camera shake, visual quality warnings.
- Audio: beats, downbeats, onsets, energy curves, music phase regions, audio event labels.
- Audio-visual: DenseAV heatmaps, sync confidence curves, correspondence scores, active performer highlights.

Rules:

- Use consistent visual style across tools.
- Keep overlays readable on small preview videos.
- Add timestamp labels when helpful.
- Reuse shared visualization/export helpers rather than each tool inventing rendering logic.

### Video Overlay Audio and Duration Contract

Use this contract whenever generating, regenerating, or reviewing demo/review overlay MP4s:

- Render the complete analyzed source duration by default. Do not shorten outputs for convenience unless the user explicitly asks for a clip.
- Preserve original source audio when the input has audio. Video-only OpenCV renders are temporary intermediates, not final user-facing artifacts.
- Mux source audio with optional audio mapping so silent inputs still succeed:

```bash
ffmpeg -y -loglevel error \
  -i overlay_video_only.mp4 \
  -i source_input.mp4 \
  -map 0:v:0 -map 1:a? \
  -c:v libx264 -preset veryfast -crf 23 \
  -pix_fmt yuv420p -movflags +faststart \
  -c:a aac -b:a 192k -shortest \
  overlay.mp4
```

- Follow the existing audio-muxing patterns in `src/av_toolbox/audio/overlay.py` and `src/av_toolbox/av/overlay.py`.
- Prefer centralizing this behavior in shared helpers such as `src/av_toolbox/video/source_overlay.py` instead of copying FFmpeg calls into every video tool.

### Video Analysis Overlay Style Contract

Apply this "dark-slate" house style to all `video.*` analysis overlay MP4s unless the user explicitly asks for a different visualization. Reference implementations: `video.camera_shake`, `video.image_quality`, `video.action_recognition`, `video.foreground_motion`, `video.optical_flow`, `video.motion`, and `video.cut_detection`.

Structure:

- Match the source/analyzed duration and preserve source audio (see the audio/duration contract above).
- Keep the video area clean: only a small translucent top-left card/badge (current value + status). Never draw the timeline or the playhead across the video image.
- Put a timeline panel under the video (`panel_h ~= 150`; `~132` for event-only tools). Header row = tool title left, optional legend/caption, `t / duration` right — none overlapping the plot.
- One amber vertical playhead, confined to the panel.

Dark-slate theme (BGR for OpenCV):

- `letterbox=(20,18,16)`, `panel_bg=(40,33,28)`, `text_hi=(243,240,238)`, `text_lo=(170,158,150)`, `axis=(86,74,66)`.
- ok/green `(150,200,96)`, alert/red `(92,96,240)`, idle/grey `(120,128,132)`, playhead amber `(90,205,255)`.
- Multi-metric hues: sharp/green `(150,200,96)`, luma/amber `(70,185,240)`, contrast/blue `(240,170,120)`.
- Never use the old pale mint panel `(238,248,245)` — it reads yellowish.

Timeline:

- Continuous values → thin (2px) anti-aliased line charts, baseline pinned at 0, self-normalized per metric. NO gridlines and NO filled/shaded areas.
- Show the threshold as a short axis tick (not a full-width rule); color the line green/red by pass/fail or state where meaningful.
- Multi-metric tools draw one colored line per metric plus a header legend (`video.image_quality`).
- Put the current reading as a small dot + value at the playhead.

Top-left card:

- Translucent dark card (`cv2.addWeighted` ~0.6) with a header, current value(s), and threshold-aware bars (green good / red failing). Draw label text with a dark outline (thick dark pass, then thin light pass) so it stays legible over any footage.

Event tools (cut detection):

- No scalar line: draw a numbered shot-segment lane (subtle alternating tints, current shot outlined), red vertical ticks at cut boundaries, and flash `CUT` on the badge near a boundary.

Implementation pattern:

- Each tool owns a dedicated `_render_<tool>_overlay` in its own module. Do NOT restyle the shared `render_source_video_overlay` in `source_overlay.py` — the box/pose/mask/segmentation tools still depend on its look. Reuse only `mux_video_overlay_with_source_audio` for audio.
- Re-render fast from the tool's timeline JSON (no re-detection) when only the overlay changed.
- Before handoff, verify the overlay length, audio stream, bottom timeline panel, and playhead placement with ffprobe plus at least one representative frame/contact sheet.

- Validate before handoff in three passes:
  1. Structural: every MP4 decodes, has H.264/yuv420p video, and duration matches the source within 0.25 seconds.
  2. Content: inspect representative frames/contact sheets so overlays, labels, masks, timelines, and markers are visibly present.
  3. Audio: when the source has audio, every final overlay MP4 has an audio stream and remains full-length.
- Never stage generated MP4/JPG/PNG/WAV artifacts unless the user explicitly asks and they are covered by Git LFS policy.

## 9. Timeline JSON

Every tool should export machine-readable results when requested.

Example:

```json
{
  "tool": "audio.beat_detection",
  "input_path": "input.mp4",
  "duration": 30.0,
  "events": [
    {"time": 1.24, "type": "beat", "confidence": 0.91}
  ],
  "metadata": {"sample_rate": 44100, "fps": 25.0}
}
```

Rules:

- Use seconds as the standard time unit.
- Use stable field names.
- Include tool name and input path.
- Include confidence when available.
- Keep JSON easy to parse.

Acceptance check:

```bash
python -m json.tool outputs/*timeline*.json
```

## 10. CSV Feature Export

CSV exports should be stable and research-friendly.

Example columns:

```text
Video: time,frame_idx,blur_score,brightness,exposure_score,motion_score,quality_score
Audio: time,rms_energy,spectral_flux,onset_strength,beat_confidence
AV: time,av_similarity,sync_score,correspondence_score
```

Rules:

- Use seconds for `time`.
- Keep column names stable.
- Avoid storing large tensors in CSV.
- Store large arrays separately if needed.

## 11. Video Analysis Tools

Category: `src/av_toolbox/video/`.

Current or expected tools:

```text
video.blur_exposure
video.cut_detection
video.motion
video.visual_quality
```

Rules:

- Start with simple, reliable algorithms before deep models.
- Use the default demo video for smoke tests.
- Output overlay, timeline JSON, and/or CSV when relevant.
- Keep algorithm code separate from CLI/web code.

Example:

```bash
av-toolbox video blur-exposure \
  /home/yanp/projects/av_toolbox/data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/video_blur
```

## 12. Audio Analysis Tools

Category: `src/av_toolbox/audio/`.

Current or expected tools:

```text
audio.beat_detection
audio.music_phase
audio.event_detection
audio.energy_flux
```

Rules:

- Accept both audio files and video files.
- Extract audio first when input is video.
- Export event/phase/beat timelines in JSON.
- Generate overlay video for video input when requested.
- Generate waveform/timeline report for audio-only input when useful.

Example:

```bash
av-toolbox audio beat-detection \
  /home/yanp/projects/av_toolbox/data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/audio_beat
```

## 13. Music Phase Detection

Coarse phase labels may include:

```text
intro
verse
chorus
bridge
drop
outro
main_section
unknown
```

Rules:

- Start with simple segmentation if no model is available.
- Do not overclaim accuracy.
- Label uncertain regions as `unknown` or `main_section`.
- Make visualization clear.

## 14. Audio-Visual Tools

Category: `src/av_toolbox/av/`.

Current or expected tools:

```text
av.denseav
av.sync_correspondence
av.audio_visual_sync
av.audio_visual_correspondence
```

Rules:

- Keep heavy dependencies optional.
- If DenseAV is unavailable, fail gracefully with a clear message or skip tests.
- Provide lightweight fallbacks where possible.
- Export timeline scores even when heatmap rendering is unavailable.
- Do not make DenseAV required for basic package installation.

Optional install examples:

```bash
pip install -e ".[av,torch]"
pip install -e ".[denseav]"
```

## 15. CLI

CLI entry point: `av-toolbox`.

Expected commands include:

```bash
av-toolbox list-tools
av-toolbox info video.blur_exposure
av-toolbox run video.blur_exposure input.mp4 --output outputs/
av-toolbox video blur-exposure input.mp4 --output outputs/
av-toolbox video cut-detection input.mp4 --output outputs/
av-toolbox video motion input.mp4 --output outputs/
av-toolbox audio beat-detection input.mp4 --output outputs/
av-toolbox audio music-phase input.mp4 --output outputs/
av-toolbox audio event-detection input.mp4 --output outputs/
av-toolbox av denseav input.mp4 --output outputs/
av-toolbox av sync-correspondence input.mp4 --output outputs/
av-toolbox serve
```

Rules:

- CLI must call `av_toolbox.run_tool` or registry APIs.
- Do not duplicate algorithm logic in CLI.
- Print output paths or result JSON after successful runs.
- Return non-zero exit codes on failure.
- Provide helpful errors.

## 16. Local Web App

Preferred command:

```bash
av-toolbox serve
```

User flow:

```text
Upload video/audio -> select category -> select tool -> configure parameters -> run analysis -> preview overlay -> download MP4/JSON/CSV/report
```

Rules:

- Keep the web app thin.
- Use the same registry as CLI/Python API.
- Do not place analysis logic in Streamlit callbacks.
- Store uploads in a safe temp/workspace directory.
- Show errors clearly.

## 17. Testing

Required or useful tests:

```text
test_import.py
test_registry.py
test_demo_video.py
test_media_io.py
test_cli.py
test_blur_exposure_tool.py
test_motion_tool.py
test_ported_tools.py
test_audio_tools.py
test_denseav_tool.py
test_output_json_schema.py
test_output_video_codec.py
```

Smoke commands:

```bash
pytest
av-toolbox list-tools
av-toolbox video blur-exposure \
  /home/yanp/projects/av_toolbox/data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output /tmp/av_toolbox_test/video_blur
av-toolbox audio beat-detection \
  /home/yanp/projects/av_toolbox/data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output /tmp/av_toolbox_test/audio_beat
```

Rules:

- Every new tool gets at least one smoke test.
- Tests should not require multiple synchronized videos unless the tool truly needs them.
- DenseAV tests should skip if optional dependencies are missing.
- Avoid huge external downloads in default tests.

## 18. CI / GitHub Actions

CI should install the package, run tests, run configured lint/type checks, run a CLI smoke test, validate JSON output, and validate MP4 output codec when feasible.

Rules:

- Do not enable PyPI publishing by default.
- Keep CI lightweight.
- Avoid GPU-only tests in default CI.
- Use demo media if available or generate tiny synthetic fallback media.

## 19. Docker

Docker should support:

```bash
docker compose up
docker run -p 8501:8501 av-toolbox
av-toolbox list-tools
av-toolbox serve
```

Rules:

- Include FFmpeg in the image.
- Keep CPU as the default image.
- Put CUDA/GPU behavior in a separate image or optional path later.
- Add smoke tests in Docker build or CI when feasible.

## 20. Documentation

README/docs should include project purpose, one-video quickstart, install instructions, Python API example, CLI example, local web example, tool categories, expected outputs, and troubleshooting.

Rules:

- Update docs when adding public commands.
- Prefer practical examples over long theory.
- Use the default demo video in examples.
- Include expected output paths.

## 21. AI-Agent Task Decomposition

Good task shape:

```text
Task:
Implement export_preview_mp4 in src/av_toolbox/core/export.py.

Requirements:
- Input: any video path.
- Output: MP4 with H.264 video, yuv420p pixel format, AAC audio if source has audio.
- Use ffmpeg.
- Add ffprobe validation helper.
- Add pytest using Clever_Cat_Outsmarts_Warrior_square.mp4.
- Do not modify unrelated files.

Acceptance:
- pytest passes.
- ffprobe shows codec_name=h264 and pix_fmt=yuv420p.
```

Avoid vague tasks such as "Make the AV tools better."

Rules:

- Prefer one small task at a time.
- Add tests with each feature.
- Summarize changed files.
- Do not refactor unrelated modules.
- Preserve production compatibility.

## 22. Migration

When moving useful existing functionality into `av_toolbox`:

1. Inspect existing code.
2. Identify reusable modules.
3. Classify each module as copy directly, wrap into `BaseTool`, rename by category, leave untouched, or remove/deprecate.
4. Move incrementally.
5. Add tests after each move.
6. Verify local package imports correctly.
7. Check for import conflicts.

Rules:

- Do not move everything blindly.
- Keep migration commits small.
- Prefer wrappers before deep rewrites.
- Make the local package work independently.

## 23. Debugging And Visualization

For model or algorithm output, inspect:

```text
input media
extracted frames
extracted audio
intermediate features
timeline JSON
overlay video
failure cases
```

Rules:

- Start simple before using heavy models.
- Visualize results early.
- Verify data before blaming the model.
- Check shapes, timestamps, FPS, sample rate, channel count, and alignment.
- Do not trust a metric without watching or inspecting the output artifact.

## 24. Recommended Foundation

Prioritize these foundations before larger features:

```text
1. get_media_info()
2. extract_audio()
3. export_preview_mp4()
4. validate_output_video()
5. write_timeline_json()
6. write_features_csv()
7. render_basic_timeline_overlay()
8. AVResult
9. BaseTool
10. ToolRegistry
11. av-toolbox list-tools
12. one smoke test using Clever_Cat_Outsmarts_Warrior_square.mp4
```

Several items already exist in the current repo. Extend or harden them rather than recreating them.

## 25. Completion Checklist

Before finishing a task, report:

```text
Changed files
New files
Tests added
Commands run
Whether pytest passed
Whether CLI smoke test passed
Known limitations
Next recommended step
```

Minimum validation:

```bash
pytest
av-toolbox list-tools
```

For video outputs, validate codec/pixel format:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt \
  -of default=noprint_wrappers=1 \
  output.mp4
```

Expected:

```text
codec_name=h264
pix_fmt=yuv420p
```
