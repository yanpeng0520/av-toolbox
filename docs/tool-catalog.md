# Tool Catalog

`av-toolbox` exposes one registry shared by the Python API, CLI, and web UI.
Use `av-toolbox list-tools` or `av_toolbox.list_tools()` to inspect the live
registry.

## Artifact Contract

Tools return an `AVResult` with declared artifact fields:

```text
tool_name
input_path
output_dir
overlay_path
timeline_json
csv_path
report_html
config_path
log_path
metadata
```

Artifact filenames use the input stem and registry name:

```text
<output_dir>/<input_stem>_<tool_name_with_underscores>_timeline.json
<output_dir>/<input_stem>_<tool_name_with_underscores>_features.csv
<output_dir>/<input_stem>_<tool_name_with_underscores>_report.html
<output_dir>/<input_stem>_<tool_name_with_underscores>_config.yaml
<output_dir>/<input_stem>_<tool_name_with_underscores>_log.txt
<output_dir>/<input_stem>_<tool_name_with_underscores>_overlay.mp4
```

Use `--no-json`, `--no-csv`, `--no-report`, or `--no-overlay` to disable
optional outputs from the CLI. Python callers use `export_json=False`,
`export_csv=False`, `export_report=False`, or `export_overlay=False`.

Overlay MP4s are intended to be browser/player friendly: H.264 video,
`yuv420p` pixel format, AAC audio when audio is present, and fast-start muxing
where feasible.

## Runtime Controls

The shared runtime accepts the same resource controls through Python, CLI, and
the local web UI:

```bash
--device cuda:0
--batch-size 16
--fp16
--cache-dir ~/.cache/av_toolbox
--workspace-dir /tmp/av_toolbox/debug_run
--keep-workspace
```

Python callers pass the same names to `av_toolbox.run_tool(...)`.

## Registered Tools

| Registry name | CLI command | Input | Main outputs | Declared artifacts |
| --- | --- | --- | --- | --- |
| `video.blur_exposure` | `av-toolbox video blur-exposure` | Video | Per-frame blur, luminance, dark-frame, and overexposure samples. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.obstruction` | `av-toolbox video obstruction` | Video | Low-variance bright-frame obstruction events. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.motion` | `av-toolbox video motion` | Video | Frame-to-frame motion intensity samples and motion events. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.optical_flow` | `av-toolbox video optical-flow` | Video | Dense Farneback flow magnitude samples and pixel-motion events. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.foreground_motion` | `av-toolbox video foreground-motion` | Video | Foreground-biased optical-flow samples, with optional YOLO masks. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.camera_shake` | `av-toolbox video camera-shake` | Video | Sparse optical-flow translation jitter and shake events. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.shot_boundary` | `av-toolbox video shot-boundary` | Video | Lightweight sampled histogram shot-boundary events and scene segments. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.cut_detection` | `av-toolbox video cut-detection` | Video | Cut events and segments using TransNetV2, PySceneDetect, or lightweight fallback. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.object_detection` | `av-toolbox video object-detection` | Video | YOLO object boxes and class confidences. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.segmentation` | `av-toolbox video segmentation` | Video | YOLO instance boxes, classes, confidences, and mask area summaries. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.pose` | `av-toolbox video pose` | Video | MediaPipe pose landmark events. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.shot_type` | `av-toolbox video shot-type` | Video | Transformers/BEiT-style per-frame shot-type labels and top-k probabilities. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.action_recognition` | `av-toolbox video action-recognition` | Video | SlowFast/PyTorchVideo window-level action labels. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.st_action` | `av-toolbox video st-action` | Video | MMAction2 spatio-temporal action predictions when config/checkpoint are supplied. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.beat_detection` | `av-toolbox audio beat-detection` | Audio or video with audio | Beats, heuristic downbeats, onsets, tempo, and waveform/timeline overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.energy` | `av-toolbox audio energy` | Audio or video with audio | RMS, dB energy, spectral centroid, zero-crossing, and silence samples. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.event_detection` | `av-toolbox audio event-detection` | Audio or video with audio | Impacts, low/high-energy regions, spectral changes, tonal shifts, and overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.music_phase` | `av-toolbox audio music-phase` | Audio or video with audio | Coarse music phase segments and overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.transcription` | `av-toolbox audio transcription` | Audio or video with audio | faster-whisper speech segments and transcript text. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `av.sync_correspondence` | `av-toolbox av sync-correspondence` | Video with audio | Audio events, visual motion peaks, sync matches, offsets, and overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `av.denseav` | `av-toolbox av denseav` | Video with audio | DenseAV similarity summaries, per-head statistics, attention videos, and optional similarity matrix. | Overlay MP4 when enough frames are sampled, timeline JSON, feature CSV, HTML report, config YAML, log text. DenseAV may also write an additional `_2head_attention.mp4` path listed inside the timeline JSON. |

Model-backed tools keep imports lazy. Install optional extras as needed: `.[transcription]` for Whisper, `.[vision-models]` for YOLO/shot-type, `.[pose]` for MediaPipe, `.[action]` for PyTorchVideo action recognition, `.[cut-detection]` for PySceneDetect fallback support, and `.[denseav]` for DenseAV.

## Generic CLI Form

Every registered tool can run through the generic registry command:

```bash
av-toolbox run video.motion \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/generic_motion \
  --sample-fps 5 \
  --max-seconds 5 \
  --device cpu
```

Category commands are thin wrappers around the same registry call.
