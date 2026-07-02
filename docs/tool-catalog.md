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
| `video.image_quality` | `av-toolbox video image-quality` | Video | Per-frame sharpness/blur, luma/exposure, contrast, and lens-obstruction samples, with a source-video quality overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.motion` | `av-toolbox video motion` | Video | Frame-to-frame motion intensity samples, motion events, and source-video motion overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.optical_flow` | `av-toolbox video optical-flow` | Video | Dense Farneback flow magnitude samples, pixel-motion events, and source-video flow-mask overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.foreground_motion` | `av-toolbox video foreground-motion` | Video | Foreground-biased optical-flow samples, optional YOLO masks, and source-video flow-mask overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.camera_shake` | `av-toolbox video camera-shake` | Video | Sparse optical-flow translation jitter, shake events, and source-video shake overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.cut_detection` | `av-toolbox video cut-detection` | Video | TransNetV2 cut events, scene segments, and source-video cut timeline overlay. PySceneDetect/lightweight remain explicit fallback backends. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.object_detection` | `av-toolbox video object-detection` | Video | YOLO object boxes, class confidences, and source-video box overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.segmentation` | `av-toolbox video segmentation` | Video | YOLO instance boxes, classes, confidences, mask-area summaries, and source-video segment overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.pose` | `av-toolbox video pose` | Video | MediaPipe pose landmark events and source-video pose overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.shot_type` | `av-toolbox video shot-type` | Video | Transformers/BEiT-style per-frame shot-type labels, top-k probabilities, and source-video label overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.action_recognition` | `av-toolbox video action-recognition` | Video | SlowFast/PyTorchVideo window-level action labels and source-video action overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.st_action` | `av-toolbox video st-action` | Video | MMAction2 spatio-temporal action predictions and source-video action overlay when config/checkpoint are supplied. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.beat_detection` | `av-toolbox audio beat-detection` | Audio or video with audio | Beats, heuristic downbeats, onsets, tempo, and waveform/timeline overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.energy` | `av-toolbox audio energy` | Audio or video with audio | RMS, dB energy, spectral centroid, zero-crossing, silence samples, and waveform/energy overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.event_detection` | `av-toolbox audio event-detection` | Audio or video with audio | Impacts, low/high-energy regions, spectral changes, tonal shifts, and overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.music_phase` | `av-toolbox audio music-phase` | Audio or video with audio | Coarse music phase segments and overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.transcription` | `av-toolbox audio transcription` | Audio or video with audio | faster-whisper speech segments and transcript text. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `av.sync_correspondence` | `av-toolbox av sync-correspondence` | Video with audio | Audio events, visual motion peaks, sync matches, offsets, and overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `av.denseav` | `av-toolbox av denseav` | Video with audio | DenseAV similarity summaries, per-head statistics, attention videos, and optional similarity matrix. | Overlay MP4 when enough frames are sampled, timeline JSON, feature CSV, HTML report, config YAML, log text. DenseAV may also write an additional `_2head_attention.mp4` path listed inside the timeline JSON. |

Model-backed tools keep imports lazy. Install optional extras as needed: `.[transcription]` for Whisper, `.[vision-models]` for YOLO/shot-type, `.[pose]` for MediaPipe, `.[action]` for PyTorchVideo action recognition, `.[cut-detection]` for TransNetV2 and PySceneDetect cut detection support, and `.[denseav]` for DenseAV.

## Requirements At A Glance

| Tool | Input | Main output | Optional dependency | GPU/model requirement |
| --- | --- | --- | --- | --- |
| `video.motion` | Video | Motion intensity overlay, timeline, CSV, report | `.[video]` | CPU |
| `video.image_quality` | Video | Sharpness/luma/contrast quality overlay | `.[video]` | CPU |
| `video.blur_exposure` | Video | Blur, exposure, dark/overexposed frame overlay | `.[video]` | CPU |
| `video.obstruction` | Video | Lens/scene obstruction overlay | `.[video]` | CPU |
| `video.optical_flow` | Video | Dense optical-flow mask overlay | `.[video]` | CPU |
| `video.foreground_motion` | Video | Foreground-biased flow overlay | `.[video]`; YOLO modes use `.[vision-models]` | CPU by default; YOLO can use GPU |
| `video.camera_shake` | Video | Camera shake score overlay | `.[video]` | CPU |
| `video.cut_detection` | Video | TransNetV2 cut timeline overlay | `.[cut-detection]` | TransNetV2 model; GPU optional |
| `video.object_detection` | Video | YOLO bounding-box overlay | `.[vision-models]` | YOLO model; GPU recommended |
| `video.segmentation` | Video | YOLO instance-mask overlay | `.[vision-models]` | YOLO model; GPU recommended |
| `video.pose` | Video with people | MediaPipe pose overlay | `.[pose]` | MediaPipe model; CPU/GPU depends install |
| `video.shot_type` | Video | Shot-type label/confidence overlay | `.[vision-models]` | Transformers image model; GPU recommended |
| `video.action_recognition` | Video | SlowFast action-label overlay | `.[action]` | PyTorchVideo model; GPU recommended |
| `video.st_action` | Video | MMAction2 spatio-temporal action overlay | MMAction2/MMCV manual setup | Config + checkpoint required; GPU recommended |
| `audio.beat_detection` | Audio or video with audio | Beat/downbeat/onset waveform overlay | `.[audio]` | CPU |
| `audio.energy` | Audio or video with audio | RMS/dB/silence waveform overlay | `.[audio]` | CPU |
| `audio.event_detection` | Audio or video with audio | Impact/energy/spectral event overlay | `.[audio]` | CPU |
| `audio.music_phase` | Music audio/video | Coarse music phase overlay | `.[audio]` | CPU |
| `audio.transcription` | Speech audio/video | Whisper transcript timeline/artifacts | `.[transcription]` | faster-whisper model; GPU recommended |
| `av.sync_correspondence` | Video with audio | Audio-event to visual-motion sync overlay | `.[video,audio]` | CPU |
| `av.denseav` | Video with audio | DenseAV attention/similarity overlays | `.[denseav]` | DenseAV checkpoint; GPU recommended |

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
