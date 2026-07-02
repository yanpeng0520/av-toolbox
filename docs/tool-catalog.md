# Tool Catalog

`av-toolbox` exposes one registry shared by the Python API, CLI, and web UI.
Pick a registry name from the table below, then run it from the CLI, Python, or
Streamlit UI.

## Capability Matrix

Tools are grouped by modality - which input streams they consume:

- **Video** - frame data only.
- **Audio** - waveform data. Video files are accepted when they contain audio.
- **Audio-visual** - aligned video and audio streams.

| Modality | Tool | Granularity | GPU helpful? | Output kind |
| --- | --- | --- | --- | --- |
| Video | `video.motion` | sampled frame pair | no | motion scores + events + overlay |
| Video | `video.image_quality` | sampled frame | no | blur/luma/contrast/obstruction scores + overlay |
| Video | `video.optical_flow` | sampled frame pair | no | dense flow magnitude + pixel-motion events + overlay |
| Video | `video.foreground_motion` | sampled frame pair | yes for YOLO masks | foreground flow scores + overlay |
| Video | `video.camera_shake` | rolling sparse-flow window | no | shake score + shake events + overlay |
| Video | `video.cut_detection` | frame/scene segment | optional | cuts + scene segments + overlay |
| Video | `video.object_detection` | sampled frame | yes | YOLO boxes/classes + overlay |
| Video | `video.segmentation` | sampled frame | yes | YOLO masks/classes + overlay |
| Video | `video.pose` | sampled frame | optional | human pose landmarks + overlay |
| Video | `video.shot_type` | sampled frame | yes | shot labels/top-k probabilities + overlay |
| Video | `video.action_recognition` | video window | yes | action labels/top-k probabilities + overlay |
| Audio | `audio.beat_detection` | audio segment/event | no | beats/downbeats/onsets + overlay |
| Audio | `audio.energy` | audio frame/window | no | RMS/dB/spectral features + overlay |
| Audio | `audio.event_detection` | audio event/region | no | impacts/energy/spectral/tonal events + overlay |
| Audio | `audio.music_phase` | music phrase/segment | no | coarse phase segments + overlay |
| Audio | `audio.transcription` | speech segment | yes | transcript timeline + report |
| Audio-visual | `av.denseav` | sampled AV segment | strongly recommended | DenseAV attention/similarity artifacts + overlay |

## Use A Tool

List tools:

```bash
av-toolbox list-tools
```

Run any tool by registry name:

```bash
av-toolbox run video.motion input.mp4 --output outputs/motion_demo
```

Or use the category shortcut shown in the table:

```bash
av-toolbox video motion input.mp4 --output outputs/motion_demo
```

Use Python:

```python
from av_toolbox import run_tool

result = run_tool("video.motion", input_path="input.mp4", output_dir="outputs/motion_demo")
print(result.overlay_path)
```

Use the UI:

```bash
av-toolbox serve --host 127.0.0.1 --port 8501 --output-root outputs/web_runs
```

Common limits and runtime flags:

```bash
--sample-fps 5 --max-seconds 10 --device cpu --no-overlay
```

## Configuration

For a normal run, set only three things:

| What | CLI argument | Example |
| --- | --- | --- |
| Tool | registry name or category command | `video.motion` or `av-toolbox video motion` |
| Input media | positional input path | `input.mp4` |
| Output folder | `--output` | `outputs/motion_demo` |

Most demo/local runs add one or two limits so results are quick:

| Knob | When to set it | Example |
| --- | --- | --- |
| `--max-seconds` | Limit analyzed duration | `--max-seconds 10` |
| `--sample-fps` | Limit video frame sampling | `--sample-fps 5` |
| `--device` | Choose CPU/GPU for model-backed tools | `--device cuda:0` |
| `--cache-dir` | Put model weights somewhere writable/shared | `--cache-dir /mnt/models/av_toolbox_cache` |

YOLO-backed tools (`video.object_detection`, `video.segmentation`,
`video.pose`, and YOLO foreground masks) preflight torchvision CUDA NMS before
using GPU. With `--device auto`, they fall back to CPU if CUDA is visible but
NMS cannot run on this GPU; the generated config records the actual YOLO device.
Use explicit `--device cuda` only after installing compatible PyTorch and
torchvision CUDA wheels for the machine.

Advanced parameters are tool-specific. Use them only when the selected tool needs
one: `--model-name`, `--checkpoint`, `--config-path`, `--labels-path`,
`--threshold`, `--confidence`, `--image-size`, `--top-k`, `--mask-mode`,
`--language`, or audio/windowing options.

## Per-Tool Parameters

Use these three places when tuning a specific tool:

| Need | Where to look |
| --- | --- |
| Current defaults | Source file linked below; each tool's `_run(...)` signature is the source of truth. |
| CLI flags | `av-toolbox run <tool> --help` or `av-toolbox <category> <tool> --help` for the common CLI-exposed subset. |
| Full advanced control | Python `run_tool(...)`, the local UI advanced controls, and each run's generated config YAML. |

| Tool | Source defaults | Main parameters to tune |
| --- | --- | --- |
| [`video.motion`](#video-motion) | [motion.py](../src/av_toolbox/video/motion.py) | `sample_fps`, `max_seconds`, `threshold`, `active_pct_threshold`, `downscale_width`, `overlay_fps` |
| [`video.image_quality`](#video-image-quality) | [image_quality.py](../src/av_toolbox/video/image_quality.py) | `sample_fps`, `max_seconds`, `blur_threshold`, `dark_threshold`, `overexposed_threshold`, `obstruction_threshold` |
| [`video.optical_flow`](#video-optical-flow) | [optical_flow.py](../src/av_toolbox/video/optical_flow.py) | `sample_fps`, `max_seconds`, `active_threshold_px`, `event_threshold_px`, `active_pct_threshold`, `overlay_alpha` |
| [`video.foreground_motion`](#video-foreground-motion) | [foreground_motion.py](../src/av_toolbox/video/foreground_motion.py) | `sample_fps`, `max_seconds`, `mask_mode`, `model_name`, `confidence`, `active_threshold_px`, `event_threshold_px` |
| [`video.camera_shake`](#video-camera-shake) | [camera_shake.py](../src/av_toolbox/video/camera_shake.py) | `sample_fps`, `max_seconds`, `threshold`, `window`, `min_features`, `max_features`, `redetect_interval` |
| [`video.cut_detection`](#video-cut-detection) | [cut_detection.py](../src/av_toolbox/video/cut_detection.py) | `backend`, `threshold`, `weights_path`, `min_distance_frames`, `scenedetect_threshold`, `max_seconds` |
| [`video.object_detection`](#video-object-detection) | [object_detection.py](../src/av_toolbox/video/object_detection.py) | `sample_fps`, `max_seconds`, `model_name`, `confidence`, `image_size` |
| [`video.segmentation`](#video-segmentation) | [segmentation.py](../src/av_toolbox/video/segmentation.py) | `sample_fps`, `max_seconds`, `model_name`, `confidence`, `image_size` |
| [`video.pose`](#video-pose) | [pose.py](../src/av_toolbox/video/pose.py) | `sample_fps`, `max_seconds`, `model_name`, `confidence`, `keypoint_threshold` |
| [`video.shot_type`](#video-shot-type) | [shot_type.py](../src/av_toolbox/video/shot_type.py) | `sample_fps`, `max_seconds`, `model_name`, `downscale_width`, `top_k`, `offline` |
| [`video.action_recognition`](#video-action-recognition) | [action_recognition.py](../src/av_toolbox/video/action_recognition.py) | `model_name`, `max_seconds`, `window_seconds`, `step_seconds`, `top_k`, `confidence`, `labels_path` |
| [`audio.beat_detection`](#audio-beat-detection) | [beat_detection.py](../src/av_toolbox/audio/beat_detection.py) | `sample_rate`, `hop_length`, `max_seconds`, `window_sec`, `overlay_fps` |
| [`audio.energy`](#audio-energy) | [energy.py](../src/av_toolbox/audio/energy.py) | `sample_rate`, `hop_length`, `frame_length`, `rms_floor_db`, `silence_threshold_db`, `window_sec` |
| [`audio.event_detection`](#audio-event-detection) | [event_detection.py](../src/av_toolbox/audio/event_detection.py) | `impact_delta`, `spectral_delta`, `tonal_delta`, `silence_threshold`, `high_energy_quantile`, `min_region_seconds` |
| [`audio.music_phase`](#audio-music-phase) | [music_phase.py](../src/av_toolbox/audio/music_phase.py) | `sample_rate`, `hop_length`, `min_phase_seconds`, `phrase_bars`, `window_sec` |
| [`audio.transcription`](#audio-transcription) | [transcription.py](../src/av_toolbox/audio/transcription.py) | `model_name`, `language`, `beam_size`, `vad_filter`, `compute_type`, `max_seconds` |
| [`av.denseav`](#av-denseav) | [denseav.py](../src/av_toolbox/av/denseav.py) | `model_name`, `checkpoint`, `sample_fps`, `load_size`, `plot_size`, `max_seconds`, `include_sim_matrix` |

## Tool Instructions

All runnable examples below use placeholder media paths. Replace `input.mp4` or
`input.wav` with your file. Use `--max-seconds` for quick tests, and keep output
folders under `outputs/` or another writable directory.

<a id="video-motion"></a>
### video.motion

**Purpose:** Estimate how much frame-to-frame motion is present in a video.

**Technique:** Sample frames, resize them, convert to grayscale, compute absolute
frame differences, and count pixels whose difference exceeds `threshold`.

**Granularity:** Sampled frame pair.

**Output:** Rows with `mean_diff`, `max_diff`, `active_pct`, and `is_motion`; motion events; overlay MP4; timeline JSON; feature CSV; HTML report; config YAML; log text.

<details><summary>CLI</summary>

```bash
av-toolbox video motion input.mp4 \
  --output outputs/video_motion \
  --sample-fps 5 \
  --max-seconds 10
```

Use this first when you want a fast sanity check for cuts, action, camera moves,
or activity density. Lower `--sample-fps` for speed; raise it when short motion
bursts matter.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.motion",
    input_path="input.mp4",
    output_dir="outputs/video_motion",
    sample_fps=5,
    max_seconds=10,
)
print(result.overlay_path)
print(result.metadata["summary"])
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[video]`.
- Useful knobs: `sample_fps`, `threshold`, `active_pct_threshold`, `downscale_width`.
- Use `threshold` for pixel-level sensitivity and `active_pct_threshold` for event-level sensitivity.
- Runs on CPU.

</details>

---

<a id="video-image-quality"></a>
### video.image_quality

**Purpose:** Summarize visual quality problems such as blur, dark frames,
overexposure, low contrast, and possible obstruction.

**Technique:** Sample frames and compute Laplacian sharpness, grayscale
luminance, luminance standard deviation, exposure tiers, and low-variance
obstruction checks.

**Granularity:** Sampled frame.

**Output:** Per-frame quality samples; blur/dark/overexposed/obstruction events;
source-video quality overlay; timeline JSON; feature CSV; HTML report; config
YAML; log text.

<details><summary>CLI</summary>

```bash
av-toolbox video image-quality input.mp4 \
  --output outputs/video_quality \
  --sample-fps 5 \
  --max-seconds 10
```

Use this before model-backed analysis when you want to know whether the source is
too blurry, too dark, overexposed, or blocked for reliable downstream inference.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.image_quality",
    input_path="input.mp4",
    output_dir="outputs/video_quality",
    sample_fps=5,
    max_seconds=10,
)
print(result.timeline_json)
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[video]`.
- Useful knobs: `blur_threshold`, `dark_threshold`, `super_dark_threshold`, `overexposed_threshold`, `obstruction_threshold`.
- Use Python or the local UI for thresholds not exposed by the CLI.
- Runs on CPU.

</details>

---

<a id="video-optical-flow"></a>
### video.optical_flow

**Purpose:** Measure dense visual motion and produce a motion-mask overlay.

**Technique:** Compute Farneback dense optical flow between sampled frames, turn
flow vectors into magnitude maps, and threshold active pixels/events.

**Granularity:** Sampled frame pair.

**Output:** Flow magnitude samples, pixel-motion events, overlay MP4, timeline
JSON, feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox video optical-flow input.mp4 \
  --output outputs/video_optical_flow \
  --sample-fps 5 \
  --max-seconds 10
```

Use this when direction-independent motion intensity matters more than object
identity.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.optical_flow",
    input_path="input.mp4",
    output_dir="outputs/video_optical_flow",
    sample_fps=5,
    max_seconds=10,
)
print(result.csv_path)
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[video]`.
- Useful knobs: `active_threshold_px`, `event_threshold_px`, `active_pct_threshold`, `downscale_width`, `overlay_alpha`.
- Raise thresholds to ignore tiny camera noise; lower them to catch subtle movement.
- Runs on CPU.

</details>

---

<a id="video-foreground-motion"></a>
### video.foreground_motion

**Purpose:** Measure motion primarily inside foreground regions instead of the
whole frame.

**Technique:** Compute dense optical flow and optionally restrict measurement to
YOLO foreground masks with `mask_mode='yolo'` or `mask_mode='yolo_seg'`.

**Granularity:** Sampled frame pair, optionally foreground-masked.

**Output:** Foreground motion samples, event regions, overlay MP4, timeline JSON,
feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox video foreground-motion input.mp4 \
  --output outputs/video_foreground_motion \
  --sample-fps 5 \
  --max-seconds 10 \
  --mask-mode none
```

For YOLO-assisted foreground masks:

```bash
av-toolbox video foreground-motion input.mp4 \
  --output outputs/video_foreground_motion_yolo \
  --max-seconds 10 \
  --mask-mode yolo_seg \
  --model-name yolov8n-seg.pt \
  --confidence 0.25
```

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.foreground_motion",
    input_path="input.mp4",
    output_dir="outputs/video_foreground_motion",
    mask_mode="yolo_seg",
    max_seconds=10,
)
print(result.metadata["summary"])
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[video]`; YOLO mask modes also need `.[vision-models]`.
- Useful knobs: `mask_mode`, `model_name`, `confidence`, `active_threshold_px`, `event_threshold_px`.
- Start with `mask_mode='none'`; switch to YOLO when camera/background motion creates too many false positives.
- GPU is helpful for YOLO mask modes when torchvision CUDA NMS works; otherwise `--device auto` falls back to CPU.

</details>

---

<a id="video-camera-shake"></a>
### video.camera_shake

**Purpose:** Detect jittery camera shake rather than ordinary scene motion.

**Technique:** Track sparse optical-flow feature points, estimate frame-to-frame
translation, detrend rolling movement, and flag windows whose residual jitter
exceeds `threshold`.

**Granularity:** Rolling sampled-frame window.

**Output:** Shake scores, shake events, source-video shake overlay, timeline
JSON, feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox video camera-shake input.mp4 \
  --output outputs/video_camera_shake \
  --sample-fps 25 \
  --max-seconds 10 \
  --threshold 0.5
```

Use this for handheld footage, vibration, and mount-stability checks.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.camera_shake",
    input_path="input.mp4",
    output_dir="outputs/video_camera_shake",
    sample_fps=25,
    max_seconds=10,
)
print(result.overlay_path)
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[video]`.
- Useful knobs: `sample_fps`, `window`, `threshold`, `min_features`, `max_features`, `redetect_interval`.
- Increase `threshold` if normal pans are being flagged; decrease it for sensitive vibration checks.
- Runs on CPU.

</details>

---

<a id="video-cut-detection"></a>
### video.cut_detection

**Purpose:** Find shot boundaries and scene segments.

**Technique:** Use TransNetV2 by default, with explicit PySceneDetect or
lightweight fallback backends when selected.

**Granularity:** Frame-level cut score and scene segment.

**Output:** Cut events, scene segments, timeline overlay, timeline JSON, feature
CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox video cut-detection input.mp4 \
  --output outputs/video_cut_detection \
  --max-seconds 30 \
  --backend transnetv2 \
  --threshold 0.5
```

Use this for editing boundaries, scene indexing, and splitting longer videos
into meaningful chunks.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.cut_detection",
    input_path="input.mp4",
    output_dir="outputs/video_cut_detection",
    backend="transnetv2",
    max_seconds=30,
)
print(result.metadata["summary"])
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[cut-detection]` for TransNetV2/PySceneDetect support.
- Useful knobs: `backend`, `threshold`, `weights_path`, `min_distance_frames`, `scenedetect_threshold`.
- GPU can help model-backed TransNetV2 runs.

</details>

---

<a id="video-object-detection"></a>
### video.object_detection

**Purpose:** Detect objects and draw bounding boxes on sampled frames.

**Technique:** Run a YOLO object detector on sampled frames and collect class
labels, confidences, and bounding boxes.

**Granularity:** Sampled frame.

**Output:** Object events/samples, bounding-box overlay MP4, timeline JSON,
feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox video object-detection input.mp4 \
  --output outputs/video_object_detection \
  --sample-fps 2 \
  --max-seconds 10 \
  --model-name yolov8n.pt \
  --confidence 0.25
```

Use this when you care which objects appear, not just that motion occurred.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.object_detection",
    input_path="input.mp4",
    output_dir="outputs/video_object_detection",
    model_name="yolov8n.pt",
    confidence=0.25,
    max_seconds=10,
)
print(result.timeline_json)
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[vision-models]`.
- Useful knobs: `sample_fps`, `model_name`, `confidence`, `image_size`.
- GPU is recommended for larger YOLO models or longer clips when torchvision CUDA NMS works; otherwise `--device auto` falls back to CPU.

</details>

---

<a id="video-segmentation"></a>
### video.segmentation

**Purpose:** Segment object instances with masks, boxes, classes, and
confidences.

**Technique:** Run a YOLO segmentation model on sampled frames and render masks
back over the source video.

**Granularity:** Sampled frame.

**Output:** Instance-mask samples, mask-area summaries, segmentation overlay MP4,
timeline JSON, feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox video segmentation input.mp4 \
  --output outputs/video_segmentation \
  --sample-fps 2 \
  --max-seconds 10 \
  --model-name yolov8n-seg.pt \
  --confidence 0.25
```

Use this when object shape/region matters more than a bounding box.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.segmentation",
    input_path="input.mp4",
    output_dir="outputs/video_segmentation",
    model_name="yolov8n-seg.pt",
    max_seconds=10,
)
print(result.overlay_path)
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[vision-models]`.
- Useful knobs: `sample_fps`, `model_name`, `confidence`, `image_size`.
- GPU is recommended for practical throughput when torchvision CUDA NMS works; otherwise `--device auto` falls back to CPU.

</details>

---

<a id="video-pose"></a>
### video.pose

**Purpose:** Detect human pose landmarks in sampled frames.

**Technique:** Run a YOLOv8-pose model and write landmark events plus an
overlay with skeleton/landmark visualization.

**Granularity:** Sampled frame.

**Output:** Pose landmark events, source-video pose overlay, timeline JSON,
feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox video pose input.mp4 \
  --output outputs/video_pose \
  --sample-fps 10 \
  --max-seconds 10
```

Use this for human-body layout, presence, and rough movement verification.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.pose",
    input_path="input.mp4",
    output_dir="outputs/video_pose",
    sample_fps=10,
    max_seconds=10,
)
print(result.csv_path)
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[vision-models]`.
- Useful knobs: `model_name`, `confidence`, `keypoint_threshold`, `sample_fps`.
- GPU is recommended for practical throughput when torchvision CUDA NMS works; otherwise `--device auto` falls back to CPU.

</details>

---

<a id="video-shot-type"></a>
### video.shot_type

**Purpose:** Label sampled frames by cinematographic shot type.

**Technique:** Run a Transformers image classifier on sampled frames and keep the
top-k label probabilities.

**Granularity:** Sampled frame.

**Output:** Shot-type label samples, top-k probabilities, source-video label
overlay, timeline JSON, feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox video shot-type input.mp4 \
  --output outputs/video_shot_type \
  --sample-fps 1 \
  --max-seconds 10 \
  --top-k 3
```

Use this for coarse visual language like close-up, medium shot, wide shot, and
other model-provided labels.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.shot_type",
    input_path="input.mp4",
    output_dir="outputs/video_shot_type",
    top_k=3,
    max_seconds=10,
)
print(result.metadata["summary"])
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[vision-models]`.
- Useful knobs: `model_name`, `sample_fps`, `downscale_width`, `top_k`, `offline`.
- GPU is recommended for larger Transformers models.

</details>

---

<a id="video-action-recognition"></a>
### video.action_recognition

**Purpose:** Classify short video windows by action label.

**Technique:** Sample temporal windows and run a SlowFast/PyTorchVideo action
recognition model, then render top labels back over the source video.

**Granularity:** Sliding video window.

**Output:** Window-level action labels, confidences, source-video action overlay,
timeline JSON, feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox video action-recognition input.mp4 \
  --output outputs/video_action_recognition \
  --max-seconds 10 \
  --window-seconds 2 \
  --step-seconds 1 \
  --top-k 3 \
  --confidence 0.3
```

Use this when object labels are not enough and you need activity labels over
time.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "video.action_recognition",
    input_path="input.mp4",
    output_dir="outputs/video_action_recognition",
    window_seconds=2,
    step_seconds=1,
    top_k=3,
    max_seconds=10,
)
print(result.timeline_json)
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[action]`.
- Useful knobs: `model_name`, `window_seconds`, `step_seconds`, `top_k`, `confidence`, `labels_path`.
- GPU is recommended.

</details>

---

---

<a id="audio-beat-detection"></a>
### audio.beat_detection

**Purpose:** Detect beats, heuristic downbeats, onsets, and tempo.

**Technique:** Load/resample audio, compute an onset envelope, run librosa beat
tracking with onset fallback, and mark every fourth beat as a heuristic downbeat.

**Granularity:** Audio event timeline.

**Output:** Beat/downbeat/onset events, tempo summary, waveform/timeline overlay,
timeline JSON, feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox audio beat-detection input.wav \
  --output outputs/audio_beat_detection \
  --sample-rate 22050 \
  --hop-length 512 \
  --max-seconds 30
```

This also accepts video files with audio tracks.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "audio.beat_detection",
    input_path="input.wav",
    output_dir="outputs/audio_beat_detection",
    sample_rate=22050,
    max_seconds=30,
)
print(result.metadata["summary"])
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[audio]`.
- Useful knobs: `sample_rate`, `hop_length`, `window_sec`, `overlay_fps`.
- Runs on CPU.

</details>

---

<a id="audio-energy"></a>
### audio.energy

**Purpose:** Measure loudness/energy and silence over time.

**Technique:** Load audio and compute frame-level RMS, dB energy, spectral
centroid, zero-crossing features, and silence labels.

**Granularity:** Audio frame/window.

**Output:** Energy samples, silence/high-energy regions, waveform/energy overlay,
timeline JSON, feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox audio energy input.wav \
  --output outputs/audio_energy \
  --sample-rate 22050 \
  --hop-length 512 \
  --max-seconds 30
```

Use this as a fast first pass for music/audio intensity and silence detection.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "audio.energy",
    input_path="input.wav",
    output_dir="outputs/audio_energy",
    sample_rate=22050,
    max_seconds=30,
)
print(result.csv_path)
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[audio]`.
- Useful knobs: `sample_rate`, `hop_length`, `frame_length`, `rms_floor_db`, `silence_threshold_db`, `window_sec`.
- Use Python or the local UI for thresholds not exposed by the CLI.
- Runs on CPU.

</details>

---

<a id="audio-event-detection"></a>
### audio.event_detection

**Purpose:** Detect audio impacts, high/low-energy regions, spectral changes,
and tonal shifts.

**Technique:** Compute audio features over windows, compare deltas and quantiles,
and group sustained regions with minimum duration rules.

**Granularity:** Audio event/region.

**Output:** Impact/energy/spectral/tonal events, overlay MP4, timeline JSON,
feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox audio event-detection input.wav \
  --output outputs/audio_event_detection \
  --sample-rate 22050 \
  --hop-length 512 \
  --max-seconds 30
```

Use this when you need timestamps for meaningful audio changes rather than a
continuous energy trace.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "audio.event_detection",
    input_path="input.wav",
    output_dir="outputs/audio_event_detection",
    impact_delta=0.28,
    spectral_delta=0.34,
    max_seconds=30,
)
print(result.timeline_json)
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[audio]`.
- Useful knobs: `impact_delta`, `spectral_delta`, `tonal_delta`, `silence_threshold`, `high_energy_quantile`, `min_region_seconds`.
- Runs on CPU.

</details>

---

<a id="audio-music-phase"></a>
### audio.music_phase

**Purpose:** Estimate coarse music sections or phase changes.

**Technique:** Analyze audio feature changes over musical phrases and group them
into coarse phase segments.

**Granularity:** Music segment/phase.

**Output:** Phase events, waveform/phase overlay, timeline JSON, feature CSV,
HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox audio music-phase input.wav \
  --output outputs/audio_music_phase \
  --sample-rate 22050 \
  --hop-length 512 \
  --max-seconds 60
```

Use this for rough verse/chorus/bridge-like structure cues, not precise music
source separation.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "audio.music_phase",
    input_path="input.wav",
    output_dir="outputs/audio_music_phase",
    min_phase_seconds=6,
    phrase_bars=4,
    max_seconds=60,
)
print(result.metadata["summary"])
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[audio]`.
- Useful knobs: `sample_rate`, `hop_length`, `min_phase_seconds`, `phrase_bars`, `window_sec`.
- Runs on CPU.

</details>

---

<a id="audio-transcription"></a>
### audio.transcription

**Purpose:** Transcribe speech from audio or video files.

**Technique:** Load the audio track and run faster-whisper, optionally with VAD
filtering and a chosen compute type.

**Granularity:** Speech segment.

**Output:** Transcript segments, transcript text in report/artifacts, timeline
JSON, feature CSV, HTML report, config YAML, log text. This tool does not
usually produce an overlay MP4.

<details><summary>CLI</summary>

```bash
av-toolbox audio transcription input.mp4 \
  --output outputs/audio_transcription \
  --model-name base \
  --language en \
  --max-seconds 60
```

Use this for speech-heavy clips where text timing is more important than visual
overlays.

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "audio.transcription",
    input_path="input.mp4",
    output_dir="outputs/audio_transcription",
    model_name="base",
    language="en",
    max_seconds=60,
)
print(result.report_html)
```

</details>

<details><summary>Tuning</summary>

- Install extra: `.[transcription]`.
- Useful knobs: `model_name`, `language`, `beam_size`, `vad_filter`, `compute_type`, `max_seconds`.
- GPU is recommended for larger Whisper models; CPU works for smaller models.

</details>

---

---

<a id="av-denseav"></a>
### av.denseav

**Purpose:** Produce DenseAV audio-visual correspondence and attention overlays.

**Technique:** Load a DenseAV checkpoint, sample video/audio, compute similarity
or attention outputs, and render model attention summaries.

**Granularity:** Sampled audio-visual segment.

**Output:** DenseAV similarity summaries, per-head statistics, attention videos,
optional similarity matrix, overlay MP4 when enough frames are sampled, timeline
JSON, feature CSV, HTML report, config YAML, log text.

<details><summary>CLI</summary>

```bash
av-toolbox av denseav input.mp4 \
  --output outputs/av_denseav \
  --model-name sound_and_language \
  --max-seconds 5 \
  --load-size 224 \
  --plot-size 720
```

Use this only after the DenseAV dependency and checkpoint files are installed.
For setup details, see [denseav.md](denseav.md).

</details>

<details><summary>Use as a Library</summary>

```python
from av_toolbox import run_tool

result = run_tool(
    "av.denseav",
    input_path="input.mp4",
    output_dir="outputs/av_denseav",
    model_name="sound_and_language",
    max_seconds=5,
)
print(result.overlay_path)
```

</details>

<details><summary>Tuning</summary>

- Install runtime extra: `.[denseav]`, then `python -m pip install "git+https://github.com/mhamilton723/DenseAV.git"`.
- Useful knobs: `model_name`, `checkpoint`, `sample_fps`, `load_size`, `plot_size`, `max_seconds`, `include_sim_matrix`.
- GPU is strongly recommended.
- Use `offline=True` and explicit checkpoints for reproducible no-download runs.

</details>

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
| `video.pose` | `av-toolbox video pose` | Video | Multi-person YOLOv8-pose keypoints and a per-frame synced multi-person skeleton overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.shot_type` | `av-toolbox video shot-type` | Video | Transformers/BEiT-style per-frame shot-type labels, top-k probabilities, and source-video label overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `video.action_recognition` | `av-toolbox video action-recognition` | Video | SlowFast/PyTorchVideo window-level action labels and source-video action overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.beat_detection` | `av-toolbox audio beat-detection` | Audio or video with audio | Beats, heuristic downbeats, onsets, tempo, and waveform/timeline overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.energy` | `av-toolbox audio energy` | Audio or video with audio | RMS, dB energy, spectral centroid, zero-crossing, silence samples, and waveform/energy overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.event_detection` | `av-toolbox audio event-detection` | Audio or video with audio | Impacts, low/high-energy regions, spectral changes, tonal shifts, and overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.music_phase` | `av-toolbox audio music-phase` | Audio or video with audio | Coarse music phase segments and overlay. | Overlay MP4, timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `audio.transcription` | `av-toolbox audio transcription` | Audio or video with audio | faster-whisper speech segments and transcript text. | Timeline JSON, feature CSV, HTML report, config YAML, log text. |
| `av.denseav` | `av-toolbox av denseav` | Video with audio | DenseAV similarity summaries, per-head statistics, attention videos, and optional similarity matrix. | Overlay MP4 when enough frames are sampled, timeline JSON, feature CSV, HTML report, config YAML, log text. DenseAV may also write an additional `_2head_attention.mp4` path listed inside the timeline JSON. |

Model-backed tools keep imports lazy. Install optional extras as needed: `.[transcription]` for Whisper, `.[vision-models]` for YOLO pose/object/segmentation and shot-type, `.[action]` for PyTorchVideo action recognition, `.[cut-detection]` for TransNetV2 and PySceneDetect cut detection support, and `.[denseav]` for DenseAV runtime prerequisites. DenseAV itself is installed from `git+https://github.com/mhamilton723/DenseAV.git`.

## Requirements At A Glance

| Tool | Input | Main output | Optional dependency | GPU/model requirement |
| --- | --- | --- | --- | --- |
| `video.motion` | Video | Motion intensity overlay, timeline, CSV, report | `.[video]` | CPU |
| `video.image_quality` | Video | Sharpness/luma/contrast quality overlay | `.[video]` | CPU |
| `video.optical_flow` | Video | Dense optical-flow mask overlay | `.[video]` | CPU |
| `video.foreground_motion` | Video | Foreground-biased flow overlay | `.[video]`; YOLO modes use `.[vision-models]` | CPU by default; YOLO can use GPU |
| `video.camera_shake` | Video | Camera shake score overlay | `.[video]` | CPU |
| `video.cut_detection` | Video | TransNetV2 cut timeline overlay | `.[cut-detection]` | TransNetV2 model; GPU optional |
| `video.object_detection` | Video | YOLO bounding-box overlay | `.[vision-models]` | YOLO model; GPU recommended |
| `video.segmentation` | Video | YOLO instance-mask overlay | `.[vision-models]` | YOLO model; GPU recommended |
| `video.pose` | Video with people | Multi-person YOLOv8-pose overlay | `.[vision-models]` | Downloads yolov8n-pose.pt on first run |
| `video.shot_type` | Video | Shot-type label/confidence overlay | `.[vision-models]` | Transformers image model; GPU recommended |
| `video.action_recognition` | Video | SlowFast action-label overlay | `.[action]` | PyTorchVideo model; GPU recommended |
| `audio.beat_detection` | Audio or video with audio | Beat/downbeat/onset waveform overlay | `.[audio]` | CPU |
| `audio.energy` | Audio or video with audio | RMS/dB/silence waveform overlay | `.[audio]` | CPU |
| `audio.event_detection` | Audio or video with audio | Impact/energy/spectral event overlay | `.[audio]` | CPU |
| `audio.music_phase` | Music audio/video | Coarse music phase overlay | `.[audio]` | CPU |
| `audio.transcription` | Speech audio/video | Whisper transcript timeline/artifacts | `.[transcription]` | faster-whisper model; GPU recommended |
| `av.denseav` | Video with audio | DenseAV attention/similarity overlays | `.[denseav]` plus DenseAV Git install | DenseAV checkpoint; GPU recommended |
