# UI Tool Test Plan

Use this checklist when testing every tool from the local Streamlit UI.

Start the full local UI, not public demo mode:

```bash
cd /path/to/av-toolbox
source .venv/bin/activate
av-toolbox serve \
  --host 127.0.0.1 \
  --port 8501 \
  --output-root outputs/web_runs \
  --page-title "AV Toolbox Demo"
```

Open:

```text
http://127.0.0.1:8501
```

## Test Loop

For each tool:

1. Select the tool in the UI.
2. Use the default sample video, upload a matching sample, or set Media Path.
3. Keep defaults first; only use Advanced when a tool requires model/runtime setup.
4. Click Run.
5. Confirm there is no red Streamlit error.
6. Confirm Input stays on the left, Output on the right, and Results underneath.
7. Confirm expected artifacts appear and downloads work. Some tools show an overlay MP4; tools without an overlay should still show a source preview or a clear output message.
8. Tick the tool row below and write notes.

## Sample Inputs

- Video or AV tools: `data_segments/CatFu.mp4`
- Generated video with audio: `data_segments/synthetic_hiphop_60s.mp4`
- Generated audio: `data_segments/synthetic_hiphop_60s.wav`

If the LFS video is missing:

```bash
git lfs pull --include="data_segments/*.mp4"
```

If generated media is missing:

```bash
av-toolbox generate-demo-media --output-dir data_segments --duration 60
```

## Checklist

| Done | Tool | Input | Expected UI result | Notes |
| --- | --- | --- | --- | --- |
| [ ] | `video.motion` | Video | Motion overlay MP4 on Output; timeline, CSV/report/config/log underneath. | |
| [ ] | `video.image_quality` | Video | Sharp/luma/contrast + obstruction overlay MP4 on Output with 3-metric card and colored line timeline; timeline, CSV/report/config/log underneath. | |
| [ ] | `video.cut_detection` | Video | TransNetV2 cut overlay MP4 on Output with cut timeline; timeline, CSV/report/config/log underneath. | |
| [ ] | `video.optical_flow` | Video | Optical-flow masked-map overlay MP4 on Output; timeline, CSV/report/config/log underneath. | |
| [ ] | `video.foreground_motion` | Video | Foreground flow-mask overlay MP4 on Output; timeline, CSV/report/config/log underneath. | |
| [ ] | `video.camera_shake` | Video | Source-video shake overlay MP4 on Output; timeline, CSV/report/config/log underneath. | |
| [ ] | `video.object_detection` | Video | Source-video box overlay MP4 on Output; timeline, CSV/report/config/log underneath. Requires `vision-models`. | |
| [ ] | `video.segmentation` | Video | Source-video segment overlay MP4 on Output; timeline, CSV/report/config/log underneath. Requires `vision-models`. | |
| [ ] | `video.pose` | Video with people | Multi-person YOLOv8-pose skeleton overlay MP4 on Output (per-frame synced, one color per person, people-count timeline); timeline, CSV/report/config/log underneath. Requires `vision-models`. | |
| [ ] | `video.shot_type` | Video | Source-video shot-label overlay MP4 on Output; timeline, CSV/report/config/log underneath. Requires `vision-models`. | |
| [ ] | `video.action_recognition` | Video | Source-video action-label overlay MP4 on Output; timeline, CSV/report/config/log underneath. Requires `action`. | |
| [ ] | `audio.beat_detection` | Audio or video with audio | Beat/onset summary, overlay, and artifacts. | |
| [ ] | `audio.energy` | Audio or video with audio | Energy/silence waveform overlay MP4 on Output; timeline, CSV/report/config/log underneath. | |
| [ ] | `audio.event_detection` | Audio or video with audio | Event/region summary, overlay, and artifacts. | |
| [ ] | `audio.music_phase` | Audio or video with audio | Phase segments, overlay, and artifacts. | |
| [ ] | `audio.transcription` | Speech audio/video | Transcript segments and artifacts. Requires `transcription`. | |
| [ ] | `av.denseav` | Video with audio | DenseAV summary and attention overlay. Requires `denseav` and checkpoint. | |

## After Each Tool

Tick the `Done` box only after:

- [ ] Run completed without a UI exception.
- [ ] Output panel rendered correctly.
- [ ] Results panel rendered underneath.
- [ ] At least one expected artifact was produced.
- [ ] Any failure or missing dependency is written in Notes.

When you finish a batch, send me the checked rows and notes; I can help decide whether failures are real bugs, missing dependencies, or sample-media issues.
