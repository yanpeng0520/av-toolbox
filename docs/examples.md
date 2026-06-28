# API And CLI Examples

These examples use the packaged registry names and the default demo media:

```text
data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4
```

Generate the synthetic hip-hop WAV/MP4 demo when you want a longer music input:

```bash
av-toolbox generate-demo-media --output-dir data_segments --duration 60
```

## Python: Single Tool

```python
from pathlib import Path
import av_toolbox

media = Path("data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4")

result = av_toolbox.run_tool(
    "video.motion",
    input_path=media,
    output_dir="outputs/python_motion",
    sample_fps=5,
    max_seconds=5,
    device="cpu",
)

print(result.timeline_json)
print(result.to_dict())
```

## Python: Batch Several Tools

```python
from pathlib import Path
import av_toolbox

media = Path("data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4")
tools = [
    "video.blur_exposure",
    "video.motion",
    "video.shot_boundary",
    "audio.beat_detection",
    "audio.event_detection",
    "audio.music_phase",
    "av.sync_correspondence",
]

for tool_name in tools:
    slug = tool_name.replace(".", "_")
    result = av_toolbox.run_tool(
        tool_name,
        input_path=media,
        output_dir=Path("outputs/python_batch") / slug,
        sample_fps=5,
        max_seconds=8,
        overlay_fps=5,
        device="cpu",
        export_json=True,
        export_csv=True,
        export_report=True,
        export_overlay=True,
    )
    print(tool_name, result.to_dict())
```

## Python: Hardware, Cache, And Workspace

```python
import av_toolbox

result = av_toolbox.run_tool(
    "av.denseav",
    input_path="data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4",
    output_dir="outputs/python_denseav",
    model_name="sound_and_language",
    cache_dir="~/.cache/av_toolbox",
    workspace_dir="/tmp/av_toolbox/denseav_debug",
    keep_workspace=True,
    device="cuda:0",
    batch_size=4,
    fp16=True,
    max_seconds=5,
    sample_fps=5,
)
```

## CLI: Generic Registry Batch

```bash
MEDIA=data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4

for TOOL in \
  video.blur_exposure \
  video.motion \
  video.shot_boundary \
  audio.beat_detection \
  audio.event_detection \
  audio.music_phase \
  av.sync_correspondence
do
  SLUG=${TOOL//./_}
  av-toolbox run "$TOOL" "$MEDIA" \
    --output "outputs/cli_batch/$SLUG" \
    --sample-fps 5 \
    --max-seconds 8 \
    --overlay-fps 5 \
    --device cpu
done
```

## CLI: Category Commands

```bash
av-toolbox video blur-exposure \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/video_blur \
  --sample-fps 5 \
  --max-seconds 10

av-toolbox video motion \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/video_motion \
  --sample-fps 5 \
  --max-seconds 10

av-toolbox video shot-boundary \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/video_shots \
  --sample-fps 5 \
  --max-seconds 10
```

```bash
av-toolbox audio beat-detection \
  data_segments/synthetic_hiphop_60s.wav \
  --output outputs/audio_beats \
  --window-sec 4 \
  --overlay-fps 5

av-toolbox audio event-detection \
  data_segments/synthetic_hiphop_60s.wav \
  --output outputs/audio_events \
  --window-sec 4 \
  --overlay-fps 5

av-toolbox audio music-phase \
  data_segments/synthetic_hiphop_60s.wav \
  --output outputs/audio_phase \
  --window-sec 4 \
  --overlay-fps 5
```

```bash
av-toolbox av sync-correspondence \
  data_segments/synthetic_hiphop_60s.mp4 \
  --output outputs/av_sync \
  --sample-fps 10 \
  --max-seconds 20 \
  --overlay-fps 5
```

## CLI: DenseAV

```bash
av-toolbox av denseav \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/denseav \
  --model-name sound_and_language \
  --max-seconds 5 \
  --sample-fps 5 \
  --load-size 224 \
  --plot-size 720 \
  --device cuda:0 \
  --batch-size 4 \
  --fp16
```

See `docs/denseav.md` for checkpoint setup before running DenseAV.

## CLI: Output Selection

```bash
av-toolbox run video.motion \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/motion_json_only \
  --max-seconds 5 \
  --no-csv \
  --no-report \
  --no-overlay
```

The command prints the `AVResult` JSON, including paths for the artifacts that
were actually produced.
