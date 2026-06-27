# av-toolbox

`av-toolbox` is the installable Python package for this project. The package
uses `src/av_toolbox` as its only importable package root and exposes one console
entry point:

```bash
av-toolbox
```

Existing folders such as `omni-video-pipeline/`, `testing/`, `data_segments/`,
`data_samples/`, `multicam_samples/`, and `models/` are project assets or
migration sources. They are not included as Python package contents.

## Development Install

```bash
pip install -e ".[audio,video,av,dev]"
av-toolbox list-tools
av-toolbox generate-demo-media --output-dir data_segments --duration 60
```

The generated MP4 preview is muxed from the WAV source with AAC audio for broad player compatibility.

Run the packaged video tools on the default demo clip:

```bash
av-toolbox video blur-exposure \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_blur_exposure \
  --sample-fps 5

av-toolbox video motion \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_motion \
  --sample-fps 5

av-toolbox video shot-boundary \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_shot_boundary \
  --sample-fps 5
```

Run the packaged audio tools on the generated hip-hop demo:

```bash
av-toolbox audio beat-detection \
  data_segments/synthetic_hiphop_60s.wav \
  --output outputs/demo_beats

av-toolbox audio event-detection \
  data_segments/synthetic_hiphop_60s.wav \
  --output outputs/demo_events

av-toolbox audio music-phase \
  data_segments/synthetic_hiphop_60s.wav \
  --output outputs/demo_music_phase
```

Run the packaged audio-visual tools:

```bash
av-toolbox av sync-correspondence \
  data_segments/synthetic_hiphop_60s.mp4 \
  --output outputs/demo_sync

av-toolbox av denseav \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_denseav_clever_cat \
  --max-seconds 5 \
  --sample-fps 5 \
  --device auto
```

DenseAV is an optional heavy dependency. Attention MP4s default to a 720px render size while keeping model inference at 224px; use `--plot-size` or `--load-size` to override those separately. Install it with `pip install -e ".[denseav]"`
and place model weights in the unified cache before running inference:

```bash
mkdir -p ~/.cache/av_toolbox/weights
# Expected default names:
# ~/.cache/av_toolbox/weights/denseav_2head.ckpt
# ~/.cache/av_toolbox/weights/denseav_sound.ckpt
```

You can also point at an existing local checkpoint without copying it:

```bash
av-toolbox av denseav \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_denseav_clever_cat \
  --checkpoint /path/to/denseav_2head.ckpt \
  --max-seconds 5 \
  --device cuda:0 \
  --batch-size 2 \
  --fp16
```

## Local Web UI

```bash
av-toolbox serve --host 127.0.0.1 --port 8501
```

The local UI uses the same registry and runtime contract as the Python API and CLI.
It can upload or select media, run any registered tool, preview MP4/audio outputs,
and download declared artifacts.

## Tests, CI, And Docker

```bash
python -m pytest
```

CI is defined in `.github/workflows/ci.yml` and runs the packaged `tests/` suite
against Python 3.10, 3.11, and 3.12, plus a Docker build/CLI smoke job. The
default CI dependency set avoids installing heavyweight DenseAV dependencies or
downloading model weights.

Optional DenseAV GPU validation lives in `.github/workflows/denseav-gpu-smoke.yml`.
Run it manually on a self-hosted GPU runner with `~/.cache/av_toolbox/weights/denseav_2head.ckpt`
or pass the cache path through the workflow dispatch input.

Build and run the local Docker image:

```bash
docker build -t av-toolbox .
docker run --rm -p 8501:8501 \
  -v "$PWD/data_segments:/app/data_segments" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$HOME/.cache/av_toolbox:/cache/av_toolbox" \
  av-toolbox
```

See `docs/quickstart.md` for the full install, CLI, DenseAV, web UI, and Docker
quickstart.

## Python API

```python
import av_toolbox

tools = av_toolbox.list_tools()
```

The package currently includes the core registry/runtime foundation plus the
first packaged video tools, `video.blur_exposure`, `video.motion`, and
`video.shot_boundary`, the first audio tools, `audio.beat_detection`,
`audio.event_detection`, and `audio.music_phase`, and audio-visual tools
`av.sync_correspondence` and `av.denseav`. Additional production `av_tools` and
`omni-video-pipeline` capabilities will be wrapped into this package in later
milestones.

