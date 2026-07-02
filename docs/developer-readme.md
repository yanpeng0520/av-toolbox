# av-toolbox Developer README

This is the developer README for local installs, CLI usage, tests, Docker,
and the web UI. The public GitHub landing page lives in [../README.md](../README.md).


`av-toolbox` is the project and CLI name. The PyPI distribution is
`av-analysis-toolbox`; the import package remains `av_toolbox`. The package
uses `src/av_toolbox` as its only importable package root and exposes one console
entry point:

```bash
av-toolbox
```

Existing folders such as `omni-video-pipeline/`, `testing/`, `data_segments/`,
`data_samples/`, `multicam_samples/`, and `models/` are project assets or
migration sources. They are not included as Python package contents.

## Quick Start

```bash
git clone https://github.com/yanpeng0520/av-toolbox.git
cd av-toolbox

conda create -n av-toolbox python=3.12 -y
conda activate av-toolbox

python -m pip install --upgrade pip
python -m pip install -e ".[audio,video,av,dev]"
git lfs install
git lfs pull --include="data_segments/*.mp4"
av-toolbox list-tools
av-toolbox generate-demo-media --output-dir data_segments --duration 60
```

Optional migrated model tools use additional extras, for example `.[transcription]` for Whisper, `.[vision-models]` for YOLO pose/object/segmentation and shot-type classification, `.[action]` for PyTorchVideo action recognition, and `.[cut-detection]` for TransNetV2 and PySceneDetect support.

Requires `ffmpeg` on PATH for video/audio media operations.

The generated MP4 preview is muxed from the WAV source with AAC audio for broad player compatibility.

Run the packaged video tools on the default demo clip:

```bash
av-toolbox video image-quality \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_image_quality \
  --sample-fps 5

av-toolbox video motion \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_motion \
  --sample-fps 5

av-toolbox video cut-detection \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_cut_detection \
  --max-seconds 8
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
av-toolbox av denseav \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_denseav_clever_cat \
  --max-seconds 5 \
  --sample-fps 5 \
  --device auto
```

DenseAV is an optional heavy dependency. Attention MP4s default to a 720px render size while keeping model inference at 224px; use `--plot-size` or `--load-size` to override those separately. Install runtime prerequisites with `pip install -e ".[denseav]"`, then install DenseAV with `pip install "git+https://github.com/mhamilton723/DenseAV.git"`
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
It opens with the generated demo clip, writes to `outputs/web_runs/latest`, and
keeps tool parameters at their built-in defaults unless Advanced overrides are
enabled. The Advanced parameter area adapts to the selected tool and starts from
that tool's current defaults. It can upload or select media, preview MP4/audio
inputs and outputs, and download declared artifacts.

## Public Demo Mode

Public demo mode starts a bounded Streamlit origin. Keep the listener local
when placing it behind your own tunnel, proxy, or access layer:

```bash
av-toolbox serve \
  --host 127.0.0.1 \
  --port 8501 \
  --output-root /srv/av-toolbox-demo/runs \
  --page-title "AV Toolbox Demo" \
  --public-demo \
  --public-max-seconds 20 \
  --public-max-upload-mb 10
```

Use `--public-enable-denseav` only after DenseAV dependencies and weights are
installed. Public mode starts with a generated sample clip, also
accepts uploads, exposes only bounded demo settings, hides local filesystem and
runtime controls, caps uploads and analyzed duration, creates server-side output
directories, and serializes public Streamlit runs.

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

See [quickstart.md](quickstart.md) for the full install, CLI, DenseAV, web UI, and Docker
quickstart.

## Docs

- [tool-catalog.md](tool-catalog.md): every registered tool name, CLI wrapper, input type,
  artifact outputs, and runtime controls.
- [denseav.md](denseav.md): DenseAV install, exact checkpoint filenames, cache
  locations, GPU flags, and troubleshooting.
- [examples.md](examples.md): Python API examples and CLI batch examples.

## Python API

```python
import av_toolbox

tools = av_toolbox.list_tools()
result = av_toolbox.run_tool(
    "video.motion",
    input_path="data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4",
    output_dir="outputs/python_motion",
    sample_fps=5,
    max_seconds=5,
    device="cpu",
)
print(result.timeline_json)
```

The package now includes the registry/runtime foundation plus classical video
quality and motion tools, audio feature/event/phase tools, audio-visual sync and
DenseAV tools, and native wrappers for migrated Whisper, YOLO pose/object/segmentation,
shot-type, cut-detection, and action-recognition capabilities. Model-backed
tools are registered by default but import their optional dependencies only when
run. Use `.[transcription]`, `.[vision-models]`, `.[action]`,
`.[cut-detection]`, or `.[denseav]` for heavier runtime prerequisites. DenseAV itself is installed from `git+https://github.com/mhamilton723/DenseAV.git`.

