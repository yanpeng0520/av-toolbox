# AV Toolbox Demo

[![PyPI version](https://badge.fury.io/py/av-analysis-toolbox.svg)](https://pypi.org/project/av-analysis-toolbox/)
[![Python Versions](https://img.shields.io/pypi/pyversions/av-analysis-toolbox.svg)](https://pypi.org/project/av-analysis-toolbox/)
[![Build Status](https://github.com/yanpeng0520/av-toolbox/actions/workflows/ci.yml/badge.svg)](https://github.com/yanpeng0520/av-toolbox/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Upload a video, get visual/audio/AV diagnostics with overlay videos.

**Live demo:** [AV Toolbox Demo](https://demo.yan-peng.com) - choose a Video, Audio, or Audio-Visual tool; use the sample clip or upload a short non-sensitive file; then view/download the overlay MP4, metrics, and artifacts.

`av-toolbox` is an installable audio, video, and audio-visual analysis toolbox with one Python registry, one CLI, and a Streamlit demo UI.

PyPI distribution name: `av-analysis-toolbox` (the import package remains `av_toolbox`, and the CLI remains `av-toolbox`).

## Tool Catalog

See [docs/tool-catalog.md](docs/tool-catalog.md) for detailed per-tool
instructions, CLI and Python examples, UI notes, generated config files, input
types, output artifacts, optional dependency extras, and GPU/model requirements.

## Overlay Examples

The overlays below are rendered on demo footage from [this YouTube video](https://www.youtube.com/watch?v=THjXkQLy4wE). All rights to the original footage remain with its creator; it is included here for demonstration only. See [Credits](#credits).

**Video editing**

| [Cut Detection](docs/tool-catalog.md#video-cut-detection) | [Shot Type](docs/tool-catalog.md#video-shot-type) |
| --- | --- |
| <img src="docs/assets/gallery/video-cut-detection.gif" alt="Cut detection" width="420"> | <img src="docs/assets/gallery/video-shot-type.gif" alt="Shot type" width="420"> |

**Video quality**

| [Image Quality](docs/tool-catalog.md#video-image-quality) | [Camera Shake](docs/tool-catalog.md#video-camera-shake) |
| --- | --- |
| <img src="docs/assets/gallery/video-quality.gif" alt="Image quality" width="420"> | <img src="docs/assets/gallery/video-camera-shake.gif" alt="Camera shake" width="420"> |

**Motion detection**

| [Motion](docs/tool-catalog.md#video-motion) | [Optical Flow](docs/tool-catalog.md#video-optical-flow) |
| --- | --- |
| <img src="docs/assets/gallery/video-motion.gif" alt="Motion" width="420"> | <img src="docs/assets/gallery/video-optical-flow.gif" alt="Optical flow" width="420"> |
| [Foreground Motion](docs/tool-catalog.md#video-foreground-motion) |  |
| <img src="docs/assets/gallery/video-foreground-motion.gif" alt="Foreground motion" width="420"> |  |

**Object and action understanding**

| [Segmentation](docs/tool-catalog.md#video-segmentation) | [Action Recognition](docs/tool-catalog.md#video-action-recognition) |
| --- | --- |
| <img src="docs/assets/gallery/video-segmentation.gif" alt="Segmentation" width="420"> | <img src="docs/assets/gallery/video-action-recognition.gif" alt="Action recognition" width="420"> |
| [Pose Detection](docs/tool-catalog.md#video-pose) |  |
| <img src="docs/assets/gallery/video-pose.gif" alt="Pose detection" width="420"> |  |

**Audio tools**

| [Beat Detection](docs/tool-catalog.md#audio-beat-detection) | [Audio Energy](docs/tool-catalog.md#audio-energy) |
| --- | --- |
| <img src="docs/assets/gallery/audio-beat-detection.gif" alt="Beat detection" width="420"> | <img src="docs/assets/gallery/audio-energy.gif" alt="Audio energy" width="420"> |
| [Audio Events](docs/tool-catalog.md#audio-event-detection) |  |
| <img src="docs/assets/gallery/audio-event-detection.gif" alt="Audio events" width="420"> |  |

**Audio-visual foundation model**

<p>
  <a href="docs/tool-catalog.md#av-denseav">DenseAV on CatFu</a><br>
  <img src="docs/assets/gallery/av-denseav.gif" alt="DenseAV on CatFu" width="720">
</p>

## Happy Path: Local Install And Demo

This path works from a fresh clone without private media, cloud services, or model checkpoints. It needs Python 3.10+ and FFmpeg on `PATH`, installs the local UI plus the lightweight audio/video tools, generates a small synthetic demo clip, runs one CLI tool, and starts the web UI.

```bash
git clone https://github.com/yanpeng0520/av-toolbox.git
cd av-toolbox
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[web,audio,video]"

av-toolbox generate-demo-media --output-dir data_segments --duration 12

av-toolbox video motion \
  data_segments/CatFu.mp4 \
  --output outputs/motion_demo \
  --sample-fps 5 \
  --max-seconds 8

av-toolbox serve \
  --host 127.0.0.1 \
  --port 8501 \
  --output-root outputs/web_runs
```

Open `http://127.0.0.1:8501`, choose a tool, use the generated sample or upload a short local clip, and inspect the overlay, transcript/metrics, and downloadable artifacts in Results.

## Optional Model Tools

Install heavier extras only for the tools you plan to run:

```bash
# YOLO object detection/segmentation/pose and shot-type classification
python -m pip install -e ".[vision-models]"

# TransNetV2/PySceneDetect cut detection backends
python -m pip install -e ".[cut-detection]"

# PyTorchVideo action recognition
python -m pip install -e ".[action]"

# faster-whisper transcription
python -m pip install -e ".[transcription]"
```

DenseAV is a separate heavyweight install because it needs the DenseAV Git package and checkpoint setup:

```bash
python -m pip install -e ".[denseav]"
python -m pip install "git+https://github.com/mhamilton723/DenseAV.git"
```

## GPU And Model Cache

Classical tools run on CPU. Model-backed tools can use GPU when their PyTorch/accelerator stack is installed and the tool supports it.

Recommended cache setup:

```bash
export AV_TOOLBOX_CACHE_DIR=/mnt/models/av_toolbox_cache
mkdir -p "$AV_TOOLBOX_CACHE_DIR/weights"
```

You can also pass `--cache-dir` through CLI/runtime options. The default cache is under `~/.cache/av_toolbox/weights`; if that directory is root-owned or unwritable, set `AV_TOOLBOX_CACHE_DIR` to a writable path before running model-backed tools.

DenseAV checkpoints require explicit setup. See [docs/denseav.md](docs/denseav.md).

## Developer Docs

- [Developer README](docs/developer-readme.md): local development, CLI examples, tests, Docker, Python API, and web UI commands.
- [Tool catalog](docs/tool-catalog.md): registered tools, CLI wrappers, inputs, outputs, and runtime controls.
- [DenseAV setup](docs/denseav.md): optional DenseAV dependencies, checkpoint names, cache paths, and GPU flags.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev
setup, how to add a tool, the overlay style guide, and PR expectations. To report
a security issue, follow [SECURITY.md](SECURITY.md) (please do not open a public
issue).

## Credits

- Demo/sample footage is sourced from [this YouTube video](https://www.youtube.com/watch?v=THjXkQLy4wE) and used solely to demonstrate the tools' overlays. All rights to the original footage belong to its creator. The `av-toolbox` source code is licensed separately under the [MIT License](LICENSE).
