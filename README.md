# av-toolbox

Upload a video, get visual/audio/AV diagnostics with overlay videos.

[Open the live demo](https://demo.yan-peng.com)

`av-toolbox` is an installable audio, video, and audio-visual analysis toolbox with one Python registry, one CLI, and a Streamlit demo UI. It turns short media clips into overlay MP4s, timeline JSON, feature CSVs, and HTML reports.

## Overlay Examples

**Video editing**

| Cut Detection | Shot Type |
| --- | --- |
| ![Cut detection](docs/assets/gallery/video-cut-detection.gif) | ![Shot type](docs/assets/gallery/video-shot-type.gif) |

**Video quality**

| Image Quality | Camera Shake |
| --- | --- |
| ![Image quality](docs/assets/gallery/video-quality.gif) | ![Camera shake](docs/assets/gallery/video-camera-shake.gif) |

**Motion detection**

| Motion | Optical Flow | Foreground Motion |
| --- | --- | --- |
| ![Motion](docs/assets/gallery/video-motion.gif) | ![Optical flow](docs/assets/gallery/video-optical-flow.gif) | ![Foreground motion](docs/assets/gallery/video-foreground-motion.gif) |

**Object and action understanding**

| Object Detection | Segmentation | Action Recognition | Pose Detection |
| --- | --- | --- | --- |
| ![Object detection](docs/assets/gallery/video-object-detection.gif) | ![Segmentation](docs/assets/gallery/video-segmentation.gif) | ![Action recognition](docs/assets/gallery/video-action-recognition.gif) | ![Pose detection](docs/assets/gallery/video-pose.gif) |

**Audio and audio-visual tools**

| Beat Detection | Audio Energy | Audio Events | AV Sync |
| --- | --- | --- | --- |
| ![Beat detection](docs/assets/gallery/audio-beat-detection.gif) | ![Audio energy](docs/assets/gallery/audio-energy.gif) | ![Audio events](docs/assets/gallery/audio-event-detection.gif) | ![AV sync](docs/assets/gallery/av-sync-correspondence.gif) |

## Try The Demo

1. Open <https://demo.yan-peng.com>.
2. Choose a tool type: Video, Audio, or Audio-Visual.
3. Pick a tool, use the sample clip or upload a short non-sensitive file, then click Run.
4. Inspect the input, output overlay, summary metrics, and downloadable artifacts.

The public demo runs Streamlit on a DGX Spark origin behind Cloudflare Tunnel. Public mode keeps uploads bounded, hides server filesystem paths, and exposes only demo-safe controls. Please do not upload private or sensitive media.

## Install

Minimum editable install:

```bash
git clone https://github.com/yanpeng0520/av-toolbox.git
cd av-toolbox
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Typical local/web install:

```bash
python -m pip install -e ".[web,video,audio]"
```

Install optional model-backed tools as needed:

```bash
python -m pip install -e ".[vision-models,cut-detection,pose,action,transcription]"
```

DenseAV is heavier because it installs from the DenseAV Git repository:

```bash
python -m pip install -e ".[denseav]"
```

A full development install is available, but it pulls heavy model stacks:

```bash
python -m pip install -e ".[all]"
```

## Quick Start

Run one tool from the CLI:

```bash
av-toolbox run video.motion \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/motion_demo \
  --sample-fps 5 \
  --max-seconds 10
```

Run the local Streamlit UI:

```bash
av-toolbox serve \
  --host 127.0.0.1 \
  --port 8501 \
  --output-root outputs/web_runs \
  --page-title "AV Toolbox Demo"
```

## GPU And Model Cache

Classical tools run on CPU. Model-backed tools can use GPU when their PyTorch/accelerator stack is installed and the tool supports it.

Recommended cache setup:

```bash
export AV_TOOLBOX_CACHE_DIR=/mnt/models/av_toolbox_cache
mkdir -p "$AV_TOOLBOX_CACHE_DIR"
```

You can also pass `--cache-dir` through CLI/runtime options. The default cache is under `~/.cache/av_toolbox/weights`; if that directory is root-owned or unwritable, set `AV_TOOLBOX_CACHE_DIR` to a writable path before running model-backed tools.

DenseAV checkpoints require explicit setup. See [docs/denseav.md](docs/denseav.md).

## Tool Catalog

See [docs/tool-catalog.md](docs/tool-catalog.md) for every registered tool,
input type, output artifacts, optional dependency extras, GPU/model
requirements, CLI forms, and artifact naming rules.

## Public Deployment

The public deployment path is:

```text
GitHub README -> https://demo.yan-peng.com -> Cloudflare Tunnel -> Streamlit on DGX Spark -> av_toolbox + GPU
```

The Streamlit origin should stay bound to `127.0.0.1`; Cloudflare Tunnel provides the public URL. See [docs/cloudflare-demo.md](docs/cloudflare-demo.md) for tunnel, public-demo mode, systemd, and hardening notes.

## Developer Docs

- [Developer README](docs/developer-readme.md): local development, CLI examples, tests, Docker, Python API, and web UI commands.
- [Tool catalog](docs/tool-catalog.md): registered tools, CLI wrappers, inputs, outputs, and runtime controls.
- [DenseAV setup](docs/denseav.md): optional DenseAV dependencies, checkpoint names, cache paths, and GPU flags.
- [DGX + Cloudflare deployment](docs/cloudflare-demo.md): public demo mode, tunnel routing, and hardening notes.
