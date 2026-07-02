# av-toolbox Quickstart

This is the fresh-clone path for a local public-demo-quality run. It avoids private media, cloud services, and model checkpoints. It needs Python 3.10+ and FFmpeg on `PATH`.

## 1. Install

```bash
git clone https://github.com/yanpeng0520/av-toolbox.git
cd av-toolbox
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[web,audio,video]"
```

After the package is published on PyPI, users who do not need an editable clone can install the released distribution instead:

```bash
python -m pip install "av-analysis-toolbox[web,audio,video]"
```

## 2. Generate Demo Media

```bash
av-toolbox generate-demo-media --output-dir data_segments --duration 12
```

This writes a small synthetic WAV/MP4 pair under `data_segments/`. That directory is ignored by git so local media does not leak into commits.

## 3. Run One CLI Tool

```bash
av-toolbox video motion \
  data_segments/synthetic_hiphop_60s.mp4 \
  --output outputs/demo_motion \
  --sample-fps 5 \
  --max-seconds 8
```

Outputs are written under `outputs/`, also ignored by git. Each run produces standard artifacts such as timeline JSON, feature CSV, report HTML, config YAML, log text, and an overlay MP4 when the tool supports overlays.

## 4. Start The Web UI

```bash
av-toolbox serve \
  --host 127.0.0.1 \
  --port 8501 \
  --output-root outputs/web_runs
```

Open `http://127.0.0.1:8501`. Choose a Video, Audio, or Audio-Visual tool, use the generated sample or upload a short local clip, then inspect Results and download artifacts.

## 5. Optional Model Tools

Install only the extras needed for the tools you want:

```bash
python -m pip install -e ".[vision-models]"     # YOLO object/segmentation/pose and shot-type models
python -m pip install -e ".[cut-detection]"     # TransNetV2/PySceneDetect
python -m pip install -e ".[action]"            # PyTorchVideo action recognition
python -m pip install -e ".[transcription]"     # faster-whisper transcription
```

For model-backed tools, use a writable cache directory:

```bash
export AV_TOOLBOX_CACHE_DIR="$HOME/.cache/av_toolbox"
mkdir -p "$AV_TOOLBOX_CACHE_DIR/weights"
```

DenseAV is heavier and needs the DenseAV Git package plus checkpoints:

```bash
python -m pip install -e ".[denseav]"
python -m pip install "git+https://github.com/mhamilton723/DenseAV.git"
```

See [denseav.md](denseav.md) for checkpoint names, cache paths, and GPU notes.

## Smoke Tests

Install the lightweight test dependency first if you did not include `dev` during setup:

```bash
python -m pip install -e ".[dev]"
av-toolbox list-tools
python -m pytest tests/test_import.py tests/test_registry.py tests/test_cli.py
```

The full test suite is available with `python -m pytest`, but model-backed tests may need optional extras or cached weights.

## Public Demo Mode

Use bounded upload/runtime controls when exposing the UI beyond localhost:

```bash
av-toolbox serve \
  --host 127.0.0.1 \
  --port 8501 \
  --output-root /srv/av-toolbox-demo/runs \
  --public-demo \
  --public-max-seconds 20 \
  --public-max-upload-mb 10
```

Use `--public-enable-denseav` only after DenseAV dependencies and weights are installed.

## Docker

```bash
docker build -t av-toolbox .
docker run --rm -p 8501:8501 \
  -v "$PWD/data_segments:/app/data_segments" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$HOME/.cache/av_toolbox:/cache/av_toolbox" \
  av-toolbox
```

Do not bake large media, generated outputs, logs, or model weights into the image.

## More Docs

- [tool-catalog.md](tool-catalog.md): registry names, CLI commands, artifacts, optional dependencies.
- [examples.md](examples.md): Python API and CLI batch examples.
- [denseav.md](denseav.md): DenseAV checkpoint/cache setup and GPU notes.
