# av-toolbox Quickstart

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[audio,video,av,dev]"
```

Model-backed migrated tools are optional too: use `.[transcription]` for Whisper, `.[vision-models]` for YOLO and shot-type classification, `.[pose]` for MediaPipe, `.[action]` for PyTorchVideo action recognition, and `.[cut-detection]` for PySceneDetect.

Use the heavier DenseAV extra only when you need DenseAV inference:

```bash
python -m pip install -e ".[denseav]"
mkdir -p ~/.cache/av_toolbox/weights
```

DenseAV checkpoints are not committed to the repository. By default,
`av.denseav` expects weights under `~/.cache/av_toolbox/weights/`, or you can
pass a local checkpoint with `--checkpoint`.

Expected default DenseAV checkpoint paths:

```text
~/.cache/av_toolbox/weights/denseav_2head.ckpt
~/.cache/av_toolbox/weights/denseav_sound.ckpt
```

## Smoke Test

```bash
av-toolbox list-tools
python -m pytest
```

The test suite is configured to collect the packaged `tests/` directory only.
Legacy migration scripts under `testing/` are not part of CI.

## Demo Media

Curated sample videos live in Git LFS. Pull them after cloning:

```bash
git lfs install
git lfs pull --include="data_segments/*.mp4"
```

You can also generate synthetic audio/video media locally:

```bash
av-toolbox generate-demo-media --output-dir data_segments --duration 60
```

## Run Tools

```bash
av-toolbox video motion \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_motion \
  --sample-fps 5

av-toolbox audio beat-detection \
  data_segments/synthetic_hiphop_60s.wav \
  --output outputs/demo_beats

av-toolbox av sync-correspondence \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_sync
```

## DenseAV

```bash
av-toolbox av denseav \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/demo_denseav \
  --checkpoint /path/to/denseav_2head.ckpt \
  --sample-fps 25 \
  --plot-size 720 \
  --device cuda:0 \
  --batch-size 4 \
  --fp16
```

`--load-size` controls model inference resolution. `--plot-size` controls the
rendered attention MP4 resolution.

## Web UI

```bash
av-toolbox serve --host 127.0.0.1 --port 8501
```

If Streamlit is installed, `serve` starts the Streamlit app. Otherwise it falls
back to the built-in local web UI. Both UIs open with the generated synthetic
demo clip and default outputs under `outputs/web_runs/latest`. Tool parameters
use each tool's built-in defaults unless Advanced overrides are enabled; when
you select a tool, Advanced shows that tool's own default parameter space.

Public demo mode for a Cloudflare Tunnel origin on the DGX Spark:

```bash
av-toolbox serve \
  --host 127.0.0.1 \
  --port 8501 \
  --output-root /srv/av-toolbox-demo/runs \
  --public-demo \
  --public-max-seconds 20 \
  --public-max-upload-mb 100
```

Use `--public-enable-denseav` only when the DGX has DenseAV dependencies and
weights installed. See [cloudflare-demo.md](cloudflare-demo.md).

## Docker

```bash
docker build -t av-toolbox .
docker run --rm -p 8501:8501 \
  -v "$PWD/data_segments:/app/data_segments" \
  -v "$PWD/outputs:/app/outputs" \
  -v "$HOME/.cache/av_toolbox:/cache/av_toolbox" \
  av-toolbox
```

Build with DenseAV dependencies:

```bash
docker build -t av-toolbox:denseav --build-arg INCLUDE_DENSEAV=1 .
```

Keep model weights in the mounted cache or pass explicit checkpoint paths. Do
not bake large weights into the image.

## More Docs

- [docs/tool-catalog.md](tool-catalog.md): registry names, CLI commands, and artifact outputs.
- [docs/denseav.md](denseav.md): DenseAV checkpoint/cache setup and GPU notes.
- [docs/examples.md](examples.md): Python API and CLI batch examples.
- [docs/cloudflare-demo.md](cloudflare-demo.md): Cloudflare Tunnel and DGX Spark public demo deployment.
