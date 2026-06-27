# av-toolbox Quickstart

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[audio,video,av,dev]"
```

Use the heavier DenseAV extra only when you need DenseAV inference:

```bash
python -m pip install -e ".[denseav]"
mkdir -p ~/.cache/av_toolbox/weights
```

DenseAV checkpoints are not committed to the repository. By default,
`av.denseav` expects weights under `~/.cache/av_toolbox/weights/`, or you can
pass a local checkpoint with `--checkpoint`.

## Smoke Test

```bash
av-toolbox list-tools
python -m pytest
```

The test suite is configured to collect the packaged `tests/` directory only.
Legacy migration scripts under `testing/` are not part of CI.

## Generate Demo Media

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
  data_segments/synthetic_hiphop_60s.mp4 \
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
back to the built-in local web UI.

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
