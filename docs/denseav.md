# DenseAV Setup

`av.denseav` is optional because it depends on PyTorch, DenseAV, model
checkpoints, and more VRAM than the classical video/audio tools. The base
package does not download or commit DenseAV weights.

## Install Dependencies

From the repository root:

```bash
python -m pip install -e ".[denseav]"
```

The DenseAV extra installs PyAV, Pillow, PyTorch packages, and DenseAV from its
GitHub source declared in `pyproject.toml`.

## Checkpoint Names And Locations

Default cache root:

```text
~/.cache/av_toolbox/
```

Default weights directory:

```text
~/.cache/av_toolbox/weights/
```

Expected DenseAV checkpoint files:

| `model_name` | Checkpoint filename | Default path |
| --- | --- | --- |
| `sound_and_language` | `denseav_2head.ckpt` | `~/.cache/av_toolbox/weights/denseav_2head.ckpt` |
| `sound` | `denseav_sound.ckpt` | `~/.cache/av_toolbox/weights/denseav_sound.ckpt` |

Create the cache directory:

```bash
mkdir -p ~/.cache/av_toolbox/weights
```

Then copy checkpoints from your DenseAV source, production cache, or model
artifact store into that directory. Do not commit checkpoints into this repo and
do not bake them into the default Docker image.

## Cache Overrides

Use a different cache root with an environment variable:

```bash
export AV_TOOLBOX_CACHE_DIR=/mnt/models/av_toolbox_cache
```

Or pass it per run:

```bash
av-toolbox av denseav input.mp4 \
  --output outputs/denseav \
  --cache-dir /mnt/models/av_toolbox_cache
```

Python:

```python
import av_toolbox

result = av_toolbox.run_tool(
    "av.denseav",
    input_path="input.mp4",
    output_dir="outputs/denseav",
    cache_dir="/mnt/models/av_toolbox_cache",
)
```

## Explicit Checkpoint Path

You can use a checkpoint without copying it into the cache:

```bash
av-toolbox av denseav \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/denseav_explicit \
  --checkpoint /path/to/denseav_2head.ckpt \
  --model-name sound_and_language \
  --max-seconds 5 \
  --sample-fps 5
```

Relative checkpoint paths are checked relative to the current working directory
and the configured cache weights directory.

## Verify A Cached Checkpoint

```bash
python - <<'PY'
from av_toolbox.core.cache import ModelCache

cache = ModelCache()
for name in ("denseav_2head.ckpt", "denseav_sound.ckpt"):
    path = cache.weights_dir / name
    print(path, "exists" if path.exists() else "missing")
    if path.exists():
        print(ModelCache.sha256(path))
PY
```

Python callers can pass `expected_sha256="..."` to `av_toolbox.run_tool(...)`
for checksum verification. The CLI currently resolves local files but does not
expose a checksum flag.

## CPU Smoke Run

```bash
av-toolbox av denseav \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/denseav_cpu_smoke \
  --model-name sound_and_language \
  --max-seconds 2 \
  --sample-fps 2 \
  --plot-size 720 \
  --device cpu
```

`--load-size` controls model inference resolution and defaults to `224`.
`--plot-size` controls rendered attention video size and defaults to `720`.

## GPU Run

```bash
av-toolbox av denseav \
  data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4 \
  --output outputs/denseav_gpu \
  --model-name sound_and_language \
  --max-seconds 8 \
  --sample-fps 5 \
  --device cuda:0 \
  --batch-size 4 \
  --fp16
```

Use smaller `--max-seconds`, `--sample-fps`, `--load-size`, or `--batch-size`
when VRAM is tight. `--fp16` only activates autocast on CUDA devices.

## Offline Mode

```bash
av-toolbox av denseav input.mp4 \
  --output outputs/denseav_offline \
  --offline
```

Offline mode makes missing cached weights fail explicitly. The current resolver
does not download DenseAV weights; it only resolves and verifies local files.

## Troubleshooting

- `ImportError: av.denseav requires ...`: install `python -m pip install -e ".[denseav]"`.
- `model weight not found in cache`: copy the expected checkpoint into `~/.cache/av_toolbox/weights/`, set `AV_TOOLBOX_CACHE_DIR`, pass `--cache-dir`, or pass `--checkpoint`.
- `Requested DenseAV device 'cuda:0', but CUDA is unavailable`: use `--device cpu` or run on a CUDA-enabled machine.
- No attention overlay: sample at least two frames by increasing `--max-seconds` or `--sample-fps`.
- Large outputs: keep `--include-sim-matrix` off unless you need the dense similarity array in JSON.
