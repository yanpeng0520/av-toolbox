# Contributing to av-toolbox

Thanks for your interest in improving `av-toolbox`! This project is an installable
audio / video / audio-visual analysis toolbox with a single Python registry that
backs the Python API, the `av-toolbox` CLI, and the Streamlit UI.

## Development setup

You need Python 3.10+ and FFmpeg on your `PATH`.

```bash
git clone https://github.com/yanpeng0520/av-toolbox.git
cd av-toolbox
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,web,audio,video]"
```

Install additional extras only for the tools you touch (e.g. `vision-models`,
`cut-detection`, `action`, `transcription`, `denseav`). See the README for the
full list.

## Tests and linting

```bash
python -m pytest          # run the test suite
ruff check .              # lint
ruff format .            # auto-format
```

- CI runs `python -m pytest` on Python 3.10 / 3.11 / 3.12 plus a Docker build
  smoke test. Please make sure tests pass locally before opening a PR.
- Model-backed tests may skip or require network/optional extras; the classical
  audio/video tests run on CPU with no downloads.

## Project layout

- `src/av_toolbox/core/` — shared infrastructure: `BaseTool`, `AVResult`,
  `ToolRegistry`, media I/O, and the `write_standard_artifacts` helper.
- `src/av_toolbox/video|audio|av/` — the tools, one module per tool.
- `src/av_toolbox/builtins.py` — registers the built-in tools.
- `src/av_toolbox/cli.py`, `web_app.py`, `web_server.py`, `ui_defaults.py` — the
  CLI and web surfaces. **Route everything through the registry and
  `av_toolbox.run_tool`; do not duplicate algorithm logic in the CLI or UI.**

## Adding or changing a tool

1. Add a module under the right category directory. Subclass `BaseTool`, set
   `name` (dot style, e.g. `video.my_tool`), `category`, and `description`, and
   implement `_run(...)` returning an `AVResult` — reuse
   `core.simple_outputs.write_standard_artifacts` for JSON/CSV/report/config/log.
2. Register it in `builtins.py` and export it from the category `__init__.py`.
3. Add a CLI subcommand + name mapping in `cli.py`, and (optionally) a workflow
   entry in `ui_defaults.py`.
4. Add tests (registration, declared artifacts, and any pure helpers).
5. Update `docs/tool-catalog.md`.

Use seconds as the time unit in JSON/CSV outputs, and keep heavy dependencies
optional (fail gracefully / skip tests when a model dependency is missing).

## Overlay style (dark-slate house style)

Video overlays share one look so the gallery stays consistent. When you add or
restyle a `video.*` overlay:

- Keep the raw video clean except a small translucent top-left card/badge.
- Put a dark-slate timeline panel **under** the video; draw a single amber
  playhead **inside the panel only** (never across the video).
- Use thin anti-aliased line charts for time-varying values — **no gridlines and
  no filled/shaded areas**; show a threshold as a short axis tick.
- Threshold-aware scalars use green (good) / red (failing) fills; multi-metric
  tools use one colored line per metric plus a small legend.
- Each tool owns a dedicated `_render_<tool>_overlay`; reuse only
  `source_overlay.mux_video_overlay_with_source_audio` for audio. Do not restyle
  the shared `render_source_video_overlay` (box/pose/mask tools depend on it).
- Overlays must cover the full source duration and preserve source audio; export
  browser-friendly H.264 / `yuv420p` / AAC with fast-start.

Reference implementations: `video.camera_shake`, `video.image_quality`,
`video.foreground_motion`, `video.cut_detection`, `video.pose`.

## Pull requests

- Keep PRs small and focused; describe the change and how you verified it.
- Add or update tests and docs for any public behavior (registry names, CLI
  commands, output schemas, artifact naming).
- Do not commit media, model weights, or generated outputs — they are gitignored.
  Regenerate gallery GIFs with `scripts/make_gallery_gif.sh` (they are Git-LFS
  tracked under `docs/assets/gallery/`).

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
