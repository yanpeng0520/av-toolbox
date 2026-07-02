#!/usr/bin/env bash
# Generate a README gallery GIF from an overlay MP4.
# Matches the docs/assets/gallery format: 320px wide, 6.25 fps, ~6s, palette-optimized.
#
# Usage: scripts/make_gallery_gif.sh <overlay.mp4> <output.gif> [start_seconds] [duration_seconds]
set -euo pipefail

IN="${1:?usage: make_gallery_gif.sh <overlay.mp4> <output.gif> [start] [duration]}"
OUT="${2:?output gif path required}"
SS="${3:-0}"
DUR="${4:-6}"

PAL="$(mktemp --suffix=.png)"
trap 'rm -f "$PAL"' EXIT

FILT="fps=6.25,scale=320:-2:flags=lanczos"
ffmpeg -y -loglevel error -ss "$SS" -t "$DUR" -i "$IN" -vf "${FILT},palettegen=max_colors=96:stats_mode=diff" "$PAL"
ffmpeg -y -loglevel error -ss "$SS" -t "$DUR" -i "$IN" -i "$PAL" \
  -lavfi "${FILT}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" "$OUT"

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
