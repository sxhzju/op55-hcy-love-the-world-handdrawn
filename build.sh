#!/usr/bin/env bash
# Render op55-hd.mp4 from subtitles.srt + original.wav (audio is not in the repo).
set -euo pipefail
cd "$(dirname "$0")"
uv sync --frozen
uv run python src/render.py --srt subtitles.srt --audio original.wav --out op55-hd.mp4
# 640px-long-edge preview: 24fps, H.264 400k, AAC 64k
ffmpeg -y -hide_banner -loglevel error -i op55-hd.mp4 \
  -vf "scale=480:640:flags=lanczos,fps=24" -c:v libx264 -preset slow -b:v 400k -maxrate 450k -bufsize 800k \
  -pix_fmt yuv420p -c:a aac -b:a 64k -movflags +faststart preview.mp4
