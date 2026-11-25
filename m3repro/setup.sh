#!/usr/bin/env bash
# Fetch the released M3 artifacts (code only used as reference; nothing from it is executed) + CPU deps.
set -euo pipefail
EXT=${EXT:-/home/user/ext}
mkdir -p "$EXT"
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q "transformers>=4.50" numpy huggingface_hub
[ -d "$EXT/m3-market-microstructure" ] || git clone --depth 1 https://github.com/ArthurZhang02/m3-market-microstructure "$EXT/m3-market-microstructure"
cd "$EXT/m3-market-microstructure"
hf download Arthur210/M3-market-microstructure tiny/best.pt small/best.pt base/best.pt tokenizer/base/best.pt --local-dir .
