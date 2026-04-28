#!/bin/bash
# One-command baseline training script

set -e

echo "[+] SynthRiders Baseline Training"
echo "[+] Step 1: Extract features (if not already done)"
if [ ! -d "dataset/features" ] || [ -z "$(ls -A dataset/features)" ]; then
    python3 src/features/feature_engineering.py \
        --parsed-dir dataset/parsed \
        --output-dir dataset/features \
        --difficulty Hard
fi

echo "[+] Step 2: Extract audio features (if not already done)"
if [ ! -d "dataset/audio_features" ] || [ -z "$(ls -A dataset/audio_features)" ]; then
    python3 scripts/extract_audio_batch.py
fi

echo "[+] Step 3: Run fast overfit test"
python3 -u src/training/overfit_fast.py \
    --features-dir dataset/features \
    --num-maps 20 \
    --epochs 100 \
    --max-frames 3000

echo "[+] Step 4: Full baseline training with real audio (CPU/MPS)"
python3 -u src/training/train_baseline.py \
    --features-dir dataset/features \
    --audio-features-dir dataset/audio_features \
    --output-dir models/checkpoints \
    --difficulty Hard \
    --num-maps 100 \
    --epochs 100 \
    --batch-size 4 \
    --device cpu

echo "[+] Done. Check models/checkpoints/ for results."
