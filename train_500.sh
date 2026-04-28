#!/bin/bash
# Train baseline model on 500 maps with proper train/val split
# Launched: $(date)

cd /Volumes/Second-Brain-1/AI/Synth

python3 src/training/train_baseline.py \
    --features-dir dataset/features \
    --audio-features-dir dataset/audio_features \
    --output-dir models/checkpoints \
    --difficulty Hard \
    --num-maps 500 \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-3 \
    --device mps \
    --train-split 0.8 \
    2>&1 | tee models/checkpoints/train_500.log
