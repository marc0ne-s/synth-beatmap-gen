#!/bin/bash
# Train beatmap model with focal loss
# Fast iteration: small model (1.1M params), 100 maps, 50 epochs

cd /Volumes/Second-Brain-1/AI/Synth

python3 src/training/train_focal.py \
    --features-dir dataset/features \
    --audio-features-dir dataset/audio_features \
    --output-dir models/checkpoints \
    --difficulty Hard \
    --num-maps 100 \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3 \
    --device mps \
    --train-split 0.8 \
    --alpha 0.25 \
    --gamma 2.0 \
    2>&1 | tee models/checkpoints/train_focal.log
