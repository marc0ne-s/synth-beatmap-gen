#!/bin/bash
# Autoresearch runner — one experiment iteration
# Called by cron or background loop

set -e

cd /Volumes/Second-Brain-1/AI
BRANCH="autoresearch/synth-apr27"
EXPERIMENT_DIR="Synth/autoresearch"
TIMESTAMP=$(date +%s)
LOGFILE="${EXPERIMENT_DIR}/run_${TIMESTAMP}.log"

echo "=== Autoresearch Iteration: $(date) ===" | tee -a "$LOGFILE"

# Ensure we're on the right branch
git checkout "$BRANCH" 2>/dev/null || true

# Check current best score from results.tsv
if [ -f "${EXPERIMENT_DIR}/results.tsv" ]; then
    BEST_SCORE=$(tail -n +2 "${EXPERIMENT_DIR}/results.tsv" | awk -F'\t' 'NR==1 || $2 > best {best=$2} END {print best}')
    echo "Current best score: ${BEST_SCORE:-0.0}" | tee -a "$LOGFILE"
else
    echo "No results yet — establishing baseline" | tee -a "$LOGFILE"
    BEST_SCORE="0.0"
fi

# Run training
cd /Volumes/Second-Brain-1/AI/Synth
python autoresearch/synth_train.py > "$LOGFILE" 2>&1

# Extract score
SCORE=$(grep "^score:" "$LOGFILE" | awk '{print $2}')
RECALL=$(grep "^val_recall_50:" "$LOGFILE" | awk '{print $2}')
MSE=$(grep "^val_position:" "$LOGFILE" | awk '{print $2}')

echo "Results — Score: ${SCORE}, Recall: ${RECALL}, MSE: ${MSE}" | tee -a "$LOGFILE"

# Commit or reset
if (( $(echo "$SCORE > $BEST_SCORE" | bc -l) )); then
    echo "IMPROVEMENT — Keeping experiment" | tee -a "$LOGFILE"
    git add "${EXPERIMENT_DIR}/synth_train.py" "${EXPERIMENT_DIR}/results.tsv" 2>/dev/null || true
    git commit -m "Autoresearch: score ${SCORE} (recall ${RECALL}, mse ${MSE})"
else
    echo "No improvement — Reverting" | tee -a "$LOGFILE"
    git reset --hard HEAD
fi

echo "=== Iteration complete ===" | tee -a "$LOGFILE"
