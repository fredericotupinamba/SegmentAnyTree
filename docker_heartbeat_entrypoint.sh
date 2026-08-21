#!/bin/bash
# Wraps the real entrypoint with a background heartbeat so `docker logs` shows
# independent proof that the container is alive and doing GPU work, instead of
# relying only on tqdm/print output that may be buffered or silent for long
# stretches. Each line is unbuffered bash output, written every 30s regardless
# of what the Python process is doing. It also surfaces the batch
# orchestrator's live progress pointer (tupisat_inference/batch_state.py),
# so "which point cloud / which step is running right now" is visible from
# `docker logs` alone even during a long silent step like eval.py.
set -e

CURRENT_STATE_FILE="/home/nibio/mutable-outside-world/bucket_out_folder/.sat_state/current.json"

(
  while true; do
    gpu_line=$(nvidia-smi --query-gpu=pstate,utilization.gpu,power.draw,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable")
    progress_line=""
    if [ -f "$CURRENT_STATE_FILE" ]; then
        progress_line=$(python3 -c "
import json, sys
try:
    with open('$CURRENT_STATE_FILE') as f:
        data = json.load(f)
    current = data.get('current')
    if current:
        print(f\" progress: {current['index']}/{current['total']} file={current['file']} step={current['step']}\")
except Exception:
    pass
" 2>/dev/null || true)
    fi
    echo "[heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ)] gpu: ${gpu_line}${progress_line}"
    sleep 30
  done
) &

exec bash run_batch_pipeline.sh "$@"
