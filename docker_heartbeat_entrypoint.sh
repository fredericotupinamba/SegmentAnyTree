#!/bin/bash
# Wraps the real entrypoint with a background heartbeat so `docker logs` shows
# independent proof that the container is alive and doing GPU work, instead of
# relying only on tqdm/print output that may be buffered or silent for long
# stretches. Each line is unbuffered bash output, written every 30s regardless
# of what the Python process is doing.
set -e

(
  while true; do
    gpu_line=$(nvidia-smi --query-gpu=pstate,utilization.gpu,power.draw,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable")
    echo "[heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ)] gpu: ${gpu_line}"
    sleep 30
  done
) &

exec bash run_oracle_pipeline.sh "$@"
