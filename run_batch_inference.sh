#!/bin/bash
set -e

# Manual/interactive entry point for the batch orchestrator, e.g. when
# exec'ing into a running container via run_bash_in_podman_with_gpu.sh.
# The actual batching/resumability/logging logic lives in
# tupisat_inference/batch_orchestrator.py -- this is just an argument-passing
# wrapper so it is not reimplemented a second time in bash.
#
# Usage: run_batch_inference.sh [input_dir] [output_dir] [-- extra orchestrator args]

RUNNING_DIR="/home/nibio/mutable-outside-world"

INPUT_DIR="${1:-$RUNNING_DIR/bucket_in_folder}"
OUTPUT_DIR="${2:-$RUNNING_DIR/bucket_out_folder}"
shift $(( $# >= 2 ? 2 : $# )) || true

export PYTHONPATH="$RUNNING_DIR:$PYTHONPATH"

exec python3 "$RUNNING_DIR/tupisat_inference/batch_orchestrator.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --work-dir "$RUNNING_DIR/work" \
    "$@"
