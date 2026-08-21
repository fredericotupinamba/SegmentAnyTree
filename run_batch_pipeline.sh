#!/bin/bash
set -e

# Default docker ENTRYPOINT target (see docker_heartbeat_entrypoint.sh).
# Mounts expected: bucket_in_folder (read) and bucket_out_folder (write) --
# see README.md quick start.
#
# The actual batching, resumability (skips point clouds already finished on
# a previous run) and step-by-step logging live in
# tupisat_inference/batch_orchestrator.py; this script only wires up the
# well-known bind-mount paths.
#
# Note: this used to also support an Oracle Cloud Function deployment
# (OBJ_INPUT_LOCATION/OBJ_OUTPUT_LOCATION bucket remapping, zipped results
# upload). That path is unused and was removed -- local bind mounts are the
# only supported mode now.

RUNNING_DIR="/home/nibio/mutable-outside-world"
IN_FOLDER="$RUNNING_DIR/bucket_in_folder"
OUT_FOLDER="$RUNNING_DIR/bucket_out_folder"

mkdir -p "$IN_FOLDER" "$OUT_FOLDER"

export PYTHONPATH="$RUNNING_DIR:$PYTHONPATH"

python3 "$RUNNING_DIR/tupisat_inference/batch_orchestrator.py" \
    --input-dir "$IN_FOLDER" \
    --output-dir "$OUT_FOLDER" \
    --work-dir "$RUNNING_DIR/work" \
    "$@"
