#!/usr/bin/env python3
"""Batch point-cloud inference orchestrator.

Replaces the old bash batch loops (`run_oracle_pipeline.sh`,
`run_batch_inference.sh`). Processes one point cloud at a time (the model
pipeline is not safe to run in parallel/in-process -- see
Dockerfile.pandas-fix for the OOM/CUDA-deadlock history), but adds what the
bash loops never had:

- Resumability: a JSON manifest in the output folder tracks per-file status,
  so a container restart only reprocesses what did not finish.
- Observability: every step transition is logged with the current file
  index/name/step, and a `current.json` pointer is kept up to date so
  external tooling (or the docker heartbeat) can report live progress
  without parsing logs.
- Failure isolation: a failure on one point cloud is recorded and the batch
  moves on to the next one, instead of the whole run dying.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# All the per-step scripts below are invoked as subprocesses and themselves
# do `from tupisat_inference.xxx import yyy`, so REPO_ROOT must be on
# PYTHONPATH for every child process we spawn, regardless of whether the
# caller already set it (matches run_inference.sh's `export PYTHONPATH`).
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if REPO_ROOT not in _existing_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (REPO_ROOT, _existing_pythonpath) if p
    )

from tupisat_inference.batch_state import StateStore  # noqa: E402

SUPPORTED_EXTENSIONS = (".las", ".laz", ".ply")
TAIL_LINES = 40


def log(level, **fields):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    parts = [f"{ts} {level:<5}"]
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value)
        if " " in text:
            text = f'"{text}"'
        parts.append(f"{key}={text}")
    print(" ".join(parts), flush=True)


class StepError(RuntimeError):
    pass


def run_subprocess(cmd, cwd=None, env=None):
    """Run a command, streaming its output into our own log, and raise
    StepError with the tail of the output if it fails."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    tail = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line:
            print(f"    | {line}", flush=True)
            tail.append(line)
            if len(tail) > TAIL_LINES:
                tail.pop(0)
    proc.wait()
    if proc.returncode != 0:
        raise StepError(
            f"command {cmd!r} exited with {proc.returncode}\n" + "\n".join(tail)
        )


def discover_input_files(input_dir, extract_dir):
    """Return a sorted list of (file_id, source_path). file_id is stable
    across runs and used as the state-manifest key. .zip archives found at
    the top level of input_dir are extracted (without mutating input_dir)
    and their supported contents are included too."""
    items = []

    for entry in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, entry)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            items.append((entry, full_path))
        elif ext == ".zip":
            dest = os.path.join(extract_dir, os.path.splitext(entry)[0])
            os.makedirs(dest, exist_ok=True)
            with zipfile.ZipFile(full_path) as zf:
                zf.extractall(dest)
            for root, _dirs, files in os.walk(dest):
                for name in sorted(files):
                    if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                        rel = os.path.relpath(os.path.join(root, name), dest)
                        file_id = f"{os.path.splitext(entry)[0]}/{rel}".replace(os.sep, "/")
                        items.append((file_id, os.path.join(root, name)))

    return items


def safe_slug(file_id):
    """Turn a file_id into a filesystem-safe directory name that does not
    itself contain a point-cloud extension (.las/.laz/.ply). Some of the
    pipeline scripts derive output paths with a naive
    path.replace('.las', '.laz') instead of an extension-aware swap, which
    silently mangles any '.las'/'.laz' substring earlier in the path -- see
    pandas_to_las.py. Keeping those extensions out of the work-dir name
    avoids re-triggering that class of bug from this side."""
    slug = file_id.replace("/", "__").replace(os.sep, "__")
    stem, ext = os.path.splitext(slug)
    if ext.lower() in SUPPORTED_EXTENSIONS:
        slug = stem + ext.lower().replace(".", "_dot_")
    return slug


def process_one_file(file_id, source_path, output_dir, work_root, index, total, state):
    """Run the full single-file pipeline (ported from run_inference.sh),
    logging and checkpointing state at each step."""
    file_dir = os.path.join(work_root, safe_slug(file_id))
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
    os.makedirs(file_dir)

    def step(name):
        state.mark_step(file_id, name, index, total)
        log("INFO", idx=f"{index}/{total}", file=file_id, step=name, status="running")

    input_data_dir = os.path.join(file_dir, "input_data")
    utm2local_dir = os.path.join(file_dir, "utm2local")
    eval_yaml_path = os.path.join(file_dir, "eval.yaml")
    final_dir = os.path.join(file_dir, "final_results")

    state.mark_running(file_id, "prepare_workdir", index, total)
    log("INFO", idx=f"{index}/{total}", file=file_id, step="prepare_workdir", status="running")
    os.makedirs(input_data_dir, exist_ok=True)
    shutil.copy2(source_path, os.path.join(input_data_dir, os.path.basename(source_path)))

    step("fix_naming")
    run_subprocess([sys.executable,
                     os.path.join(REPO_ROOT, "tupisat_inference", "fix_naming_of_input_files.py"),
                     input_data_dir])

    step("utm2local")
    run_subprocess([sys.executable,
                     os.path.join(REPO_ROOT, "tupisat_inference", "pipeline_utm2local_parallel.py"),
                     "-i", input_data_dir, "-o", utm2local_dir])

    step("prepare_eval_config")
    shutil.copy2(os.path.join(REPO_ROOT, "conf", "eval.yaml"), eval_yaml_path)
    run_subprocess([sys.executable,
                     os.path.join(REPO_ROOT, "tupisat_inference", "modify_eval.py"),
                     eval_yaml_path, utm2local_dir, file_dir])

    step("clear_cache")
    run_subprocess([sys.executable,
                     os.path.join(REPO_ROOT, "tupisat_inference", "clear_cache.py"),
                     "--eval_yaml", eval_yaml_path])

    step("inference")
    run_subprocess([sys.executable, "eval.py", "--config-name", eval_yaml_path], cwd=REPO_ROOT)

    step("rename_results")
    run_subprocess([sys.executable,
                     os.path.join(REPO_ROOT, "tupisat_inference", "rename_result_files_instance.py"),
                     eval_yaml_path, file_dir])
    run_subprocess([sys.executable,
                     os.path.join(REPO_ROOT, "tupisat_inference", "rename_result_files_segmentation.py"),
                     eval_yaml_path, file_dir])

    step("merge")
    run_subprocess([sys.executable,
                     os.path.join(REPO_ROOT, "tupisat_inference", "merge_pt_ss_is_in_folders.py"),
                     "-i", utm2local_dir, "-s", file_dir, "-o", final_dir, "-v"])

    merged_files = [f for f in glob.glob(os.path.join(final_dir, "*"))
                     if os.path.splitext(f)[1].lower() in (".las", ".laz")]
    if not merged_files:
        raise StepError(f"merge step produced no point cloud in {final_dir}")
    merged_las_path = merged_files[0]
    merged_stem = os.path.splitext(os.path.basename(merged_las_path))[0]

    # Everything downstream of this point -- the segmented point cloud
    # itself and every forest_metrics output -- lands together in one
    # <name>_SAT_output/ folder, mirroring FSCT's <name>_FSCT_output/
    # convention, so a user gets one self-contained result per input file.
    sat_output_dir = os.path.join(final_dir, f"{merged_stem}_SAT_output")
    os.makedirs(sat_output_dir, exist_ok=True)
    segmented_las_path = os.path.join(sat_output_dir, os.path.basename(merged_las_path))
    shutil.move(merged_las_path, segmented_las_path)

    step("forest_metrics")
    try:
        run_subprocess([sys.executable,
                         os.path.join(REPO_ROOT, "tupisat_inference", "forest_metrics", "forest_metrics.py"),
                         "--input-las", segmented_las_path,
                         "--output-dir", sat_output_dir,
                         "--stem", merged_stem])
    except StepError as exc:
        # A metrics-computation failure must not cost the user their already-
        # produced segmented point cloud -- log and continue rather than abort.
        log("WARN", idx=f"{index}/{total}", file=file_id, step="forest_metrics",
            status="failed", msg=str(exc)[:500])

    step("finalize")
    if not os.listdir(sat_output_dir):
        raise StepError(f"no output produced in {sat_output_dir}")
    result_name = f"{merged_stem}_SAT_output"
    dest_dir = os.path.join(output_dir, result_name)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(sat_output_dir, dest_dir)
    if not os.listdir(dest_dir):
        raise StepError(f"output folder {dest_dir} is empty")

    state.mark_done(file_id, result_name)
    shutil.rmtree(file_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Batch point-cloud inference orchestrator.")
    parser.add_argument("--input-dir", default=os.path.join(REPO_ROOT, "bucket_in_folder"))
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "bucket_out_folder"))
    parser.add_argument("--work-dir", default=os.path.join(REPO_ROOT, "work"))
    parser.add_argument("--force", action="store_true",
                         help="Ignore existing state and reprocess every file.")
    parser.add_argument("--stop-on-error", action="store_true",
                         help="Abort the whole batch on the first failure instead of continuing.")
    parser.add_argument("--max-attempts", type=int, default=3,
                         help="Retries before a file is marked error_permanent.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        log("ERROR", msg=f"input dir does not exist: {args.input_dir}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.work_dir):
        shutil.rmtree(args.work_dir)
    os.makedirs(args.work_dir)
    extract_dir = os.path.join(args.work_dir, "_extracted")
    os.makedirs(extract_dir, exist_ok=True)

    state = StateStore(args.output_dir, max_attempts=args.max_attempts)
    if args.force:
        state.reset()

    files = discover_input_files(args.input_dir, extract_dir)
    total = len(files)
    log("INFO", msg=f"discovered {total} point cloud(s) in {args.input_dir}")

    had_failure = False
    for index, (file_id, source_path) in enumerate(files, start=1):
        if state.is_done(file_id, args.output_dir):
            log("INFO", idx=f"{index}/{total}", file=file_id, step="skip", status="already_done")
            continue
        if state.is_permanently_failed(file_id):
            log("WARN", idx=f"{index}/{total}", file=file_id, step="skip",
                status="error_permanent", msg="exceeded max attempts, not retrying")
            had_failure = True
            continue

        try:
            process_one_file(file_id, source_path, args.output_dir, args.work_dir,
                              index, total, state)
            log("INFO", idx=f"{index}/{total}", file=file_id, step="done", status="done")
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill the batch
            state.mark_failed(file_id, exc)
            log("ERROR", idx=f"{index}/{total}", file=file_id, status="failed", msg=str(exc)[:500])
            had_failure = True
            if args.stop_on_error:
                break

    state.clear_current(total)

    counts = state.summary()
    log("INFO", msg="batch complete", done=counts.get("done", 0),
        failed=counts.get("failed", 0) + counts.get("error_permanent", 0))
    for file_id, record in state.failed_files().items():
        log("WARN", file=file_id, status=record.get("status"),
            step=record.get("step"), error=record.get("error"))

    return 1 if had_failure else 0


if __name__ == "__main__":
    sys.exit(main())
