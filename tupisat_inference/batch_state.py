"""Persistent per-file state for the batch orchestrator.

The manifest lives inside the (bind-mounted) output folder so it survives a
container restart: that is the only thing that lets the orchestrator know,
on the next `docker run`, which point clouds are already done.
"""

import json
import os
import tempfile
import time

STATE_DIRNAME = ".sat_state"
MANIFEST_FILENAME = "manifest.json"
CURRENT_FILENAME = "current.json"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_ERROR_PERMANENT = "error_permanent"


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path, data):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


class StateStore:
    """Tracks per-file processing status across container restarts."""

    def __init__(self, output_dir, max_attempts=3):
        self.state_dir = os.path.join(output_dir, STATE_DIRNAME)
        self.manifest_path = os.path.join(self.state_dir, MANIFEST_FILENAME)
        self.current_path = os.path.join(self.state_dir, CURRENT_FILENAME)
        self.max_attempts = max_attempts
        self._records = self._load()

    def _load(self):
        if not os.path.exists(self.manifest_path):
            return {}
        try:
            with open(self.manifest_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # A corrupt manifest must never crash the batch; start clean and
            # let files be reprocessed rather than blocking the whole run.
            return {}

    def _save(self):
        _atomic_write_json(self.manifest_path, self._records)

    def reset(self):
        self._records = {}
        self._save()

    def is_done(self, file_id, output_dir):
        record = self._records.get(file_id)
        if not record or record.get("status") != STATUS_DONE:
            return False
        output_file = record.get("output_file")
        if not output_file:
            return False
        return os.path.exists(os.path.join(output_dir, output_file))

    def is_permanently_failed(self, file_id):
        record = self._records.get(file_id)
        return bool(record) and record.get("status") == STATUS_ERROR_PERMANENT

    def attempts(self, file_id):
        record = self._records.get(file_id)
        return record.get("attempts", 0) if record else 0

    def mark_running(self, file_id, step, index, total):
        record = self._records.setdefault(file_id, {"attempts": 0})
        record["status"] = STATUS_RUNNING
        record["step"] = step
        record["attempts"] = record.get("attempts", 0) + 1
        record["error"] = None
        record.setdefault("started_at", _now())
        record["updated_at"] = _now()
        self._save()
        self.set_current(file_id, step, index, total, started_at=record["started_at"])

    def mark_step(self, file_id, step, index, total):
        record = self._records.setdefault(file_id, {"attempts": 1})
        record["step"] = step
        record["updated_at"] = _now()
        self._save()
        self.set_current(file_id, step, index, total,
                          started_at=record.get("started_at", _now()))

    def mark_done(self, file_id, output_file):
        record = self._records.setdefault(file_id, {"attempts": 1})
        record["status"] = STATUS_DONE
        record["step"] = None
        record["error"] = None
        record["output_file"] = output_file
        record["finished_at"] = _now()
        record["updated_at"] = _now()
        self._save()

    def mark_failed(self, file_id, error):
        record = self._records.setdefault(file_id, {"attempts": 1})
        attempts = record.get("attempts", 1)
        record["error"] = str(error)[-2000:]
        record["updated_at"] = _now()
        if attempts >= self.max_attempts:
            record["status"] = STATUS_ERROR_PERMANENT
        else:
            record["status"] = STATUS_FAILED
        self._save()

    def summary(self):
        counts = {STATUS_PENDING: 0, STATUS_DONE: 0, STATUS_FAILED: 0,
                  STATUS_ERROR_PERMANENT: 0, STATUS_RUNNING: 0}
        for record in self._records.values():
            counts[record.get("status", STATUS_PENDING)] = \
                counts.get(record.get("status", STATUS_PENDING), 0) + 1
        return counts

    def failed_files(self):
        return {
            file_id: record
            for file_id, record in self._records.items()
            if record.get("status") in (STATUS_FAILED, STATUS_ERROR_PERMANENT)
        }

    def set_current(self, file_id, step, index, total, started_at=None):
        counts = self.summary()
        payload = {
            "overall": {
                "total": total,
                "done": counts.get(STATUS_DONE, 0),
                "failed": counts.get(STATUS_FAILED, 0) + counts.get(STATUS_ERROR_PERMANENT, 0),
                "pending": max(total - sum(counts.values()), 0),
                "running": 1,
            },
            "current": {
                "file": file_id,
                "index": index,
                "total": total,
                "step": step,
                "started_at": started_at,
                "updated_at": _now(),
            },
        }
        _atomic_write_json(self.current_path, payload)

    def clear_current(self, total):
        counts = self.summary()
        payload = {
            "overall": {
                "total": total,
                "done": counts.get(STATUS_DONE, 0),
                "failed": counts.get(STATUS_FAILED, 0) + counts.get(STATUS_ERROR_PERMANENT, 0),
                "pending": 0,
                "running": 0,
            },
            "current": None,
            "finished_at": _now(),
        }
        _atomic_write_json(self.current_path, payload)
