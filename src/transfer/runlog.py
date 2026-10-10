"""
Run logs: copy everything a tool prints to a file as well, so no result exists only in a terminal window.

    from src.transfer.runlog import tee_to
    tee_to("14_datasize_check")        # at the start of main(); writes outputs/transfer/logs/<name>_<YYYYmmdd_HHMMSS>.log

The file is written line by line and flushed, so a crashed or interrupted run still leaves its log. Copy the logs that matter into
docs/Research_Journal/packs/cross_model_transfer/logs/ (outputs/ is not in git).
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT, "outputs", "transfer", "logs")


class _Tee:
    def __init__(self, stream, handle):
        self._stream, self._handle = stream, handle

    def write(self, s):
        self._stream.write(s)
        try:
            self._handle.write(s)
            self._handle.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def tee_to(name, argv=None):
    """Start copying stdout to a new log file and return its path. The first line records the command line."""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    handle = open(path, "w", encoding="utf8")
    handle.write(f"# {name} started {time.strftime('%Y-%m-%d %H:%M:%S')}; arguments: {' '.join(sys.argv[1:] if argv is None else argv)}\n")
    handle.flush()
    sys.stdout = _Tee(sys.stdout, handle)
    return path
