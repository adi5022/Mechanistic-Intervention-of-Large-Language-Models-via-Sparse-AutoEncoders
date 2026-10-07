import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROME_DIR = REPO_ROOT / "third_party" / "rome"


def setup():
    if str(ROME_DIR) not in sys.path:
        sys.path.insert(0, str(ROME_DIR))
    os.chdir(ROME_DIR)
    return REPO_ROOT
