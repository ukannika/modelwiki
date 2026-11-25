import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelwiki import ROOT  # noqa: E402


@pytest.fixture
def root(tmp_path):
    """A copy of the project, so plans and results land in a temporary directory."""
    dst = tmp_path / "mw"
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("plans", "results", "tests", "__pycache__", "page"))
    return dst


@pytest.fixture
def tree(tmp_path):
    t = tmp_path / "tree"
    (t / "model").mkdir(parents=True)
    (t / "model/seam.py").write_text("\n".join(f"line {i}" for i in range(40)) + "\n")
    (t / "model/moe.py").write_text("\n".join(f"line {i}" for i in range(10)) + "\n")
    for cmd in (["init", "-q"], ["add", "."],
                ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "tree"]):
        subprocess.run(["git", "-C", str(t), *cmd], check=True)
    return t
