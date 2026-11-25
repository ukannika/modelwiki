import os
from pathlib import Path

ROOT = Path(os.environ.get("MODELWIKI_ROOT") or Path(__file__).resolve().parents[2])
