"""Legacy import names for the shared Vivis developer package."""
from pathlib import Path
import sys

_source = str(Path(__file__).resolve().parents[2] / "src")
if _source not in sys.path:
    sys.path.insert(0, _source)
