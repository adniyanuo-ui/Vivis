#!/usr/bin/env python3
"""Legacy CLI/import compatibility for vivis.runtime.show."""
from pathlib import Path
import importlib
import sys
_source = str(Path(__file__).resolve().parents[1] / "src")
if _source not in sys.path:
    sys.path.insert(0, _source)
_module = importlib.import_module("vivis.runtime.show")
if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
