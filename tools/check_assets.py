#!/usr/bin/env python3
"""Verify the media and model included in this clone; no external downloads."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vivis.runtime.assets import main

if __name__ == '__main__':
    main()
