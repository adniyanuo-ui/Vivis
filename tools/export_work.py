#!/usr/bin/env python3
"""Convert a customer's expression/audio pair into a new work; no hardware I/O."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vivis.motion.export import main

if __name__ == '__main__':
    main()
