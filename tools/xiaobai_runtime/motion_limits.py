"""Compatibility import; implementation lives in src/vivis."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("vivis.motion.limits")
