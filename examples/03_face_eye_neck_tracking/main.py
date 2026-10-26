#!/usr/bin/env python3
"""Developer example: eyes-neck. See README.md for inputs and axis ownership."""
from vivis.runtime.developer import run_example


def main():
    # The runner loads the robot and selected work, checks provenance, homes
    # all35 axes, and runs the shared audio/motion loop. This mode selects
    # exactly which behavior owns the eye and neck axes; it never opens a
    # second serial connection for tracking.
    return run_example('eyes-neck')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
