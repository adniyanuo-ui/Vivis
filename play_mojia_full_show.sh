#!/usr/bin/env bash
# Compatibility entry: same work/repeat defaults, canonical developer config.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${VIVIS_PYTHON:-$PWD/.venv/bin/python}" -m vivis.runtime.developer --work 0001 --repeat 3 "$@"
