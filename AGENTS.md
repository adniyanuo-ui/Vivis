# Repository Guidelines

This branch is the standalone Mojia developer delivery. Keep only active source, configuration, examples, bundled runtime assets, developer documentation, minimal tools and relevant tests.

## Structure and sources

- `src/vivis/`: shared motion/runtime/vision/Feetech implementation; never copy algorithms into examples.
- `configs/robots/mojia/`: canonical hardware, final calibration, tracking, homing and telemetry.
- `assets/works/` and `assets/models/`: actual tracked media/model files, not external pointers.
- `src/vivis/hardware/feetech/_vendor/`: required pinned SDK subset with its license.
- `tests/`: regression tests using mocks and PTYs.

Do not reintroduce legacy project paths, compatibility modules, experimental upstream tools or dependence on another checkout. Preserve original expression/audio/calibration bytes and accepted hardware parameters unless explicitly changed by the user. Configuration is the unique source of device parameters.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python tools/check_assets.py
./examples/01_face_control/run.sh --check
./examples/02_face_eye_tracking/run.sh --check
./examples/03_face_eye_neck_tracking/run.sh --check
git diff --check
```

Installation and offline checks must not open real hardware. Preserve startup recovery, constraints and stop/holding-torque behavior. Do not commission baud rates, switch drivers or alter protected registers during cleanup. Validate standalone installation and checks from a fresh clone; do not use the old checkout as a hidden runtime dependency.

Use clear Chinese documentation and local links. Distinguish source targets, transmitted positions, encoder feedback and observed skin motion. Record offline versus onsite validation accurately. Keep third-party licenses/provenance with redistributed components. Review diffs before a local commit; do not push without user instruction.
