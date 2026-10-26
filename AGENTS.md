# Repository Guidelines

## Structure

Vivis is a Mojia electromechanical head developer platform. Current entry points and installation instructions are in `README.md`; historical experiments are documented in `docs/history/` and `docs/CODE_REVIEW.md`.

- `src/vivis/motion/`: channel semantics, calibration mapping, optional expression composition and actuator ownership.
- `src/vivis/runtime/`: robot configuration, startup/homing, audio synchronization, feedback scheduling and sessions.
- `src/vivis/hardware/feetech/`: protocol and verified model-specific transport.
- `src/vivis/vision/`: camera-only detection, preview and eye/neck coordination.
- `configs/robots/mojia/`: canonical hardware, final calibration, tracking, homing and telemetry configuration.
- `examples/`: three independently runnable examples and their Chinese tutorials.
- `tools/`: diagnostics, upstream experiments, tests and compatibility entry points.

Legacy imports alias the same implementation; do not copy motion algorithms back into compatibility wrappers. Work manifests contain media paths/hashes, not hardware parameter copies. Media and external SDK/model assets are supplied separately.

## Validation

```bash
.venv/bin/python -m unittest discover -s tools -p 'test_*.py'
./examples/01_face_control/run.sh --check
./examples/02_face_eye_tracking/run.sh --check
./examples/03_face_eye_neck_tracking/run.sh --check
git diff --check
```

Tests use mocks/PTYs; offline checks do not open hardware. Treat actual motion as a separate onsite acceptance step and do not describe it as validated without evidence. Preserve the established limits, startup recovery and stop/holding-torque behavior when refactoring. Do not silently commission baud rates or alter protected device registers as part of code cleanup.

## Style and review

Preserve Chinese technical prose, numbered sections, clear tables and repository-local Markdown links. Keep original upstream expression/media files unchanged, apart from explicitly requested transformations to derived artifacts. Keep engineering targets distinct from measurements and distinguish nominal targets, transmitted commands, encoder feedback and observed skin motion.

Use one active configuration source per kind of parameter, name units explicitly, and keep device-specific tuning out of example programs. Review diffs and whitespace before saving a local commit; do not push without user instruction. Use concise commit messages and identify offline versus onsite validation in change descriptions.
