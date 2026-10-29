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

## Calibration background

Read [人工标定与表情驱动对齐原则](docs/developer/机电仿生头_人工标定与表情驱动说明.md) when discussing or changing calibration semantics. This records the user's confirmed background, with current implementation and future deployment kept distinct:

- Neutral is added once. An unlisted motor has the neutral endpoint and contributes zero delta for that shape; contributions from other shapes remain. Sparse JSON and dense AFS are position-equivalent under direct delta summation. Do not treat normalization-specific membership effects as missing basic calibration information. Direct delta summation is the reference principle; the accepted runtime currently uses `bounded_weighted_deltas` with additional eye/ownership rules. A documentation change does not switch that algorithm.
- The user confirmed that this head's `MouthClose` concerns only ID31, with the weight 0 and weight 1 targets both equal to 2047. Do not infer that 35 neutral records in AFS mean 35 active axes. An explicit neutral endpoint has zero delta; under current normalization it may still affect the denominator. The deployed mapping remains separate from this newly clarified intent.
- Confirmed recalibration can supersede old device limits, with matching hardware configuration, readback and acceptance. AFS `position` is the stored pose endpoint, not a general neutral field; resolve conflicting calibration fields before deployment. No device writes are implied by saving these principles.

## Mouth tuning preference (2026-10-28)

The user requires demonstrated software bug fixes and reasonable hardware speed/acceleration tuning, not complex mouth algorithms to conceal hardware limitations. Preserve source data, calibration and established stop policy; do not add silence-driven recentering, mouth-close priority, or timing compensation for this tuning task. The unshipped mouth-lead experiment was withdrawn. Existing accepted eye/neck Ruckig remains. Explicit task speed presets must use the same robot model for profile/startup/readback and leave canonical files unchanged.

Startup tongue/jaw order is physically validated: slowly open ID31 slightly (configured1900, cap60 counts/s), center ID29 and verify settlement, then return ID31 to neutral during full homing. An enabled tongue away from neutral also needs this order. Do not let simultaneous homing or a speed trial bypass mechanical clearance. Do not confuse this startup recovery with permission to force mouth closure during playback.

On2026-10-29 the user explicitly corrected JawOpen ID31's full-weight endpoint from1380 to1547 in 墨甲10.29.json. Neutral remains2047. Recompute calibration mapping from the corrected endpoint; do not retain the1380 curve and merely clip its end. Both works were regenerated with only ID31 positions changing. This correction does not authorize any additional hardware or gain changes.
