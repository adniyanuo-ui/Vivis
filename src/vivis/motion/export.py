"""Compile a customer's 61-channel clip using the saved robot calibration.

Offline only. Source files are never edited. A30FPS clip is held twice at60FPS;
there is no silence inference, invented blinking or expression augmentation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import wave
import numpy as np

from vivis.motion.mapping import inputs, compile_trajectory, read_json
from vivis.paths import ROOT
from vivis.runtime.robot import Robot


def export_work(robot, name, clip, audio):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Work name must be1..64 ASCII letters/digits/underscore/hyphen')
    if name in robot.works['works']:
        raise ValueError('Work already exists; use a new name')
    clip, audio = Path(clip), Path(audio)
    values, fps, order, shapes = inputs(clip, robot.calibration, robot.resolve('channels'),
                                       allowed_motor_ids=set(range(2, 37)))
    if fps not in (30, 60):
        raise ValueError('Developer examples currently accept30/60FPS sources')
    with wave.open(str(audio)) as wav:
        duration = wav.getnframes() / wav.getframerate()
        if wav.getsampwidth() != 2 or abs(len(values)/fps-duration) > 1/fps:
            raise ValueError('Need PCM16 WAV matching the clip duration within one source frame')
    if fps == 30:
        values = np.repeat(values, 2, axis=0)
    ids, positions, report = compile_trajectory(values, order, shapes, robot.motion_profile(), 'face_eyes')
    output = ROOT / 'outputs/imported_works' / name
    output.mkdir(parents=True, exist_ok=False)
    target, destination = output / 'targets.json', output / 'audio.wav'
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    payload = dict(schema_version='mojia-motor-targets-0.1', device_limits_applied=True,
                   limits_source=str(robot.resolve('hardware')),
                   calibration_sha256=robot.calibration_sha256, audio_sha256=digest(audio),
                   source_clip_sha256=digest(clip), source_fps=fps,
                   fps={'num': 60, 'den': 1}, frames=len(values), motor_ids=ids,
                   positions=positions.tolist(), trajectory_report=report)
    target.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    shutil.copyfile(audio, destination)
    shutil.copyfile(clip, output / 'source.json')
    entry = dict(name=name, targets=str(target.relative_to(ROOT)), audio=str(destination.relative_to(ROOT)),
                 targets_sha256=digest(target), audio_sha256=digest(destination),
                 calibration_sha256=robot.calibration_sha256, fps=60, frames=len(values), motor_ids=ids)
    manifest = read_json(robot.resolve('works'))
    if name in manifest['works']:
        raise ValueError('Work manifest changed concurrently; refusing to overwrite')
    manifest['works'][name] = entry
    temporary = robot.resolve('works').with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(robot.resolve('works'))
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', type=Path)
    parser.add_argument('--name', required=True)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_work(Robot(args.robot), args.name, args.clip, args.audio), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
