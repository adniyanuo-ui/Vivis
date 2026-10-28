"""Verify the bundled example media and models, without hardware I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from vivis.runtime.robot import Robot
from vivis.paths import ROOT


def verify_assets(robot, source=None):
    report = []
    for name in robot.works['works']:
        work = robot.work(name)
        for key in ('audio','targets',*(('source_clip',) if 'source_clip' in work else ())):
            target = work[key]
            expected = work[key + '_sha256']
            if source is not None and not target.is_file():
                candidates = [Path(source) / target.relative_to(ROOT),
                              Path(source) / target.name]
                origin = next((p for p in candidates if p.is_file()), None)
                if origin is None:
                    raise FileNotFoundError('Missing asset in supplied directory: ' + target.name)
                if hashlib.sha256(origin.read_bytes()).hexdigest() != expected:
                    raise ValueError('Asset SHA256 mismatch: ' + str(origin))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, target)
            if not target.is_file():
                raise FileNotFoundError('缺少示例素材：' + str(target) + '；参见 assets/README.md')
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError('Asset SHA256 mismatch: ' + str(target))
            report.append(dict(work=name, kind=key, path=str(target), verified=True))
    return report


def verify_model(root=ROOT):
    """Check the bundled CPU face detector without opening a camera."""
    manifest = json.loads((root / 'assets/models/yunet/manifest.json').read_text())
    target = root / manifest['destination']
    if (not target.is_file() or target.stat().st_size != manifest['size']
            or hashlib.sha256(target.read_bytes()).hexdigest() != manifest['sha256']):
        raise ValueError('Bundled YuNet model missing or changed: ' + str(target))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', type=Path)
    parser.add_argument('--from-dir', type=Path, help='Import missing files from a separately supplied asset bundle')
    args = parser.parse_args()
    for row in verify_assets(Robot(args.robot), args.from_dir):
        print('Verified ' + row['path'])
    print('Verified ' + str(verify_model()))


if __name__ == '__main__':
    main()
