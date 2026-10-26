"""Verify/import the separately distributed example media, without hardware I/O."""
import argparse
import hashlib
from pathlib import Path
import shutil

from vivis.runtime.robot import Robot
from vivis.paths import ROOT


def verify_assets(robot, source=None):
    report = []
    for name in robot.works['works']:
        work = robot.work(name)
        for key in ('audio', 'targets'):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', type=Path)
    parser.add_argument('--from-dir', type=Path, help='Import missing files from a separately supplied asset bundle')
    args = parser.parse_args()
    for row in verify_assets(Robot(args.robot), args.from_dir):
        print('Verified ' + row['path'])


if __name__ == '__main__':
    main()
