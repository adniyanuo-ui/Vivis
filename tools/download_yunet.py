#!/usr/bin/env python3
"""Download the pinned OpenCV YuNet model and verify it before replacement."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    manifest = json.loads((ROOT / 'tools/model-manifests/yunet.json').read_text())
    target = ROOT / manifest['destination']

    def verified(path):
        return (path.is_file() and path.stat().st_size == manifest['size']
                and hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256'])

    if not verified(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.yunet-', dir=target.parent)
        os.close(fd)
        try:
            subprocess.run(['curl', '-fLsS', '--retry', '3', '--max-time', '120',
                            manifest['url'], '-o', temporary], check=True)
            if not verified(Path(temporary)):
                raise RuntimeError('YuNet size/SHA256 verification failed')
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    print(f'Verified {target} SHA256={manifest["sha256"]}')


if __name__ == '__main__':
    main()
