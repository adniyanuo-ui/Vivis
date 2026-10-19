#!/usr/bin/env python3
"""Restore pinned external sources without installing software or touching devices."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*args):
    return subprocess.check_output(['git', *map(str, args)], text=True).strip()


def restore(entry):
    destination = ROOT / entry['path']
    if destination.exists():
        if not (destination / '.git').exists():
            raise RuntimeError(f'Refusing to overwrite existing directory: {destination}')
        if git('-C', destination, 'rev-parse', 'HEAD') != entry['commit']:
            raise RuntimeError(f'Existing checkout differs from pinned commit: {destination}')
        if git('-C', destination, 'status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError(f'Existing checkout has tracked edits: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', entry['url'], str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', entry['commit']], check=True)
    print(f"Verified {entry['path']} @ {entry['commit']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=['servo', 'a2f', 'driver', 'servo-linux', 'all'], default='servo')
    args = parser.parse_args()
    entries = json.loads((ROOT / 'tools/external-sources.json').read_text())
    for entry in entries:
        if args.component in (entry['component'], 'all'):
            restore(entry)


if __name__ == '__main__':
    main()
