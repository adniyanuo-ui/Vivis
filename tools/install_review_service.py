#!/usr/bin/env python3
"""Install the independent loopback sample review service; no hardware access."""
import argparse
import os
from pathlib import Path
import pwd
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def render(root,user):
    for value in (str(root),user):
        if any(c.isspace() or c in '%\\"\'' for c in value):raise ValueError('安装路径或用户名不支持空白、引号或systemd转义符')
    return (ROOT/'deploy/systemd/vivis-review.service.in').read_text().replace('@ROOT@',str(root)).replace('@USER@',user)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--print-only',action='store_true');a=p.parse_args()
    if os.getuid()==0:p.error('请以实际运行程序的普通用户运行')
    text=render(ROOT,pwd.getpwuid(os.getuid()).pw_name)
    if a.print_only:print(text);return
    with tempfile.TemporaryDirectory(prefix='vivis-review-') as folder:
        path=Path(folder)/'vivis-review.service';path.write_text(text)
        subprocess.run(['systemd-analyze','verify',str(path)],check=True)
        subprocess.run(['sudo','install','-m','644',str(path),'/etc/systemd/system/vivis-review.service'],check=True)
    subprocess.run(['sudo','systemctl','daemon-reload'],check=True)
    subprocess.run(['sudo','systemctl','enable','--now','vivis-review.service'],check=True)
    print('审核服务已启用：http://127.0.0.1:8766/；不启动或停止机器头交互服务。')


if __name__=='__main__':main()
