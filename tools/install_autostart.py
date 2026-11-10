#!/usr/bin/env python3
"""Install/enable a reviewed system unit; does not start or stop devices."""
import argparse
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[1]


def render(root,home,user,robot):
    values={'ROOT':str(root),'HOME':str(home),'USER':user,'ROBOT':str(robot)}
    # Keep generated ExecStart and specifier parsing unambiguous.
    for value in values.values():
        if any(c.isspace() or c in '%\\"\'' for c in value):raise ValueError('Use paths/user without spaces, quotes or systemd specifiers')
    text=(ROOT/'deploy/systemd/vivis-p1.service.in').read_text()
    for key,value in values.items():text=text.replace('@'+key+'@',value)
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    p.add_argument('--print-only',action='store_true');args=p.parse_args()
    if os.getuid()==0:p.error('请以实际运行程序的普通用户运行，工具需要时会调用sudo')
    robot=args.robot.resolve()
    if not robot.is_file() or not (ROOT/'.venv/bin/python').is_file():p.error('Robot config or project venv missing')
    user=pwd.getpwuid(os.getuid());text=render(ROOT,Path(user.pw_dir),user.pw_name,robot)
    if args.print_only:print(text);return
    with tempfile.TemporaryDirectory(prefix='vivis-service-') as directory:
        path=Path(directory)/'vivis-p1.service';path.write_text(text)
        subprocess.run(['systemd-analyze','verify',str(path)],check=True)
        subprocess.run(['sudo','install','-m','644',str(path),'/etc/systemd/system/vivis-p1.service'],check=True)
    subprocess.run(['sudo','systemctl','daemon-reload'],check=True)
    subprocess.run(['sudo','systemctl','enable','vivis-p1.service'],check=True)
    print('vivis-p1.service已安装并启用，下次开机启动；本次未启动或停止任何设备。')


if __name__=='__main__':main()
