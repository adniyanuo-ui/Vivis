#!/usr/bin/env python3
"""Save one local API key; no network, device access or service startup."""
import getpass
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.speech.credentials import save_key


def main():
    if not sys.stdin.isatty():raise SystemExit('请在本机交互终端运行，勿通过命令行或聊天传递Key')
    try:path=save_key(getpass.getpass('百炼API Key（不回显，仅保存到本机）: '))
    except ValueError as exc:raise SystemExit(str(exc)) from None
    print('已保存：'+str(path)+'（权限600，不进入Git）。无需再次使用--ask-key。')
    print('开机服务已安装时，先退出手动P1，再执行：sudo systemctl start vivis-p1.service')


if __name__=='__main__':main()
