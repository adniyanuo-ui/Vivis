#!/usr/bin/env python3
"""Wait for USB enumeration only; never open camera, audio or actuator devices."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vivis.runtime.robot import Robot


def pending(robot,cards):
    missing=[];port=robot.motion_profile()['port']
    if not os.path.exists(port):missing.append('串口未出现')
    elif not os.access(port,os.R_OK|os.W_OK):missing.append('串口无读写权限')
    speech=json.loads((ROOT/'configs/perception/speech.json').read_text())
    devices=[('麦克风',speech['capture']['device']),('音箱',robot.motion_profile()['audio_device'])]
    for label,name in devices:
        if isinstance(name,str) and name.rstrip(':').casefold() not in cards.casefold():missing.append(label+'未枚举')
    camera=robot.tracking()['camera']['device']
    if isinstance(camera,str) and camera.startswith('/dev/') and not Path(camera).exists():missing.append('相机未枚举')
    return missing


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--robot',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=120);args=p.parse_args()
    if not 0<args.timeout<=300:p.error('timeout must be within(0,300]')
    robot=Robot(args.robot);deadline=time.monotonic()+args.timeout;previous=None
    while True:
        try:cards=Path('/proc/asound/cards').read_text()
        except OSError:cards=''
        missing=pending(robot,cards)
        if not missing:print('USB枚举及串口权限已就绪；尚未驱动设备。',flush=True);return 0
        if missing!=previous:print('开机准备：'+ '；'.join(missing),flush=True);previous=missing
        if time.monotonic()>=deadline:print('USB准备超时，请检查供电、接线及权限。',flush=True);return 1
        time.sleep(1)


if __name__=='__main__':raise SystemExit(main())
