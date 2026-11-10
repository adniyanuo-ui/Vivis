#!/usr/bin/env python3
"""Offline52-channel calibration availability; never opens devices."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vivis.runtime.robot import Robot
from vivis.interaction.standby import plan


def audit(path):
    robot=Robot(path)
    settings=json.loads((ROOT/'configs/interaction/interaction.json').read_text())
    _,_,_,mapper,_,_=plan(robot,settings,1.)
    mapper.prepare_expression_mapping()
    channels={name:sorted(mapper.expression_axes[name]) for name in mapper.order[:52]}
    return dict(robot=robot.definition['name'],channel_count=len(channels),effective_channel_count=sum(bool(v) for v in channels.values()),
        no_effect_channels=[k for k,v in channels.items() if not v],empty_calibration_channels=[k for k,v in mapper.shapes.items() if not v],
        channel_actuator_ids=channels,excluded_actuator_ids=robot.definition.get('excluded_actuator_ids',[]),hardware_opened=False,
        note='Effective nominal displacement under current mapping, active inventory and limits; not measured skin motion')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    parser.add_argument('--json',action='store_true');args=parser.parse_args()
    result=audit(args.robot)
    if args.json:print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        print(f'{result["robot"]}:52通道输入；{result["effective_channel_count"]}个有有效目标位移。')
        print('当前映射无净位移/不可用：'+('、'.join(result['no_effect_channels']) or '无'))
        print('离线检查，不驱动设备，不代表实测皮肤效果。')


if __name__=='__main__':main()
