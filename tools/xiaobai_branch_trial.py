#!/usr/bin/env python3
"""Prepare or run a bounded branch-only trial. Default is offline, --play moves hardware.

For physical branch isolation after on-site disconnection: only the selected
branch is commanded AND monitored. RS485 selection includes jaw 31, never neck.
"""
import argparse
import hashlib
import json
from pathlib import Path

from xiaobai_runtime.mapping import ROOT, read_json, inputs, compile_trajectory, live_preflight


def validate_branch_probe(rows,branch):
    expected_by_scope={'ttl':set(range(2,31)), 'rs485-jaw':set(range(31,35)),
                       'both':set(range(2,35))}
    if branch not in expected_by_scope:raise ValueError('Unknown probe scope')
    expected=expected_by_scope[branch]
    if len(rows)!=33 or {r['id'] for r in rows}!=set(range(2,35)):
        raise ValueError('Branch probe must cover every known ID 2..34')
    online={r['id'] for r in rows if 'model_raw' in r}
    uncertain=[r['id'] for r in rows if 'model_raw' not in r and r.get('result')!='no_response_twice']
    wrong_model=[r['id'] for r in rows if 'model_raw' in r and r['model_raw']!=(521 if r['id']<=30 else 11272)]
    if online!=expected or uncertain or wrong_model:
        raise ValueError(f'Branch isolation not established: missing={sorted(expected-online)}, '
                         f'other_branch_online={sorted(online-expected)}, '
                         f'uncertain={uncertain}, model_mismatch={wrong_model}')


def probe_branch(profile,output,branch):
    from xiaobai_runtime.feetech_sdk import OfficialSDKBus
    from xiaobai_runtime.protocol import ReadTimeout
    import time
    output.mkdir()
    bus=OfficialSDKBus(profile['port'],profile['baud'],trace_path=output/'serial_trace.json')
    rows=[]
    try:
        for sid in range(2,35):
            row={'id':sid}
            for attempt in range(2):
                try:
                    row.update(model_raw=int.from_bytes(bus._read_once(sid,3,2),'little'),result='responded')
                    break
                except ReadTimeout:
                    row['result']='no_response_twice' if attempt else 'no_response_once'
                    if not attempt:time.sleep(.01)
                except Exception as exc:
                    row.update(result='inconclusive',error=str(exc));break
            rows.append(row)
    finally:
        bus.close()
        (output/'scan.json').write_text(json.dumps({'read_only':True,'servos':rows},indent=2)+'\n')
    validate_branch_probe(rows,branch)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch',choices=['ttl','rs485-jaw'],required=True)
    parser.add_argument('--play',action='store_true')
    parser.add_argument('--profile',type=Path,default=ROOT/'xiaobai/playback_profile_raw_trace.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Use a new output directory to preserve trial evidence')
    profile=read_json(args.profile)
    calibration=ROOT/'xiaobai/xiaobai标定文件.json'
    if hashlib.sha256(calibration.read_bytes()).hexdigest()!=profile['calibration_sha256']:
        raise SystemExit('Calibration hash differs from the approved profile')
    values,fps,order,shapes=inputs(
        ROOT/'outputs/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json',
        calibration,ROOT/'xiaobai/blendshape_channels_61.csv')
    ids,trajectory,_=compile_trajectory(values,order,shapes,profile,'face_eyes')
    selected=[sid for sid in ids if (2<=sid<=30 if args.branch=='ttl' else sid==31)]
    columns=[ids.index(sid) for sid in selected]
    live_preflight(profile,selected)
    plan=dict(branch=args.branch,command_and_monitor_ids=selected,max_seconds=12,
              motion_gain=profile['motion_gain'],hardware_requested=args.play,
              electrical_isolation='Requires onsite disconnection; full model scan checks only reachability, not electrical wiring',
              unselected_axes='No new goals and no feedback polling',neck_motion=False)
    args.output.mkdir(parents=True)
    (args.output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if args.play:
        probe_branch(profile,args.output/'branch_probe',args.branch)
        from xiaobai_runtime.playback import play
        play(profile,selected,trajectory[:,columns],fps,ROOT/'0001.wav',
             args.output/'playback_log.json',ROOT/'outputs/xiaobai/STOP',max_seconds=12)


if __name__=='__main__':main()
