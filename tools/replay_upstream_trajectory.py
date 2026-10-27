#!/usr/bin/env python3
"""Offline ID32 frozen-input comparison; never opens serial/audio/camera."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from vivis.motion.upstream import UpstreamTrajectory,settings
from vivis.runtime.robot import Robot
from vivis.vision.pursuit import neck_command_limits,NeckCommandScheduler


def response(plan,low,high,speed):
    p=UpstreamTrajectory(low,high,speed,plan);p.reset(2047.)
    baseline=NeckCommandScheduler(2047.,0.,speed,False,filter_tau_s=.15)
    first={};old_first={};previous=2047.
    for tick in range(1,121):
        now=tick/60;p.step(100.,1/60)
        baseline.step(2047+100*now,now)
        velocity=(baseline.filtered_position-previous)*60;previous=baseline.filtered_position
        for percent in (10,50,90):
            if p.velocity>=percent:first.setdefault(str(percent),now)
            if velocity>=percent:old_first.setdefault(str(percent),now)
    return dict(planner_velocity_percent_s=first,baseline_lowpass_velocity_percent_s=old_first)


def replay(path,output):
    data=json.loads(path.read_text());events=data['events']
    if len(events)<4:raise ValueError('Need at least4 playback samples for finite differences')
    robot=Robot();config=robot.tracking();profile=robot.motion_profile()
    low,high=neck_command_limits(config,profile);speed=config['neck']['speed_counts_s']
    initial=next(s['position_raw'] for s in data['preflight_snapshots'] if s['id']==32)
    # Step from the measured reference at the beginning of this frozen clip.
    initial=float(np.clip(initial,low,high))
    output.mkdir(parents=True,exist_ok=True)
    report=dict(source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_completed=data.get('completed'),samples=len(events),hardware_accessed=False,
        note='Frozen original camera/controller requests. Not a prediction of the changed physical camera loop.',
        candidates=[])
    source_positions=np.array([e['person_gaze']['smoothed_neck_goal'] for e in events])
    stamps=np.array([e['sent_monotonic'] for e in events])
    periods=np.r_[1/data['control_hz'],np.diff(stamps)]
    if not np.isfinite(periods).all() or np.any(periods<=0):raise ValueError('Invalid source timestamps')
    source_velocity=np.diff(source_positions)/periods[1:]
    source_acceleration=np.diff(source_velocity)/periods[2:]
    source_jerk=np.diff(source_acceleration)/periods[3:]
    report['source_float_reference_finite_differences']=dict(
        velocity_abs_max=float(np.max(np.abs(source_velocity))),
        acceleration_abs_max=float(np.max(np.abs(source_acceleration))),
        jerk_abs_max=float(np.max(np.abs(source_jerk))))
    for acceleration,jerk in [(400.,4000.),(800.,8000.),(800.,16000.),(1600.,16000.)]:
        plan=settings({'upstream_trajectory':dict(enabled=True,
            max_acceleration_counts_s2=acceleration,max_jerk_counts_s3=jerk)})
        p=UpstreamTrajectory(low,high,speed,plan,config['neck']['return_duration_s']);p.reset(initial)
        rows=[];elapsed=[]
        for e,dt in zip(events,periods):
            d=e['person_gaze'];started=time.perf_counter()
            pos=p.step(d['neck_requested_velocity'],float(dt),d['returning_to_center'])
            elapsed.append((time.perf_counter()-started)*1e6)
            rows.append(dict(audio_seconds=e['audio_seconds'],dt=float(dt),
                requested_velocity=d['neck_requested_velocity'],returning=d['returning_to_center'],
                source_reference=d['planned_neck_goal'],baseline_smoothed=d['smoothed_neck_goal'],
                planned_position=pos,integer_goal=round(pos),velocity=p.velocity,acceleration=p.acceleration,
                jerk_average=p.last_diagnostic['jerk_average'],mode=p.last_diagnostic['mode']))
        matrix=np.array([[r[k] for k in ('planned_position','velocity','acceleration','jerk_average')] for r in rows])
        delta=np.abs(matrix[:,0]-source_positions)
        result=dict(max_acceleration_counts_s2=acceleration,max_jerk_counts_s3=jerk,
            min_position=float(matrix[:,0].min()),max_position=float(matrix[:,0].max()),
            max_abs_velocity=float(np.abs(matrix[:,1]).max()),max_abs_acceleration=float(np.abs(matrix[:,2]).max()),
            max_abs_average_jerk=float(np.abs(matrix[:,3]).max()),final_position=float(matrix[-1,0]),
            difference_from_recorded_smoothed_position_mean=float(delta.mean()),
            difference_from_recorded_smoothed_position_p95=float(np.percentile(delta,95)),
            calculation_us_p50=float(np.percentile(elapsed,50)),calculation_us_p99=float(np.percentile(elapsed,99)),
            calculation_us_max=float(max(elapsed)),response=response(plan,low,high,speed))
        report['candidates'].append(result)
        if (acceleration,jerk)==(800.,8000.):
            with (output/'selected_reference.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(json.dumps(result,ensure_ascii=False))
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    replay(args.log,args.output)
