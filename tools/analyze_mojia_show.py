#!/usr/bin/env python3
"""Summarize recorded performance timing and eye/neck feedback; no device access."""
import argparse
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def percentiles(values):
    if not values:return None
    values=sorted(values)
    return {name:round(values[max(0,math.ceil(len(values)*q)-1)],3)
            for name,q in [('p50',.5),('p95',.95),('p99',.99),('max',1.)]}


def summarize(path):
    data=json.loads(path.read_text());events=data.get('events',[])
    result=dict(path=str(path),completed=data['completed'],error=data.get('error'),phase=data.get('phase'),
                startup_communication_recoveries=data.get('startup_communication_recoveries',[]),
                audio_seconds=data.get('audio_seconds_at_stop'),sent_frames=len(events),
                skipped_source_frames=data.get('skipped_source_frames'),face_update_hz=data.get('face_update_hz',data.get('playback_fps')))
    result['command_interval_ms']=percentiles([(b['sent_monotonic']-a['sent_monotonic'])*1000
                                              for a,b in zip(events,events[1:])])
    result['stage_ms']={stage:percentiles([e['stage_ms'][stage] for e in events if 'stage_ms' in e])
                        for stage in ('gaze','command','feedback')}
    result['critical_axes']={}
    for sid in (23,32):
        samples=[]
        for event in events:
            feedback=event.get('feedback_samples',[event['feedback']] if 'feedback' in event else [])
            for sample in feedback:
                if sample['id']==sid:
                    samples.append((sample.get('sample_started_monotonic',event['sent_monotonic']),
                                    sample['position']-event['goals'][str(sid)]))
        result['critical_axes'][sid]=dict(samples=len(samples),
            feedback_interval_ms=percentiles([(b[0]-a[0])*1000 for a,b in zip(samples,samples[1:])]),
            sampled_goal_error_counts=percentiles([abs(s[1]) for s in samples]))
    serial=data.get('serial_diagnostics',{})
    result['read_failures']=len(serial.get('read_failures',[]))
    trace=path.with_name('serial_trace.json')
    result['uart_delta']=json.loads(trace.read_text()).get('uart_delta') if trace.exists() else None
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path)
    args=parser.parse_args()
    directory=args.directory
    if directory is None:directory=ROOT/Path((ROOT/'outputs/mojia/latest_full_show.txt').read_text().strip())
    if (directory/'performance').is_dir():directory=directory/'performance'
    paths=sorted(directory.glob('round_*/playback_log.json'))
    summary=directory/'summary.json'
    if not paths and not summary.exists():parser.error(f'No playback or startup summary in {directory}')
    report=dict(rounds=[summarize(p) for p in paths],
        note='反馈误差按分时采样统计，不能据此证明快速抖动已消除；现场观感仍需验收。')
    if summary.exists():
        startup=json.loads(summary.read_text())
        report.update(completed=startup['completed'],error=startup.get('error'))
        if 'homing' in startup:
            report['homing']={k:v for k,v in startup['homing'].items() if k not in ('samples','original_speeds')}
    (directory/'timing_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
