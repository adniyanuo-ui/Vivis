#!/usr/bin/env python3
"""Xiaobai: inspect, read-only scan, trajectory audit, dry scheduler and physical playback."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from xiaobai_runtime.mapping import ROOT,read_json,inputs,audit,draft_profile,compile_trajectory,live_preflight,limit_step
from xiaobai_runtime.protocol import Bus,goal_packet

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('command',choices=['doctor','audit','scan','compile','simulate','play'])
parser.add_argument('--clip',type=Path,default=ROOT/'outputs/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json')
parser.add_argument('--calibration',type=Path,default=ROOT/'xiaobai/xiaobai标定文件.json')
parser.add_argument('--header',type=Path,default=ROOT/'xiaobai/blendshape_channels_61.csv')
parser.add_argument('--profile',type=Path,default=ROOT/'xiaobai/playback_profile.json')
parser.add_argument('--output',type=Path,default=ROOT/'outputs/xiaobai')
parser.add_argument('--port');parser.add_argument('--scope',choices=['face','face_eyes','full'],default='full')
parser.add_argument('--draft',action='store_true',help='Offline only; endpoint envelopes are not safe limits')
parser.add_argument('--seconds',type=float,default=5)
parser.add_argument('--max-seconds',type=float,help='Stop physical playback after this many audio seconds; omit for full song')
parser.add_argument('--audio',type=Path,default=ROOT/'0001.wav')
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)

def save(name,value):
    (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(value,ensure_ascii=False,indent=2))

if args.command=='doctor':
    from serial.tools import list_ports
    ports=[{'port':p.device,'description':p.description,'serial':p.serial_number,'readable':os.access(p.device,os.R_OK),'writable':os.access(p.device,os.W_OK)} for p in list_ports.comports()]
    # pyserial's Linux glob does not include the WCH vendor driver's tty names.
    known={p['port'] for p in ports}
    for node in sorted(Path('/dev').glob('ttyCH343USB*')):
        if str(node) not in known:
            ports.append({'port':str(node),'description':'WCH vendor USB serial driver',
                          'sysfs':str((Path('/sys/class/tty')/node.name).resolve()),
                          'readable':os.access(node,os.R_OK),'writable':os.access(node,os.W_OK)})
    try:
        import sounddevice as sd
        audio=[{'index':i,**dict(d)} for i,d in enumerate(sd.query_devices()) if d['max_output_channels']>0]
    except Exception as e:audio={'error':str(e)}
    save('environment.json',{'ports':ports,'audio_devices':audio})
elif args.command=='audit':
    report=audit(args.clip,args.calibration,args.header)
    if not args.profile.exists():args.profile.write_text(json.dumps(draft_profile(report),ensure_ascii=False,indent=2)+'\n')
    save('calibration_audit.json',report)
elif args.command=='scan':
    profile=read_json(args.profile);port=args.port or profile['port'];results=[]
    try:
        if profile.get('serial_backend')=='feetech_python':
            from xiaobai_runtime.feetech_sdk import OfficialSDKBus
            bus=OfficialSDKBus(port,profile['baud'],profile.get('feedback_mode','block'))
        else:
            bus=Bus(port,profile['baud'])
    except Exception as e:
        save('scan.json',{'port':port,'read_only':True,'error':str(e),'servos':[]});raise SystemExit(2)
    try:
        for sid in range(2,35):
            try:results.append(bus.snapshot(sid))
            except Exception as e:results.append({'id':sid,'error':str(e)})
    finally:bus.close()
    save('scan.json',{'port':port,'baud':profile['baud'],'read_only':True,'servos':results})
else:
    profile=read_json(args.profile)
    if args.port:profile['port']=args.port
    if profile.get('calibration_sha256') and hashlib.sha256(args.calibration.read_bytes()).hexdigest()!=profile['calibration_sha256']:
        raise SystemExit('Calibration differs from the profile; regenerate and review the profile before compiling or playing')
    if args.command=='play' and args.draft:raise SystemExit('--draft is never accepted for physical playback')
    values,fps,order,shapes=inputs(args.clip,args.calibration,args.header)
    ids,trajectory,report=compile_trajectory(values,order,shapes,profile,args.scope,args.draft)
    report.update(frames=len(values),fps=fps,servo_ids=ids,
      max_unlimited_speed_counts_s={str(sid):float(np.max(abs(np.diff(trajectory[:,j])))*fps) for j,sid in enumerate(ids)})
    if args.command=='compile':
        np.savez_compressed(args.output/('trajectory_draft.npz' if args.draft else 'trajectory.npz'),positions=trajectory,ids=np.array(ids),fps=fps)
        save('trajectory_draft_report.json' if args.draft else 'trajectory_report.json',report)
    elif args.command=='simulate':
        # No serial port and no audio output are opened. Exercise actual packet encoding/rate limiting.
        speeds=np.full(len(ids),300.);last=trajectory[0].copy();start=time.monotonic();previous=start
        events=[];last_index=-1
        while time.monotonic()-start<args.seconds:
            now=time.monotonic();t=now-start;index=min(len(trajectory)-1,int(t*fps))
            if index==last_index:time.sleep(.001);continue
            dt=now-previous;previous=now
            last=limit_step(last,trajectory[index],speeds,max(dt,1e-6))
            goals={sid:int(round(v)) for sid,v in zip(ids,last)}
            raw=goal_packet(goals,{sid:300 for sid in ids})
            events.append({'frame':index,'lateness_ms':(t-index/fps)*1000,'bytes':len(raw)})
            last_index=index
        save('scheduler_simulation.json',{'hardware_connected':False,'audio_clock':'simulated monotonic clock; no physical A/V validation',
         'speed_counts_s':300,'elapsed_seconds':time.monotonic()-start,'frames_sent':len(events),
         'max_lateness_ms':max(e['lateness_ms'] for e in events),'events':events})
    elif args.command=='play':
        live_preflight(profile,ids)
        from xiaobai_runtime.playback import play
        result=play(profile,ids,trajectory,fps,args.audio,args.output/'playback_log.json',ROOT/'outputs/xiaobai/STOP',args.max_seconds)
        print('Playback completed:',result['completed'])
