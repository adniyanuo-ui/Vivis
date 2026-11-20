#!/usr/bin/env python3
"""Local camera lifecycle experiment. No face images, microphone or actuators.

Each trial runs in an isolated process with the production camera settings.
The camera-service lock is inherited by the worker, including a stuck worker;
a new trial must never race an unreaped driver operation.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def production_worker(config,seconds,event,started):
    """Exercise the actual runtime capture/configuration path, not a copy."""
    from vivis.vision.detection import LatestCamera
    camera=LatestCamera(config,progress=event)
    event('capture_start',**camera.actual)
    begin=time.monotonic();frames=0;last_sequence=-1;first=last=None;maximum_gap=0.;shape=None
    try:
        while time.monotonic()-begin<seconds:
            sample=camera.get();now=time.monotonic()
            if sample is not None and sample[0]!=last_sequence:
                frames+=1;last_sequence=sample[0];shape=list(sample[2].shape)
                if first is None:first=now;event('first_frame',shape=shape)
                if last is not None:maximum_gap=max(maximum_gap,now-last)
                last=now
            time.sleep(.002)
        event('capture_complete',frames=frames,shape=shape,capture_s=time.monotonic()-begin,
              first_frame_s=None if first is None else first-started,
              fps_measured=frames/(time.monotonic()-begin),maximum_frame_gap_s=maximum_gap)
    finally:
        event('release_start');camera.close()
        event('release_complete' if not camera.thread.is_alive() else 'release_incomplete')
    return 0 if frames and not camera.thread.is_alive() else 4


def worker(config, seconds, strategy):
    started=time.monotonic()
    def event(stage,**data):
        print(json.dumps(dict(stage=stage,elapsed_s=time.monotonic()-started,**data)),flush=True)
    event('import_cv2')
    import cv2
    if strategy=='production':return production_worker(config,seconds,event,started)
    event('open_start',opencv_version=cv2.__version__,device=config['device'])
    capture=cv2.VideoCapture(config['device'],cv2.CAP_V4L2)
    event('open_complete',opened=capture.isOpened())
    if not capture.isOpened():
        capture.release();return 2
    if strategy=='current':
        properties=[('fourcc',cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')),
            ('width',cv2.CAP_PROP_FRAME_WIDTH,config['width']),
            ('height',cv2.CAP_PROP_FRAME_HEIGHT,config['height']),
            ('fps',cv2.CAP_PROP_FPS,config['fps']),('buffersize',cv2.CAP_PROP_BUFFERSIZE,1)]
        for key,prop in [('autofocus',cv2.CAP_PROP_AUTOFOCUS),('backlight',cv2.CAP_PROP_BACKLIGHT),
                         ('brightness',cv2.CAP_PROP_BRIGHTNESS),('gain',cv2.CAP_PROP_GAIN)]:
            if key in config:properties.append((key,prop,float(config[key])))
        for name,prop,value in properties:
            event('set_start',property=name,value=value)
            accepted=capture.set(prop,value)
            event('set_complete',property=name,accepted=accepted,reported=capture.get(prop))
    event('capture_start',width=capture.get(cv2.CAP_PROP_FRAME_WIDTH),height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
          fps_reported=capture.get(cv2.CAP_PROP_FPS),fourcc=int(capture.get(cv2.CAP_PROP_FOURCC)))
    begin=time.monotonic();frames=failures=0;first=None;shape=None;maximum_gap=0.;last=None
    while time.monotonic()-begin<seconds:
        ok,frame=capture.read();now=time.monotonic()
        if ok and frame is not None:
            frames+=1;shape=list(frame.shape)
            if first is None:first=now;event('first_frame',shape=shape)
            if last is not None:maximum_gap=max(maximum_gap,now-last)
            last=now
        else:failures+=1;time.sleep(.02)
    event('capture_complete',frames=frames,failed_reads=failures,shape=shape,
          capture_s=time.monotonic()-begin,first_frame_s=None if first is None else first-started,
          fps_measured=frames/(time.monotonic()-begin),maximum_frame_gap_s=maximum_gap)
    event('release_start');capture.release();event('release_complete')
    return 0 if frames else 3


def trial(command,path,timeout,lock_fd):
    started=time.monotonic()
    with path.open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,
            env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONNOUSERSITE':'1','PYTHONUNBUFFERED':'1'},
            start_new_session=True,pass_fds=(lock_fd,))
        timed_out=False
        while child.poll() is None:
            if time.monotonic()-started>=timeout:
                timed_out=True
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=2)
                    except subprocess.TimeoutExpired:pass
                break
            time.sleep(.05)
    events=[]
    for line in path.read_text(errors='replace').splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    captures=[e for e in events if e.get('stage')=='capture_complete']
    return dict(pid=child.pid,exit_code=child.poll(),timed_out=timed_out,worker_unreaped=child.poll() is None,
                elapsed_s=time.monotonic()-started,last_event=events[-1] if events else None,
                capture=captures[-1] if captures else None,log=str(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    parser.add_argument('--mode',choices=['hold','cycles'],default='hold')
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--cycles',type=int,default=10)
    parser.add_argument('--gap',type=float,default=3)
    parser.add_argument('--startup-budget',type=float,default=12)
    parser.add_argument('--strategy',choices=['current','native','production'],default='current',
        help='production uses LatestCamera; current reproduces legacy unconditional settings; native only opens/reads')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not .2<=args.seconds<=300 or not 1<=args.cycles<=100 or not 0<=args.gap<=30 or not 2<=args.startup_budget<=60:
        parser.error('Invalid bounded experiment duration')
    if args.worker:return worker(json.loads(args.worker.read_text()),args.seconds,args.strategy)
    from vivis.runtime.robot import Robot
    camera=Robot(args.robot).tracking()['camera']
    device=Path(camera['device'])
    if not device.exists():parser.error('Camera device is missing: '+str(device))
    current=subprocess.run(['systemctl','is-active','--quiet','vivis-p1.service'],check=False)
    if current.returncode==0:parser.error('Stop vivis-p1.service first; this tool never stops the robot')
    busy=subprocess.run(['fuser',str(device.resolve())],capture_output=True,text=True)
    if busy.returncode==0:parser.error('Camera already owned by PID(s): '+busy.stdout.strip())
    output=args.output or ROOT/'outputs/diagnostics/camera_lifecycle'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    config=output/'camera.json';config.write_text(json.dumps(camera,ensure_ascii=False,indent=2)+'\n')
    lock_path=ROOT/'outputs/vision/service.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    results=[]
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('A camera service or blocked diagnostic still owns the camera lock')
        count=1 if args.mode=='hold' else args.cycles
        for index in range(count):
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',str(config),
                     '--seconds',str(args.seconds),'--strategy',args.strategy]
            result=trial(command,output/f'trial_{index+1:03d}.jsonl',args.seconds+args.startup_budget,lock.fileno())
            results.append(result)
            report=dict(mode=args.mode,strategy=args.strategy,camera=camera,requested_trials=count,
                        seconds=args.seconds,gap_s=args.gap,no_images_saved=True,actuators_opened=False,results=results)
            (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(dict(trial=index+1,**result),ensure_ascii=False),flush=True)
            if result['timed_out'] or result['exit_code']!=0:break
            if index+1<count:time.sleep(args.gap)
    print('Report:',output/'result.json',flush=True)
    return 0 if len(results)==count and all(r['exit_code']==0 for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
