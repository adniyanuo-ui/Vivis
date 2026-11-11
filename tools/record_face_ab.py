#!/usr/bin/env python3
"""Record unannotated local camera frames for face A/B; no microphone or motors."""
import argparse,json,re,subprocess,sys,time
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--person',required=True,help='Registered ID or unknown; supplied by the operator, never inferred')
    p.add_argument('--scene',choices=('normal','backlight','turning','empty'),required=True)
    p.add_argument('--seconds',type=float,default=20);p.add_argument('--delay',type=float,default=5)
    p.add_argument('--output',type=Path)
    p.add_argument('--preview',action='store_true',help='Open local framing page, then wait for Enter before countdown')
    args=p.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_-]{1,39}',args.person) or not 5<=args.seconds<=120 or not 0<=args.delay<=30:p.error('Invalid capture arguments')
    if args.scene=='empty' and args.person!='unknown':p.error('Empty scene must use --person unknown')
    active=subprocess.run(['systemctl','is-active','vivis-p1.service'],capture_output=True,text=True).stdout.strip()
    if active in ('active','activating','deactivating'):p.error('先执行 sudo systemctl stop vivis-p1.service 释放相机；录制工具不会停止服务或驱动机器头。')
    from vivis.paths import ROOT
    from vivis.runtime.robot import Robot
    from vivis.vision.detection import LatestCamera
    import cv2
    import numpy as np
    folder=args.output or ROOT/'private/face_ab'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+args.person+'_'+args.scene)
    folder.mkdir(parents=True,exist_ok=False,mode=0o700)
    camera_config=Robot(ROOT/'configs/robots/xiaoshuai/robot.json').tracking()['camera']
    print('仅录相机，不采音、不控制舵机。请保持单人入镜；身份标签由操作者确认。',flush=True)
    writer=None;camera=None;preview=None;completed=False;count=0;first=None
    try:
        camera=LatestCamera(camera_config)
        if args.preview:
            from vivis.vision.recording_preview import FramingPreview
            preview=FramingPreview(camera)
            print('取景：http://127.0.0.1:8765/ ；请确认整张脸入镜。',flush=True)
            if not preview.ready.wait(5) or preview.error:raise RuntimeError('Framing preview did not become ready')
            input('画面合适后，请在这个终端按 Enter 开始倒计时：')
            if preview.error:raise RuntimeError('Framing preview failed')
            preview.phase='countdown'
        print(f'{args.delay:g}秒后开始，录制{args.seconds:g}秒：{args.person} / {args.scene}',flush=True)
        time.sleep(args.delay);deadline=time.monotonic()+args.seconds;last_sequence=-1;last_good=time.monotonic()
        if preview is not None:preview.phase='recording'
        with (folder/'timestamps.jsonl').open('w') as stamps:
            while time.monotonic()<deadline:
                sample=camera.get()
                if sample is None or sample[0]==last_sequence or time.monotonic()-sample[1]>.5:
                    if time.monotonic()-last_good>3:raise RuntimeError('Camera stopped providing fresh frames')
                    time.sleep(.002);continue
                sequence,stamp,image=sample;last_sequence=sequence;last_good=time.monotonic()
                if writer is None:
                    h,w=image.shape[:2]
                    writer=cv2.VideoWriter(str(folder/'raw.avi'),cv2.VideoWriter_fourcc(*'FFV1'),camera_config['fps'],(w,h))
                    if not writer.isOpened():raise RuntimeError('Lossless FFV1 encoder unavailable; no lossy fallback')
                    first=stamp
                writer.write(np.ascontiguousarray(image))
                stamps.write(json.dumps(dict(frame=count,sequence=sequence,elapsed_s=stamp-first))+'\n');count+=1
        if not count:raise RuntimeError('No frames recorded')
        completed=True
    finally:
        if writer is not None:writer.release()
        if preview is not None:preview.close()
        if camera is not None:camera.close()
        metadata=dict(schema_version='vivis-face-ab-capture-1.0',completed=completed,frames=count,
            person_id=None if args.person=='unknown' else args.person,expected_faces=0 if args.scene=='empty' else 1,
            scene=args.scene,path=str((folder/'raw.avi').resolve()),timestamps=str((folder/'timestamps.jsonl').resolve()),
            camera=camera_config,annotated_preview=False,labels_source='operator_supplied',audio_recorded=False)
        (folder/'capture.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print('录制完成：'+str(folder),flush=True)
if __name__=='__main__':main()
