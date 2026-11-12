#!/usr/bin/env python3
"""Build approved songs offline, then register complete assets for one robot.

The source mix is retained in incoming/. A separate vocal stem drives native
A2F once; normal playback only reads the resulting static targets and audio.
No device is opened. Optional model tools are not runtime dependencies.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vivis.interaction.animation import A2FAdapter
from vivis.motion.mapping import compile_trajectory,inputs,read_json
from vivis.runtime.robot import Robot
from separate_song_vocals import read_wav,write_wav


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data,pretty=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2 if pretty else None,
                                   separators=None if pretty else (',',':'),allow_nan=False)+'\n')
    temporary.replace(path)


def source_clip(report,frames,order,duration):
    adapter=A2FAdapter(order[:52],'offline-song')
    adapter.bind(dict(schema_version='vivis-a2f-bindings-1.0',playback_id='offline-song',names=report['names']))
    stamps=[];weights=[]
    for frame in frames:
        value=adapter.adapt({**frame,'schema_version':'vivis-a2f-raw-1.0','playback_id':'offline-song'})
        stamps.append(value['audio_sample_index']/16000);weights.append(value['weights'])
    if (not stamps or stamps[0]>1/60 or duration-stamps[-1]>.1
            or len(stamps)>1 and np.max(np.diff(stamps))>.05):
        raise ValueError('Incomplete A2F coverage')
    values=np.array(weights);timeline=np.arange(math.ceil(duration*60))/60
    result=np.zeros((len(timeline),61))
    for col in range(52):result[:,col]=np.interp(timeline,stamps,values[:,col])
    return dict(name='apple',fps=dict(num=60,den=1),blendShapes=np.round(result,7).tolist())


def build(row,robot,engine,build_root):
    name=row['work_id'];source=ROOT/row['audio_path']
    if digest(source)!=row['audio_sha256']:raise ValueError('Changed accepted source: '+name)
    stage=build_root/name;stage.mkdir(parents=True,exist_ok=True)
    assets=ROOT/'assets/works/songs'/name;assets.mkdir(parents=True,exist_ok=True)
    marker=assets/'provenance.json'
    if marker.exists():
        previous=read_json(marker)
        if previous['source_sha256']!=row['audio_sha256']:raise ValueError('Do not overwrite another accepted version')
    vocals=stage/'vocals.wav'
    if not vocals.exists() or not vocals.with_suffix('.json').exists():
        with (stage/'separation.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'tools/separate_song_vocals.py'),'--engine',str(engine),
                '--audio',str(source),'--output',str(vocals)],check=True,stdout=log,stderr=subprocess.STDOUT)
    vocal_report=read_json(vocals.with_suffix('.json'))
    if vocal_report['source_sha256']!=digest(source) or vocal_report['inference_sha256']!=digest(vocals):
        raise ValueError('Vocal cache does not match source')
    mix,rate=read_wav(source);duration=mix.shape[1]/rate
    # Same sample-zero timeline; no trimming, loudness gain, or silence edits.
    audio=assets/'audio.wav';g=math.gcd(rate,24000)
    write_wav(audio,resample_poly(mix.mean(axis=0),24000//g,rate//g),24000)
    native=stage/'native'
    if not (native/'report.json').exists():
        if native.exists():raise ValueError('Incomplete native run; review and remove its directory before retry')
        with (stage/'a2f.log').open('w') as log:
            subprocess.run([str(ROOT/'outputs/a2f/build/a2f_chunk_probe'),str(ROOT/'outputs/a2f/model_bundle/model.json'),
                str(vocals),str(native),str(duration)],check=True,stdout=log,stderr=subprocess.STDOUT)
    report=read_json(native/'report.json')
    if not report.get('completed') or report.get('invalid_weights') or abs(report['audio_duration_seconds']-duration)>1/16000:
        raise ValueError('Invalid native inference result')
    with robot.resolve('channels').open(encoding='utf-8-sig',newline='') as f:order=next(csv.reader(f))[2:]
    clip=assets/'source.json'
    with (native/'frames.ndjson').open() as f:save(clip,source_clip(report,(json.loads(line) for line in f),order,duration))
    provenance=dict(source_url=row['download_url'],source_title=row['source_title'],source_sha256=digest(source),
        accepted_by='user',accepted_date='2026-11-12',duration_s=duration,
        playback='24kHz mono PCM16 derived from full source mix; no gain or time edits',
        expression='UVR Voc_FT separated vocals -> existing native A2F -> canonical52 + zero gaze channels -> 60FPS linear timestamp sampling',
        vocal_report=vocal_report,separator_model_sha256=digest(ROOT/'outputs/song_build/models/UVR-MDX-NET-Voc_FT.onnx'),
        a2f_model_sha256=digest(ROOT/'outputs/a2f/model_bundle/model.json'),a2f_binary_sha256=digest(ROOT/'outputs/a2f/build/a2f_chunk_probe'),
        a2f_frames=report['frames'],mouth_gain=1.,source_fps=60,onsite_motion_validated=False)
    save(marker,provenance)
    return map_work(robot,name,clip,audio,marker)


def map_work(robot,name,clip,audio,marker):
    """Reuse the shared expression/audio; only calibration targets are per robot."""
    values,fps,order,shapes=inputs(clip,robot.calibration,robot.resolve('channels'),allowed_motor_ids=set(robot.axes))
    if fps!=60:raise ValueError('Expected prepared60FPS song source')
    ids,positions,mapping_report=compile_trajectory(values,order,shapes,robot.motion_profile(),'face_eyes')
    target=ROOT/'assets/works'/robot.definition['name']/name/'targets.json'
    save(target,dict(schema_version='mojia-motor-targets-0.1',device_limits_applied=True,
        limits_source=str(robot.resolve('hardware').relative_to(ROOT)),calibration_sha256=robot.calibration_sha256,
        audio_sha256=digest(audio),source_clip_sha256=digest(clip),source_fps=60,
        fps=dict(num=60,den=1),frames=len(values),motor_ids=ids,positions=positions.tolist(),trajectory_report=mapping_report))
    entry=dict(name=name,calibration_sha256=robot.calibration_sha256,fps=60,frames=len(values),motor_ids=ids)
    for key,path in [('audio',audio),('targets',target),('source_clip',clip),('provenance',marker)]:
        entry[key]=str(path.relative_to(ROOT));entry[key+'_sha256']=digest(path)
    manifest=read_json(robot.resolve('works'));manifest['works'][name]=entry
    save(robot.resolve('works'),manifest,pretty=True)
    return entry


def register_programs(rows,robot,catalog_path):
    """Publish only fully built works; pending URLs are never advertised."""
    from vivis.runtime.assets import verify_assets
    verify_assets(robot)
    catalog=read_json(catalog_path)
    existing={p['id'] for p in catalog['programs']}
    for row in rows:
        if row['status']!='ready':continue
        robot.work(row['work_id'])
        if row['work_id'] in existing:continue
        aliases=[row['title'],*filter(None,row['aliases'].split('|')),
                 row['artist']+' '+row['title'],row['artist']+'的'+row['title']]
        catalog['programs'].append(dict(id=row['work_id'],title=row['title'],artist=row['artist'],
            kind='prerecorded',work=row['work_id'],category='song',realtime_response=False,
            aliases=list(dict.fromkeys(aliases))))
        existing.add(row['work_id'])
    save(catalog_path,catalog,pretty=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',type=Path,default=ROOT/'configs/robots/xiaoshuai/robot.json')
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--intake',type=Path,default=ROOT/'assets/program_requests/songs_20261112.csv')
    p.add_argument('--only',help='Build one accepted work')
    p.add_argument('--register-only',action='store_true',help='Verify assets and publish ready songs; no inference')
    p.add_argument('--retarget-only',action='store_true',help='Map existing ready source clips to another robot; no inference')
    a=p.parse_args();robot=Robot(a.robot);rows=list(csv.DictReader(a.intake.open()))
    if a.register_only:
        register_programs(rows,robot,ROOT/'configs/interaction/programs.json');return
    approved=[r for r in rows if r['status'] in ('approved_for_build','ready') and (not a.only or r['work_id']==a.only)]
    if not approved:raise ValueError('No approved works')
    started=time.monotonic()
    for i,row in enumerate(approved,1):
        print(f'BUILD {i}/{len(approved)} {row["work_id"]}',flush=True)
        if a.retarget_only:
            if row['status']!='ready':raise ValueError('Cannot retarget unfinished song')
            source=ROOT/'assets/works/songs'/row['work_id']
            map_work(robot,row['work_id'],source/'source.json',source/'audio.wav',source/'provenance.json')
            print('RETARGET READY '+row['work_id'],flush=True);continue
        build(row,robot,a.engine,ROOT/'outputs/song_build');row['status']='ready'
        row['note']='用户已采用；完整音频、离线A2F及小帅标定目标已生成，现场口型待验收。'
        with a.intake.with_suffix('.tmp').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        a.intake.with_suffix('.tmp').replace(a.intake)
        print(f'READY {row["work_id"]} elapsed={time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__':main()
