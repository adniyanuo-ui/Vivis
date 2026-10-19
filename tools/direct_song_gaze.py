"""Visible, phrase-directed gaze for the existing Claire song performance."""
import json
import copy
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

root=Path(__file__).resolve().parent.parent
source=root/'outputs/0001_SingingNarrative_VisibleBrows_FaceMotion52_v3.0'
out=root/'outputs/0001_SingingNarrative_Gaze_VisibleBrows_FaceMotion52_v3.0'
out.mkdir(exist_ok=True)
# Source animation is immutable. Write new eyelid geometry and render in a separate folder.
for name in ['mesh.json','timestamps.csv','export_info.json']:
    target=out/name
    if not target.exists():os.link(source/name,target)
meta=json.loads((source/'metadata.json').read_text());duration=meta['duration_seconds']
# start seconds, movement duration, target local yaw, target local pitch, intent.
# Positive yaw looks toward screen right; positive pitch looks down in this renderer.
# Targets are artistic interpretation of tentative ASR and vocal phrase timing.
events=[
 (0,.0,-4,3,'开头留白，目光稍低'),
 (1.30,.18,-1,1,'眨眼后恢复注意'),
 (2.65,.15,0,-1,'期待开口，看向观众'),
 (5.22,.18,-14,4,'期待回落，短暂侧看'),
 (7.85,.16,-5,2,'注意收回'),
 (9.25,.15,9,1,'逞强时短暂移开视线'),
 (11.10,.18,0,1,'句末回视'),
 (14.70,.22,-9,5,'音乐留白，回想'),
 (18.82,.20,0,0,'准备进入主歌'),
 (23.70,.18,7,2,'回忆中的短暂侧看'),
 (25.22,.16,0,0,'继续向观众讲述'),
 (30.95,.19,-8,3,'乐句转折'),
 (32.20,.16,0,0,'重新关注叙述'),
 (35.12,.18,-12,4,'出门后的担心'),
 (37.10,.17,-2,1,'回到倾诉'),
 (39.48,.16,11,2,'收住情绪，视线回避'),
 (41.55,.16,0,0,'逞强后回视'),
 (44.32,.19,6,3,'句尾缓和'),
 (46.97,.18,0,0,'新一句前回视'),
 (48.70,.14,-15,-1,'探看，迅速寻找'),
 (50.15,.13,-7,0,'短暂停留后收回一些'),
 (52.43,.16,0,0,'准备向对方表达'),
 (56.34,.15,5,4,'欲言又止，第一次移开'),
 (58.55,.17,13,10,'犹豫加深，侧下方停留'),
 (61.13,.24,6,12,'算了，闭眼后垂下目光'),
 (63.08,.20,0,1,'副歌重新面对观众'),
 (66.96,.16,-5,2,'温柔句尾稍缓和'),
 (68.05,.15,0,0,'继续倾诉'),
 (70.57,.22,2,5,'长音内收，伴随缓慢闭眼'),
 (72.59,.18,0,0,'下一乐句恢复目光交流'),
 (75.35,.18,-8,4,'守候感，短暂回想'),
 (77.57,.17,0,1,'回到观众'),
 (80.10,.20,3,7,'脆弱感，视线渐低'),
 (82.88,.25,-4,10,'长音闭眼后保留内收'),
 (85.32,.22,-11,6,'尾音释放，延续余韵'),
 (90.12,.22,-7,4,'间奏留白，保持同侧注视'),
 (94.05,.20,0,2,'间奏中恢复目光交流'),
 (96.42,.20,8,4,'音乐推进，另一次轻微侧看'),
 (100.02,.18,1,1,'重新集中注意'),
 (104.70,.18,0,0,'人声重新进入前回视'),
 (109.30,.16,-5,2,'跟随句意轻微移开'),
 (111.62,.18,0,1,'片段结尾回到观众')]
times=np.unique(np.r_[np.arange(0,duration,1/60),duration])
values=np.zeros((len(times),2));current=np.array([-4.,3.])
for i,(start,span,yaw,pitch,intent) in enumerate(events):
    end=events[i+1][0] if i+1<len(events) else duration+1
    mask=(times>=start)&(times<end)
    u=np.clip((times[mask]-start)/max(span,1e-6),0,1)
    # Quintic easing gives finite speed and a stationary fixation at either end.
    ease=u*u*u*(10-15*u+6*u*u)
    target=np.array([yaw,pitch],dtype=float)
    values[mask]=current+(target-current)*ease[:,None]
    current=target
keys=[dict(time_seconds=float(t),yaw_deg=float(v[0]),pitch_deg=float(v[1])) for t,v in zip(times,values)]
gaze={'source':'song-specific artistic gaze targets based on tentative words and musical phrases; not A2F-inferred gaze',
 'coordinate_convention':'preview-local Blender: positive yaw toward screen right, positive pitch downward',
 'events':[dict(start_seconds=t,movement_seconds=d,yaw_deg=y,pitch_deg=p,intent=i) for t,d,y,p,i in events],
 'keyframes':keys,'lyrics_status':'unverified automatic transcription; artistic interpretation'}
# A small delayed follow-through in the existing bust pose supports larger eye turns.
oldkeys=meta['performance_pose']['keyframes'];oldtimes=np.array([k['time_seconds'] for k in oldkeys])
follow=gaussian_filter1d(values,sigma=12,axis=0)
poses=[]
for t in np.unique(np.r_[np.arange(0,duration,.1),duration]):
    item={'time_seconds':float(t)}
    for axis in ['pitch_deg','roll_deg','yaw_deg']:
        base=float(np.interp(t,oldtimes,[k[axis] for k in oldkeys]))
        addition=0
        if axis=='yaw_deg':addition=.12*float(np.interp(t-.22,times,follow[:,0]))
        elif axis=='pitch_deg':addition=.075*float(np.interp(t-.22,times,follow[:,1]))
        item[axis]=base+addition
    poses.append(item)
# Downward gaze is accompanied by modest upper-lid lowering. Preserve complete blinks.
from facemotion52 import load_stream,validate_frame
frames=load_stream(source/'facemotion52_singing.jsonl')
stream_path=out/'facemotion52_singing.jsonl'
temporary=out/'facemotion52_singing.tmp'
with temporary.open('w') as stream:
    for original in frames:
        f=copy.deepcopy(original);t=f['timestamp_ms']/1000
        down=max(0,float(np.interp(t,times,values[:,1])))
        lid_follow=min(.22,.018*down)
        for side in ['Left','Right']:
            name='eyeBlink'+side;old=f['blendshapes'][name]
            f['blendshapes'][name]=old+(1-old)*lid_follow
            name='eyeWide'+side
            f['blendshapes'][name]*=1-lid_follow
        validate_frame(f)
        assert f['extensions']==original['extensions']
        assert all(f['blendshapes'][n]==original['blendshapes'][n] for n in f['blendshapes'] if not n.startswith(('eyeBlink','eyeWide')))
        stream.write(json.dumps(f,separators=(',',':'),allow_nan=False)+'\n')
os.replace(temporary,stream_path)
# Ensure replay can replace these files without writing through old hard links.
for name in ['skin.f32','tongue.f32','jaw.f32','mesh.json','timestamps.csv','export_info.json']:
    path=out/name
    if path.exists() and path.stat().st_nlink>1:path.unlink()
meta.update(interface_source=str(out/'facemotion52_singing.jsonl'),performance_gaze=gaze,
 performance_pose={'source':'existing preview bust pose plus small delayed eye-turn follow-through; not robot commands','keyframes':poses},
 preview_times_seconds=[3.7,5.8,9.6,35.7,40,49,54.9,58.9,60.7,63.8,80.7,86.2,96.9],
 gaze_revision='visible phrase-directed fixations and short eased eye turns',
 gaze_parent_video='0001_CLAIRE_SINGING_NARRATIVE_v3.0-Claire.mp4')
(out/'performance_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
(out/'performance_gaze.json').write_text(json.dumps(gaze,ensure_ascii=False,indent=2)+'\n')
report={'mouth_jaw_tongue_brow_expression_unchanged':True,'eyelid_follow_added':True,'source_frames':6774,'video_frames_expected':3387,
 'a2f_rerun':False,'reason':'reuse song articulation and brow/mouth expression; add gaze, eyelid follow and preview pose follow-through',
 'gaze_events':len(events),'yaw_range_degrees':[float(values[:,0].min()),float(values[:,0].max())],
 'pitch_range_degrees':[float(values[:,1].min()),float(values[:,1].max())],
 'max_gaze_speed_degrees_per_second':float(np.linalg.norm(np.diff(values,axis=0)/np.diff(times)[:,None],axis=1).max()),
 'facemotion_gaze_scope':'gaze is stored in preview metadata, not core FaceMotion fields or robot servo commands',
 'visual_acceptance':'pending preview and full-video checks','frame_schema_validation':'passed'}
assert np.isfinite(values).all()
assert np.max(abs(values[:,0]))<=15 and np.max(abs(values[:,1]))<=12
assert report['max_gaze_speed_degrees_per_second']<210
(out/'gaze_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
