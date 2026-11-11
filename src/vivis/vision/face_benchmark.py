"""Offline, labelled image/video comparison. Never opens robot devices or writes a gallery."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from vivis.paths import ROOT
from vivis.vision.seeta import Seeta


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data:=stream.read(1024*1024):h.update(data)
    return h.hexdigest()


def percentiles(values):
    return dict(zip(('p10','p50','p90','p95','max'),map(float,np.percentile(values,[10,50,90,95,100])))) if values else None


def rank(scores,threshold,margin):
    ranked=sorted(scores.items(),key=lambda p:p[1],reverse=True)
    if not ranked:return dict(prediction=None,best_id=None,best_score=None,margin=None)
    pid,score=ranked[0];gap=score-ranked[1][1] if len(ranked)>1 else None
    return dict(prediction=pid if score>=threshold and (gap is None or gap>=margin) else None,
                best_id=pid,best_score=score,margin=gap)


def capture_clip(folder):
    folder=Path(folder);metadata=json.loads((folder/'capture.json').read_text())
    if metadata.get('schema_version')!='vivis-face-ab-capture-1.0' or not metadata.get('completed'):raise ValueError('Incomplete capture')
    clip={k:metadata[k] for k in ('path','timestamps','person_id','expected_faces','scene')}
    review=folder/'review.json'
    if review.exists():
        notes=json.loads(review.read_text())
        if notes.get('usable') is not True:raise ValueError('Capture excluded by review: '+str(folder))
        if 'start_s' in notes or 'end_s' in notes:
            start=notes.get('start_s',0.);end=notes.get('end_s')
            if type(start) not in (int,float) or type(end) not in (int,float) or not 0<=start<end:raise ValueError('Invalid reviewed capture interval')
            clip.update(start_s=start,end_s=end)
    return clip




class SFace:
    def __init__(self,width=320):
        import cv2
        from vivis.vision.detection import FaceVision
        from vivis.runtime.robot import Robot
        config=Robot(ROOT/'configs/robots/xiaoshuai/robot.json').tracking()
        config['detector']['max_width']=width
        self.vision=FaceVision(config,ROOT)
        from vivis.vision.identity_config import backend_config
        familiar=backend_config(json.loads((ROOT/'configs/perception/familiar_faces.json').read_text()),'sface')
        self.minimum_face_size=familiar['min_face_size_px']
        model=ROOT/familiar['model']
        if digest(model)!=familiar['sha256']:raise ValueError('SFace model hash mismatch')
        self.recognizer=cv2.FaceRecognizerSF.create(str(model),'')
        self.width=width

    def evaluate(self,image):
        from vivis.vision.familiar import feature
        start=time.perf_counter()
        # Reuse production selection/alignment; JPEG preview generation is included.
        _,_,metrics=self.vision.detect((1,time.monotonic(),image))
        detect_ms=(time.perf_counter()-start)*1000
        row=self.vision.selected_face_row
        size_rejected=row is not None and min(row[2:4])<self.minimum_face_size
        start=time.perf_counter();vector=feature(self.recognizer,image,row) if row is not None and not size_rejected else None
        recognition_ms=(time.perf_counter()-start)*1000
        return dict(faces=metrics['faces'],bbox=list(map(float,row[:4])) if row is not None else None,
                    quality_rejected=False,size_rejected=bool(size_rejected),timing_ms=dict(detection_preview=detect_ms,recognition=recognition_ms)),vector


def frames(entry,root,hz):
    """Only files are accepted. A numeric camera ID or /dev path is never opened."""
    import cv2
    path=(root/entry['path']).resolve()
    if not path.is_file():raise ValueError('Missing dataset file: '+str(path))
    if path.suffix.lower() in ('.jpg','.jpeg','.png'):
        image=cv2.imread(str(path))
        if image is None:raise ValueError('Invalid dataset image')
        yield 0,0.,image;return
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError('Cannot decode dataset video')
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0:raise ValueError('Video lacks valid FPS')
        stamps=None
        if entry.get('timestamps'):
            stamps=[json.loads(l)['elapsed_s'] for l in (root/entry['timestamps']).read_text().splitlines()]
        index=0;next_time=float(entry.get('start_s',0.));end=float(entry.get('end_s',float('inf')))
        while True:
            ok,image=cap.read()
            if not ok:break
            if stamps is not None and index>=len(stamps):raise ValueError('More video frames than recorded timestamps')
            stamp=float(stamps[index]) if stamps is not None else index/fps
            if stamp>=end:break
            if stamp+1e-6>=next_time:
                yield index,stamp,image
                next_time=stamp+1/hz
            index+=1
    finally:cap.release()


def summarize(records,temporal):
    """Frame denominators include detection/quality rejection; never hide discarded frames."""
    known=[r for r in records if r['truth'] is not None and r['expected_faces']==1]
    stranger=[r for r in records if r['truth'] is None and r['expected_faces']==1]
    empty=[r for r in records if r['expected_faces']==0]
    single=known+stranger
    rejected=sum(r['prediction'] is None for r in known)
    wrong=sum(r['prediction'] is not None and r['prediction']!=r['truth'] for r in known)
    false_accept=sum(r['prediction'] is not None for r in stranger)
    transitions=[]
    if temporal:
        previous={}
        for r in records:
            key=r['clip'];label=r['prediction']
            if key in previous:transitions.append(int(previous[key]!=label))
            previous[key]=label
    ratio=lambda n,rows:n/len(rows) if rows else None
    return dict(frames=len(records),known_face_frames=len(known),unknown_face_frames=len(stranger),
        known_rejected_frames=rejected,known_wrong_id_frames=wrong,unknown_false_accept_frames=false_accept,
        detection_coverage=ratio(sum(r['faces']>0 for r in single),single),
        quality_rejection_rate=ratio(sum(r.get('quality_rejected',False) for r in single),single),
        known_rejection_rate=ratio(rejected,known),known_wrong_id_rate=ratio(wrong,known),
        unknown_false_accept_rate=ratio(false_accept,stranger),
        empty_scene_false_accept_rate=ratio(sum(r['prediction'] is not None for r in empty),empty),
        raw_decision_transitions=sum(transitions) if temporal else None,
        transition_opportunities=len(transitions) if temporal else None,
        true_person_scores=percentiles([r['scores'][r['truth']] for r in known if r['truth'] in r['scores']]),
        other_person_scores=percentiles([s for r in known for pid,s in r['scores'].items() if pid!=r['truth']]),
        wall_ms=percentiles([r['wall_ms'] for r in records]))


def run(manifest,output,repeats=1):
    import cv2
    manifest=Path(manifest).resolve();data=json.loads(manifest.read_text())
    if data.get('schema_version')!='vivis-face-ab-1.0':raise ValueError('Unsupported A/B manifest')
    if not data.get('gallery') or not data.get('clips'):raise ValueError('Empty gallery or clips')
    if not 1<=repeats<=20 or not .5<=data.get('sample_hz',2)<=10:raise ValueError('Invalid benchmark sampling')
    root=manifest.parent;cv2.setNumThreads(1);seeta=Seeta(1)
    try:
        baseline=SFace(320);full=SFace(640)
        galleries={name:{} for name in ('sface320','sface640','seeta6','seeta6_quality')}
        enrollment=[]
        baseline_path=(root/data['baseline_gallery']).resolve()
        with np.load(baseline_path,allow_pickle=False) as saved:
            vectors=saved['vectors'];person_ids=saved['ids']
        if vectors.ndim!=2 or vectors.shape[1]!=128 or len(vectors)!=len(person_ids):raise ValueError('Invalid existing SFace gallery')
        if set(person_ids.tolist())!={g['person_id'] for g in data['gallery']}:raise ValueError('Gallery IDs differ between arms')
        for pid,vector in zip(person_ids,vectors):
            for name in ('sface320','sface640'):galleries[name].setdefault(str(pid),[]).append(np.asarray(vector,dtype=np.float32))
        for item in data['gallery']:
            photo=(root/item['path']).resolve();image=cv2.imread(str(photo))
            if image is None:raise ValueError('Invalid registration image: '+str(photo))
            # Match the existing register_person.py preprocessing. Baseline uses
            # its actual saved vectors, never a newly enrolled diagnostic gallery.
            h,w=image.shape[:2];scale=min(1.,1280/max(h,w))
            image=cv2.resize(image,(round(w*scale),round(h*scale)))
            for name,engine in (('seeta6',seeta),):
                result,vector=engine.evaluate(image)
                if result['faces']!=1 or vector is None:raise ValueError('Registration needs exactly one detectable face: '+item['person_id']+' / '+name)
                galleries[name].setdefault(item['person_id'],[]).append(vector)
                enrollment.append(dict(person_id=item['person_id'],arm=name,image_sha256=digest(photo),
                                       brightness=result.get('brightness'),clarity=result.get('clarity')))
        galleries['seeta6_quality']=galleries['seeta6']
        ids=set(galleries['sface320'])
        if any(c.get('person_id') not in ids for c in data['clips'] if c.get('person_id') is not None):raise ValueError('Unknown ground-truth ID')
        records=[];input_hashes=[]
        for clip_index,entry in enumerate(data['clips']):
            if entry.get('expected_faces') not in (0,1):raise ValueError('Label expected_faces as zero or one; split/exclude multi-person segments')
            if entry['expected_faces']==0 and entry.get('person_id') is not None:raise ValueError('Empty scene cannot have a person ID')
            input_hashes.append(dict(path=entry['path'],sha256=digest(root/entry['path'])))
            for index,stamp,image in frames(entry,root,data.get('sample_hz',2)):
                # Rotate arm order to limit systematic cache/thermal ordering effects.
                arms=[('sface320',baseline,False),('sface640',full,False),('seeta6',seeta,False),('seeta6_quality',seeta,True)]
                offset=len(records)//4%4;arms=arms[offset:]+arms[:offset]
                for name,engine,gate in arms:
                    durations=[]
                    for _ in range(repeats):
                        start=time.perf_counter()
                        result,vector=engine.evaluate(image,gate) if isinstance(engine,Seeta) else engine.evaluate(image)
                        durations.append((time.perf_counter()-start)*1000)
                    scores={}
                    if vector is not None:
                        for pid,vectors in galleries[name].items():
                            scores[pid]=max(seeta.compare(vector,v) if isinstance(engine,Seeta) else float(vector@v) for v in vectors)
                    limits=data.get('thresholds',{}).get('seeta6' if isinstance(engine,Seeta) else 'sface',{})
                    decision=rank(scores,limits.get('score',.6 if isinstance(engine,Seeta) else .5),limits.get('margin',.08))
                    # Multi-face or empty-scene input cannot authorize a person.
                    if result['faces']!=1:decision['prediction']=None
                    records.append(dict(arm=name,clip=clip_index,frame=index,time_s=stamp,truth=entry.get('person_id'),
                        expected_faces=entry['expected_faces'],scores=scores,wall_ms=float(np.median(durations)),
                        **result,**decision))
        temporal=data.get('kind')=='video'
        report=dict(schema_version='vivis-face-ab-report-1.0',input_manifest_sha256=digest(manifest),
            baseline_gallery_sha256=digest(baseline_path),
            seeta_build_manifest_sha256=digest(ROOT/'outputs/face_ab/build_manifest.json'),
            seeta_probe_sha256=digest(ROOT/'outputs/face_ab/build/probe/libvivis_seeta_probe.so'),
            seeta_quality_library_sha256=digest(ROOT/'outputs/face_ab/src/SeetaFace6Open/build/lib/libSeetaQualityAssessor300.so'),
            annotated_preview=bool(data.get('annotated_preview')),temporal=temporal,repeats=repeats,
            gallery_images=enrollment,inputs=input_hashes,
            thresholds=data.get('thresholds',dict(sface=dict(score=.5,margin=.08),seeta6=dict(score=.6,margin=.08))),
            note='Raw per-frame decisions, BEFORE production confirmation/review. Scores from different models are not calibrated to each other. Detection coverage has no box-IoU annotation.',
            summary={name:summarize([r for r in records if r['arm']==name],temporal) for name in galleries})
        output=Path(output);output.mkdir(parents=True,exist_ok=False,mode=0o700)
        (output/'frames.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        def cell(value):return 'N/A' if value is None else f'{value:.1%}'
        lines=['# 人脸识别离线A/B结果','',
            '所有组使用相同帧和登记照片。以下为确认／复核层之前的逐帧决策；不同模型分数不能直接比较大小。',
            '带框截图仅能预检查，不支持时序结论。检测覆盖率尚未经过真实人脸框IoU标注。','',
            '| 方案 | 人脸帧检测覆盖 | 已知人拒识 | 已知人串人 | 陌生人误认 | 质量拒绝 | 总耗时P50/P95 |',
            '|---|---|---|---|---|---|---|']
        for name,row in report['summary'].items():
            t=row['wall_ms'];timing='N/A' if t is None else f'{t["p50"]:.1f}/{t["p95"]:.1f} ms'
            lines.append('| '+' | '.join([name,*[cell(row[k]) for k in ('detection_coverage','known_rejection_rate','known_wrong_id_rate','unknown_false_accept_rate','quality_rejection_rate')],timing])+' |')
        lines+=['','逐帧分数、质量等级和分阶段耗时见同目录`frames.jsonl`；分母、计数、分数分布及输入哈希见`report.json`。',
                '这份报告不自动替换运行中的模型，也不能用少量样本中零次串人宣称真实串人率为零。']
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        return report
    finally:seeta.close()
