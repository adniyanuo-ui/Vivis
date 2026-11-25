"""Local, explicit administrator selection of journal audio into voice candidates.

Never records devices, modifies original rounds, or activates an inferred label.
The review server runs this in the speech environment on a bounded worker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import wave

import numpy as np

from vivis.interaction.memory import person_key
from vivis.speech.speaker_gallery import RevisionConflict,read_gallery,sample_key
from vivis.speech.speaker_observation import SpeakerObserver,audio_quality,normalized,CONSISTENCY_THRESHOLD


def private_path(root,path):
    value=(Path(root)/path).resolve()
    if not value.is_relative_to((Path(root)/'private').resolve()):raise ValueError('私有录音路径越界')
    return value


def read_pcm(path):
    with wave.open(str(path),'rb') as w:
        if (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getcomptype())!=(1,2,16000,'NONE'):
            raise ValueError('需要16kHz单声道PCM16原始用户录音')
        if not 0<w.getnframes()<=16000*120:raise ValueError('录音为空或超过120秒，请分段选择')
        pcm=w.readframes(w.getnframes())
        if len(pcm)!=w.getnframes()*2:raise ValueError('录音文件不完整')
    return pcm


def round_sources(root,ids,*,same_visit=True):
    if not isinstance(ids,list) or not 1<=len(ids)<=32 or len(set(ids))!=len(ids):raise ValueError('请选择1至32条不重复的原话轮')
    for rid in ids:sample_key(rid)
    c=json.loads((Path(root)/'configs/interaction/continuity.json').read_text())
    folder=private_path(root,c['directory']);path=folder/'rounds.sqlite3'
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=1);db.row_factory=sqlite3.Row
    try:
        rows=[]
        for rid in ids:
            row=db.execute('SELECT * FROM rounds WHERE id=?',(rid,)).fetchone()
            if row is None:raise ValueError('原话轮已不存在，请刷新')
            if not row['closed'] or row['alignment']!='aligned' or not row['audio'] or not row['user_text']:
                raise ValueError('只能选择已定稿、输入录音完整且已结束的原话轮')
            rows.append(dict(row))
    finally:db.close()
    # A manual short-speech group still belongs to one visit, not mixed visitors.
    if same_visit and len({r['session'] for r in rows})!=1:raise ValueError('合并短句须来自同一次来访，请分次选择')
    rows.sort(key=lambda r:(r['created'],r['id']))
    for row in rows:
        row['path']=private_path(root,folder/row['audio'])
        if not row['path'].is_relative_to(folder):raise ValueError('原始录音路径越界')
    return rows


def import_rounds(root,person,ids,revision,*,confirmed=False,prepare=False,extractor=None):
    root=Path(root).resolve();person_key(person)
    familiar=json.loads((root/'configs/perception/familiar_faces.json').read_text())
    profile=private_path(root,root/familiar['people_directory']/person/'profile.json')
    if not profile.is_file():raise ValueError('请先建立该熟人的档案')
    if not confirmed and not prepare:raise ValueError('请先试听，确认所选录音均为该人单独说话且无明显回声')
    if type(revision) is not int or revision<0:raise ValueError('声纹版本不正确，请刷新')
    continuity=json.loads((root/'configs/interaction/continuity.json').read_text())
    c=json.loads((root/continuity['speaker_config']).read_text())
    gallery_path=private_path(root,root/c['directory'])/'gallery.json'
    document=read_gallery(gallery_path,c['sha256'])
    sources=round_sources(root,ids);source_ids=[r['id'] for r in sources];wanted=set(source_ids)
    for pid,refs in document['people'].items():
        for row in refs:
            used=set(row.get('source_turn_ids',[row['turn_id']]))
            if wanted&used:
                if pid==person and wanted==used:
                    return dict(ok=True,sample_id=row['turn_id'],active=row.get('active',True),already_saved=True,
                        message='这些原话已在声纹库中，未重复添加。')
                raise ValueError('所选原话已有声纹来源，不能重复或交叉登记')
    if document['revision']!=revision:raise RevisionConflict('声纹样本已变化，请刷新后重试')
    if len(document['people'].get(person,[]))>=c['maximum_samples']:raise ValueError('声纹已满32段，请先删除不合适的样本')
    minimum=(c.get('short_collection') or {}).get('target_speech_s',1.5)
    parts=[];proof=[]
    for row in sources:
        # Another familiar person's journal is not silently relabelled by a click.
        # Unassigned 'other' may be labelled only by the explicit administrator.
        if row['person'] not in (person,'other'):raise ValueError('该原话属于另一位熟人，未改写或收入候选')
        pcm=read_pcm(row['path']);x=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
        # Preserve originals; the derived reference records every selected range.
        energy=np.sqrt(np.mean(x[:len(x)//320*320].reshape(-1,320)**2,axis=1))
        voiced=np.flatnonzero(energy>.01)
        if not len(voiced):raise ValueError('所选录音没有足够清晰的发声')
        start=max(0,int(voiced[0])*320-1600);end=min(len(x),(int(voiced[-1])+1)*320+1600)
        if end-start>192000:raise ValueError('单条有效录音超过12秒，请选择较短的其他原话')
        part=x[start:end];statistics={}
        _,reason=audio_quality(np.rint(part*32768).astype('<i2').tobytes(),b'',.2,statistics)
        if reason!='accepted':raise ValueError('录音质量未通过：'+reason)
        parts.append(part);proof.append(dict(turn_id=row['id'],session=row['session'],created=row['created'],
            start_sample=start,end_sample=end,source_sha256=hashlib.sha256(pcm).hexdigest()))
    hashes={f['source_sha256'] for f in proof}
    if len(hashes)!=len(proof):raise ValueError('所选包含完全重复的录音，请选择独立原话')
    if any(hashes&{f.get('source_sha256') for f in row.get('source_fragments',[])}
            for refs in document['people'].values() for row in refs):
        raise ValueError('该音频内容已用于参考，不能更换话轮编号重复入库')
    samples=np.concatenate(parts)
    if len(samples)>192000:raise ValueError('合并录音超过12秒，请减少所选短句')
    _,reason=audio_quality(np.rint(samples*32768).astype('<i2').tobytes(),b'',minimum)
    if reason!='accepted':raise ValueError('请再选几句清晰短话，有效发声需至少'+str(minimum)+'秒：'+reason)
    observer=SpeakerObserver(c,root,lambda e:None,lambda *a,**kw:None,extractor=extractor)
    try:
        if len(parts)>1 or len(samples)>=80000:
            similarity,_=observer.check_consistency(samples,independent=True)
            if similarity is None:raise ValueError('独立片段的有效发声不足，请选择其他录音或补充短句')
            if similarity<CONSISTENCY_THRESHOLD:
                raise ValueError(f'分段一致性{similarity:.4f}低于{CONSISTENCY_THRESHOLD:.4f}，未收入候选；这不等于确认混说，请改选录音')
        vector=observer.feature(samples);ranked=[]
        for pid,refs in observer.gallery.items():
            refs=[r for r in refs if r.get('active',True)]
            if len(refs)>=c['enrollment_samples']:
                ranked.append((float(normalized(np.mean([r['vector'] for r in refs],axis=0))@vector),pid))
        ranked.sort(reverse=True)
        if ranked:
            score,pid=ranked[0];margin=score-ranked[1][0] if len(ranked)>1 else score
            if pid!=person and score>=c['threshold'] and margin>=c['minimum_margin']:
                raise ValueError('声音明确匹配到另一人，未收入此人候选；请重新核对')
        # Inference is outside the short revision-checked file commit.
        observer.refresh_gallery()
        if observer.gallery_revision!=revision:raise RevisionConflict('处理期间声纹样本已变化，请刷新后重试')
        key='review_'+hashlib.sha256(('\n'.join([person,*source_ids])).encode()).hexdigest()[:32]
        metadata=dict(origin='manual_review' if confirmed else 'onsite_label_pending_review',
            administrator_verified=bool(confirmed),candidate_reason='awaiting_manual_activation',
            source_turn_ids=source_ids,source_fragments=proof)
        if not observer._store_reference(person,samples,vector,key,sources[0]['session'],metadata,active=False):
            raise ValueError('候选未保存，可能满额或来源已使用；请刷新核对')
        return dict(ok=True,sample_id=key,active=False,already_saved=False,
            message='声纹候选已保存，尚未参与识别；试听核对后点击“激活并锁定”。')
    finally:observer.job.close()


def import_separately(root,person,ids,revision,*,confirmed=False,extractor=None):
    """One selected original produces at most one candidate; failures stay local."""
    # Validate the entire source list before starting any writes. Independent
    # recordings may span visits; merging across visits remains prohibited.
    round_sources(root,ids,same_visit=False)
    results=[];saved=0;existing=0
    for rid in ids:
        try:
            result=import_rounds(root,person,[rid],revision,confirmed=confirmed,extractor=extractor)
            results.append(dict(source_turn_id=rid,**result))
            if result.get('already_saved'):existing+=1
            else:saved+=1;revision+=1
        except RevisionConflict:
            # Do not silently adopt an unrelated administrator's new revision.
            for remaining in ids[len(results):]:results.append(dict(source_turn_id=remaining,ok=False,error='样本清单已变化，请刷新后重试；已保存部分保留。'))
            break
        except Exception as exc:
            error=str(exc) if isinstance(exc,(ValueError,OSError,sqlite3.Error)) else '声纹检查未完成：'+type(exc).__name__
            results.append(dict(source_turn_id=rid,ok=False,error=error))
    failed=sum(not r['ok'] for r in results)
    message=f'逐条选样完成：新保存{saved}条候选，已有{existing}条未重复，{failed}条未保存。'
    if saved:message+='新候选尚未激活，请试听核对后激活并锁定。'
    return dict(ok=bool(saved or existing),partial=bool(failed and (saved or existing)),
        saved_count=saved,existing_count=existing,failed_count=failed,items=results,
        message=message,error=message if failed and not(saved or existing) else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true',help='只准备未激活候选，不能宣称管理员已试听')
    a=p.parse_args()
    try:
        request=json.loads(sys.stdin.read(8193))
        if set(request)-{'person','round_ids','revision','confirmed','mode'}:raise ValueError('导入字段不正确')
        mode=request.get('mode','merge')
        if mode=='separate' and not a.prepare:
            result=import_separately(a.root,request['person'],request['round_ids'],request['revision'],confirmed=request.get('confirmed') is True)
        elif mode=='merge':
            result=import_rounds(a.root,request['person'],request['round_ids'],request['revision'],
                confirmed=request.get('confirmed') is True,prepare=a.prepare)
        else:raise ValueError('无效选样模式')
    except Exception as exc:result=dict(ok=False,error=str(exc) if isinstance(exc,(ValueError,OSError,sqlite3.Error)) else '声纹处理失败：'+type(exc).__name__)
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['ok'] else 1


if __name__=='__main__':raise SystemExit(main())
