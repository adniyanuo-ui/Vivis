#!/usr/bin/env python3
"""Read-only, source-disjoint speaker A/B replay. No devices, cloud or enrollment."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT
from vivis.speech.speaker_gallery import read_gallery
from vivis.speech.speaker_import import private_path,read_pcm
from vivis.speech.speaker_observation import SpeakerObserver


def evaluate(root,cases,people,*,extractor=None):
    """Every gallery version gets the same independent held-out cases."""
    root=Path(root);config=json.loads((root/'configs/perception/speaker_observation.json').read_text())
    with tempfile.TemporaryDirectory(prefix='vivis-speaker-ab-') as temporary:
        isolated=Path(temporary);c=dict(config,model=str(root/config['model']),short_collection={'enabled':False},short_recognition={'enabled':False})
        observer=SpeakerObserver(c,isolated,lambda e:None,lambda *a,**kw:None,extractor=extractor)
        observer.gallery=copy.deepcopy(people);observer.refresh_gallery=lambda:None
        rows=[]
        try:
            for case in cases:
                pcm=read_pcm(private_path(root,case['path']));started=time.monotonic()
                observer.infer(pcm,b'',dict(turn_id=case['source_turn_id'],session='offline',item_id=case['source_turn_id'],
                    logical_person='other',visual_person=None,enrollment_allowed=False))
                result=observer.results.get()
                rows.append(dict(source_turn_id=case['source_turn_id'],expected=case['person'],
                    confirmed=result['speaker_candidate'],best_candidate=result.get('best_candidate'),
                    score=result.get('score'),reason=result.get('recognition_reason'),quality=result.get('reason'),
                    elapsed_ms=round((time.monotonic()-started)*1000,2)))
        finally:observer.job.close()
    known=[r for r in rows if r['expected']!='other'];unknown=[r for r in rows if r['expected']=='other']
    correct=sum(r['confirmed']==r['expected'] for r in known)
    wrong=sum(bool(r['confirmed']) and r['confirmed']!=r['expected'] for r in rows)
    return dict(cases=rows,summary=dict(total=len(rows),known=len(known),stranger=len(unknown),
        correct_confirmed=correct,wrong_confirmed=wrong,confirmation_coverage=correct/len(known) if known else None,
        stranger_false_accept_rate=sum(bool(r['confirmed']) for r in unknown)/len(unknown) if unknown else None))


def independent_cases(cases,people):
    used={t for refs in people.values() for row in refs for t in row.get('source_turn_ids',[row['turn_id']])}
    return [c for c in cases if c['source_turn_id'] not in used]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--simulate-activation',help='person:sample; changes an isolated copy only')
    args=p.parse_args();manifest=private_path(ROOT,args.manifest)
    document=json.loads(manifest.read_text());cases=document['cases']
    if not isinstance(cases,list) or not 1<=len(cases)<=500:raise ValueError('评估需1..500条有人工标签的独立录音')
    if len({c['source_turn_id'] for c in cases})!=len(cases):raise ValueError('测试原话重复')
    c=json.loads((ROOT/'configs/perception/speaker_observation.json').read_text())
    gallery=read_gallery(private_path(ROOT,ROOT/c['directory'])/'gallery.json',c['sha256'])['people']
    after=copy.deepcopy(gallery)
    if args.simulate_activation:
        person,sample=args.simulate_activation.split(':',1)
        row=next((r for r in after.get(person,[]) if r['turn_id']==sample),None)
        if row is None:raise ValueError('模拟候选不存在')
        row['active']=True
    held_out=independent_cases(cases,after)
    report=dict(label_source=document.get('label_source'),simulated_only=bool(args.simulate_activation),
        excluded_reference_sources=len(cases)-len(held_out),thresholds=dict(image=.60,voice=c['threshold'],margin=c['minimum_margin']),
        before=evaluate(ROOT,held_out,gallery),after=evaluate(ROOT,held_out,after),
        limitations='按操作者标签离线单句重放；不是现场准确率或融合身份验收。无陌生人对照时不能估计陌生人误认率。')
    output=private_path(ROOT,args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(dict(output=str(output),before=report['before']['summary'],after=report['after']['summary']),ensure_ascii=False))


if __name__=='__main__':main()
