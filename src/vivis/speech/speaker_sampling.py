"""Bounded short-utterance reference batches with independent visual provenance."""
import math
import time
import uuid

import numpy as np


def visual_ranges(points,start,end,witness_s):
    """A confirmed frame labels at most the following short interval, never8s hold."""
    selected=sorted((p for p in points if start-witness_s<=p['stamp']<=end),key=lambda p:p['stamp'])
    identities={p['person_id'] for p in selected if p.get('confirmed') and not p.get('blocked')}
    if len(identities)>1:return [],True
    ranges=[]
    for i,p in enumerate(selected):
        if not p.get('confirmed') or p.get('blocked'):continue
        a=max(start,p['stamp']);b=min(end,p['stamp']+witness_s)
        if i+1<len(selected):b=min(b,selected[i+1]['stamp'])
        if b<=a:continue
        if (ranges and ranges[-1]['person_id']==p['person_id'] and ranges[-1]['epoch']==p['epoch']
                and abs(ranges[-1]['end_s']-(a-start))<1e-6):
            ranges[-1]['end_s']=b-start;ranges[-1]['visual_stamps'].append(p['stamp'])
        else:ranges.append(dict(start_s=a-start,end_s=b-start,person_id=p['person_id'],
            epoch=p['epoch'],visual_stamp=p['stamp'],visual_stamps=[p['stamp']]))
    return ranges,False


class ShortSpeechCollector:
    def __init__(self,config):
        self.c=config
        for key,lo,hi in [('minimum_fragment_s',.1,.6),('target_speech_s',1.2,4.),
                ('maximum_window_s',5.,120.),('maximum_audio_s',4.,12.),('visual_witness_s',.1,1.),('padding_ms',0,200)]:
            v=config.get(key)
            if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('Invalid short speech collection: '+key)
        maximum=config.get('maximum_speech_s',3.)
        if type(maximum) not in (int,float) or not math.isfinite(maximum) or not config['target_speech_s']<=maximum<=6:raise ValueError('Invalid maximum speech accumulation')
        self.maximum_speech_s=maximum
        self.batch=None;self.used=set();self.used_order=[];self.last_notice=None

    def remember(self,ids):
        for item in ids:
            if item not in self.used:self.used.add(item);self.used_order.append(item)
        while len(self.used_order)>1024:self.used.discard(self.used_order.pop(0))

    def clear(self):self.batch=None

    def reject_latest(self):
        """A bad newest contribution does not destroy earlier witnessed speech."""
        if not self.batch:return
        b=self.batch;latest=b['source_turns'][-1]
        b['fragments']=[f for f in b['fragments'] if f['turn_id']!=latest]
        b['source_turns'].pop();b['source_sessions'].pop()
        if not b['fragments']:self.clear();return
        b['samples']=sum(len(f['samples']) for f in b['fragments'])
        b['effective_s']=sum(f['effective_s'] for f in b['fragments'])
        b['target']=min(self.maximum_speech_s,max(self.c['target_speech_s'],b['effective_s']+.2))

    def defer(self):
        if self.batch:
            if self.batch['effective_s']>=self.maximum_speech_s:self.clear()
            else:self.batch['target']=min(self.maximum_speech_s,self.batch['effective_s']+.5)

    def expire(self,now,epoch):
        if self.batch and (now-self.batch['created']>self.c['maximum_window_s'] or self.batch['epoch']!=epoch):
            self.clear();return True
        return False

    def fragments(self,pcm,far,ranges,reference_end_aligned=False):
        """Exclude robot-output frames/tails; retain source ranges and speech context."""
        from vivis.speech.speaker_segments import clean_fragments
        return clean_fragments(pcm,far,ranges,minimum_fragment_s=self.c['minimum_fragment_s'],
            padding_ms=self.c['padding_ms'],reference_end_aligned=reference_end_aligned)

    def accept(self,pcm,far,source,query,current_epoch,maximum,existing_ids,now=None):
        now=time.monotonic() if now is None else now
        self.expire(now,current_epoch);self.remember(existing_ids)
        rid=source['turn_id'];spans=source.get('_sampling_ranges',[])
        if rid in self.used or rid in existing_ids:return None,dict(status='already_used')
        if source.get('_sampling_conflict'):
            self.remember([rid]);return None,dict(status='visual_conflict')
        if query.get('reason') in ('clipping','mixed_or_unstable_speech'):
            self.remember([rid]);return None,dict(status='unreliable_voice')
        people={s['person_id'] for s in spans};epochs={s['epoch'] for s in spans}
        if len(people)!=1 or epochs!={current_epoch}:return None,dict(status='no_reliable_visual_fragment')
        person=next(iter(people))
        if query.get('speaker_candidate') not in (None,person):
            self.remember([rid]);return None,dict(status='face_voice_conflict',person_id=person)
        if maximum<=0:
            self.clear();self.remember([rid]);return None,dict(status='pool_full',person_id=person)
        fragments,error=self.fragments(pcm,far,spans,source.get('_reference_end_aligned',False))
        if not fragments:return None,dict(status=error or 'no_clean_fragment',person_id=person)
        context=(source['session'].split(':',1)[0],source.get('voice_context'))
        if self.batch and (self.batch['person_id']!=person or self.batch['context']!=context):self.clear()
        if self.batch is None:self.batch=dict(person_id=person,epoch=current_epoch,context=context,created=now,
            fragments=[],source_turns=[],source_sessions=[],effective_s=0.,samples=0,
            target=self.c['target_speech_s'],turn_id='agg_'+uuid.uuid4().hex)
        b=self.batch;self.remember([rid]);added=False
        for f in fragments:
            if b['samples']+len(f['samples'])>16000*self.c['maximum_audio_s']:break
            b['fragments'].append(dict(f,turn_id=rid,item_id=source['item_id'],session=source['session'],
                source_start_monotonic=source.get('_source_start')))
            b['samples']+=len(f['samples']);b['effective_s']+=f['effective_s']
            added=True
        if not added:
            return None,dict(status='audio_capacity_reached',person_id=person)
        if rid not in b['source_turns']:b['source_turns'].append(rid);b['source_sessions'].append(source['session'])
        if b['effective_s']<b['target']:
            return None,dict(status='accumulating',person_id=person,effective_s=b['effective_s'],target_s=b['target'],turns=len(b['source_turns']))
        samples=np.concatenate([f['samples'] for f in b['fragments']])
        metadata=dict(sampling_method='short-visual-batch-1',source_turn_ids=b['source_turns'],
            source_sessions=b['source_sessions'],effective_speech_s=b['effective_s'],
            source_fragments=[{k:v for k,v in f.items() if k!='samples'} for f in b['fragments']])
        return dict(person_id=person,samples=samples,turn_id=b['turn_id'],session=context[0],metadata=metadata),None
