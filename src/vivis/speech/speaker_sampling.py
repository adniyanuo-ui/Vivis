"""Bounded short-utterance reference batches with independent visual provenance."""
import math
import time
import uuid

import numpy as np


def visual_ranges(points,start,end,witness_s):
    """A confirmed frame labels at most the following short interval, never8s hold."""
    selected=sorted((p for p in points if start-witness_s<=p['stamp']<=end),key=lambda p:p['stamp'])
    identities={p['person_id'] for p in selected if p.get('confirmed')}
    if len(identities)>1 or any(p.get('blocked') for p in selected):return [],True
    ranges=[]
    for i,p in enumerate(selected):
        if not p.get('confirmed'):continue
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
                ('maximum_window_s',5.,60.),('maximum_audio_s',4.,12.),('visual_witness_s',.1,1.),('padding_ms',0,200)]:
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
        from vivis.speech.speaker_observation import audio_quality
        x=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
        ref=np.frombuffer(far,dtype='<i2').astype(np.float32)/32768
        missing=0
        if len(ref)!=len(x):
            if not reference_end_aligned:return [],'reference_alignment_unavailable'
            missing=max(0,len(x)-len(ref))
            # The recorder explicitly aligns reference to the server endpoint.
            # Missing prefix is excluded, never invented as silent reference.
            ref=np.r_[np.zeros(missing,np.float32),ref][-len(x):]
        result=[];pad=round(self.c['padding_ms']*16)
        for span in ranges:
            left=max(missing,0,round(span['start_s']*16000));right=min(len(x),round(span['end_s']*16000))
            n=(right-left)//320
            if n<=0:continue
            power=np.sqrt(np.mean(ref[left:left+n*320].reshape(-1,320)**2,axis=1))
            bad=power>.004
            for offset in range(1,6):bad[offset:]|=(power[:-offset]>.004);bad[:-offset]|=(power[offset:]>.004)
            allowed=~bad;edges=np.diff(np.r_[False,allowed,False].astype(int))
            for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                i=left+int(a)*320;j=left+int(b)*320
                reference_pcm=np.rint(ref[i:j]*32768).astype('<i2').tobytes()
                stats={};samples,reason=audio_quality(pcm[i*2:j*2],reference_pcm,self.c['minimum_fragment_s'],stats)
                if samples is None:continue
                energy=np.sqrt(np.mean(x[i:j].reshape(-1,320)**2,axis=1));v=np.flatnonzero(energy>.01)
                if not len(v):continue
                begin=max(i,i+int(v[0])*320-pad);end=min(j,i+(int(v[-1])+1)*320+pad)
                result.append(dict(samples=x[begin:end].copy(),effective_s=stats['effective_speech_s'],
                    start_sample=begin,end_sample=end,visual_stamp=span.get('visual_stamp'),
                    visual_stamps=span.get('visual_stamps',[]),proof=span.get('proof','confirmed_frame')))
        return result,None

    def accept(self,pcm,far,source,query,current_epoch,maximum,existing_ids,now=None):
        now=time.monotonic() if now is None else now
        self.expire(now,current_epoch);self.remember(existing_ids)
        rid=source['turn_id'];spans=source.get('_sampling_ranges',[])
        if rid in self.used or rid in existing_ids:return None,dict(status='already_used')
        if source.get('_sampling_conflict'):
            self.clear();self.remember([rid]);return None,dict(status='visual_conflict')
        if query.get('reason') in ('clipping','mixed_or_unstable_speech'):
            self.clear();self.remember([rid]);return None,dict(status='unreliable_voice')
        people={s['person_id'] for s in spans};epochs={s['epoch'] for s in spans}
        if len(people)!=1 or epochs!={current_epoch}:return None,dict(status='no_reliable_visual_fragment')
        person=next(iter(people))
        if query.get('speaker_candidate') not in (None,person):
            self.clear();self.remember([rid]);return None,dict(status='face_voice_conflict',person_id=person)
        if maximum<=0:
            self.clear();self.remember([rid]);return None,dict(status='pool_full',person_id=person)
        fragments,error=self.fragments(pcm,far,spans,source.get('_reference_end_aligned',False))
        if not fragments:return None,dict(status=error or 'no_clean_fragment',person_id=person)
        context=source['session'].split(':',1)[0]
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
            self.clear();return None,dict(status='audio_capacity_reached',person_id=person)
        if rid not in b['source_turns']:b['source_turns'].append(rid);b['source_sessions'].append(source['session'])
        if b['effective_s']<b['target']:
            return None,dict(status='accumulating',person_id=person,effective_s=b['effective_s'],target_s=b['target'],turns=len(b['source_turns']))
        samples=np.concatenate([f['samples'] for f in b['fragments']])
        metadata=dict(sampling_method='short-visual-batch-1',source_turn_ids=b['source_turns'],
            source_sessions=b['source_sessions'],effective_speech_s=b['effective_s'],
            source_fragments=[{k:v for k,v in f.items() if k!='samples'} for f in b['fragments']])
        return dict(person_id=person,samples=samples,turn_id=b['turn_id'],session=context,metadata=metadata),None
