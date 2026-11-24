"""A bounded, disposable recognition window; never enrolls its own guesses."""
import math
import numpy as np


class ShortRecognitionWindow:
    def __init__(self,config):
        self.c=config;self.batch=None;self.used=set();self.order=[]
        for key,lo,hi in [('minimum_fragment_s',.2,.6),('target_speech_s',1.2,3),
                ('maximum_window_s',3,15),('maximum_gap_s',1,8),('maximum_audio_s',3,8)]:
            value=config.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError('Invalid short recognition window: '+key)
        maximum=config.get('maximum_speech_s',3.)
        if type(maximum) not in (int,float) or not math.isfinite(maximum) or not config['target_speech_s']<=maximum<=config['maximum_audio_s']:
            raise ValueError('Invalid maximum recognition speech duration')
        self.maximum_speech_s=maximum

    def clear(self):self.batch=None

    def defer(self):
        if self.batch:
            if self.batch['effective_s']>=self.maximum_speech_s:self.clear()
            else:self.batch['target']=min(self.maximum_speech_s,self.batch['effective_s']+.5)

    def add(self,pcm,far,source,result,current_epoch):
        from vivis.speech.speaker_observation import audio_quality
        turn=source['turn_id'];start=source.get('audio_started');end=source.get('audio_ended')
        epoch=source.get('sample_epoch');context=(source['session'],source.get('voice_context'),epoch)
        if (source.get('_sampling_conflict') or source.get('sampling_conflict') or epoch!=current_epoch
                or result.get('reason') in ('clipping','mixed_or_unstable_speech','robot_playback')):
            self.clear();return None
        if turn in self.used:return None
        self.used.add(turn);self.order.append(turn)
        while len(self.order)>256:self.used.discard(self.order.pop(0))
        if (type(start) not in (float,int) or type(end) not in (float,int)
                or not math.isfinite(start) or not math.isfinite(end) or end<start or len(pcm)!=len(far)):
            self.clear();return None
        statistics={};samples,_=audio_quality(pcm,far,self.c['minimum_fragment_s'],statistics)
        if samples is None:return None
        # Trim only this derived short-window copy. The main query and stored
        # user recording retain their original waveform and boundaries.
        rms=np.sqrt(np.mean(samples[:len(samples)//320*320].reshape(-1,320)**2,axis=1))
        voiced=np.flatnonzero(rms>.01)
        if not len(voiced):return None
        first=max(0,int(voiced[0])*320-1600);last=min(len(samples),(int(voiced[-1])+1)*320+1600)
        samples=samples[first:last].copy()
        evidence=result.get('identity_evidence',{});candidate=evidence.get('person_id')
        if (self.batch and (self.batch['context']!=context or start<self.batch['end']
                or start-self.batch['end']>self.c['maximum_gap_s']
                or end-self.batch['start']>self.c['maximum_window_s']
                or candidate and self.batch['candidate'] and candidate!=self.batch['candidate'])):self.clear()
        if self.batch is None:
            self.batch=dict(context=context,start=start,end=end,fragments=[],turns=[],effective_s=0.,samples=0,candidate=candidate,
                target=self.c['target_speech_s'])
        b=self.batch
        if b['samples']+len(samples)>self.c['maximum_audio_s']*16000:self.clear();return None
        b['fragments'].append(samples);b['turns'].append(turn);b['end']=end
        b['effective_s']+=statistics['effective_speech_s'];b['samples']+=len(samples)
        if candidate:b['candidate']=candidate
        if b['effective_s']<b['target']:return None
        return dict(samples=np.concatenate(b['fragments']),latest_samples=b['fragments'][-1],source_turn_ids=list(b['turns']),
            effective_speech_s=b['effective_s'],audio_started=b['start'],audio_ended=b['end'])
