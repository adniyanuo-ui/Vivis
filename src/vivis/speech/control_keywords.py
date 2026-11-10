"""Small local stop recognizer, independent of sentence review and the network."""
import math
import re
import json

import numpy as np

from vivis.paths import ROOT
from vivis.interaction.intents import normalize,IntentRouter


WORDS=('停止','退出','结束','待机')


def blocked(text):
    text=normalize(text)
    return bool(re.search(r'(?:不要|不用|不想|不能|别|不是)(?:再)?(?:停|退出|退|结束|待机)',text)
        or re.search(r'(?:停止|退出|结束|待机).{0,4}(?:什么|怎么|意思|吗)',text)
        or re.search(r'(?:怎么|如何|为何|为什么|什么叫).{0,6}(?:停止|退出|结束|待机)',text)
        or text.startswith(('如果','假如','要是','他说','她说','台词')))


class KeywordControl:
    def __init__(self,settings,emit,engine=None,catalog=None,wake_words=None):
        self.c=settings;self.emit=emit
        self.router=IntentRouter(catalog if catalog is not None else json.loads((ROOT/'configs/interaction/programs.json').read_text()),
                                 wake_words=wake_words)
        for name,low,high in [('confirmation_s',.1,.5),('reference_guard_s',.2,1.5),('refractory_s',.5,3.)]:
            value=settings[name]
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError('Invalid local control timing: '+name)
        if engine is None:
            import sherpa_onnx
            folder=ROOT/settings['directory']
            engine=sherpa_onnx.KeywordSpotter(**{k:str(folder/settings[k]) for k in ('encoder','decoder','joiner','tokens')},
                keywords_file=str(ROOT/settings['keywords_file']),num_threads=settings['threads'],
                keywords_score=settings['keywords_score'],keywords_threshold=settings['keywords_threshold'])
        self.engine=engine;self.near=engine.create_stream();self.far=engine.create_stream()
        self.pending=None;self.last_stamp=None;self.latest_text=('',-math.inf)
        self.reference={};self.last_emitted=-math.inf;self.sequence=0

    def feed(self,pcm,reference,stamp,text=''):
        if len(pcm)!=320 or reference is not None and len(reference)!=320:raise ValueError('Control requires20ms/16k frames')
        now=stamp+.02
        if self.last_stamp is not None and not 0<stamp-self.last_stamp<.1:
            self.near=self.engine.create_stream();self.far=self.engine.create_stream();self.pending=None;self.reference.clear()
        self.last_stamp=stamp
        self.latest_text=(text,now)
        self.far.accept_waveform(16000,np.zeros(320,np.float32) if reference is None else reference)
        while self.engine.is_ready(self.far):
            self.engine.decode_stream(self.far);word=self.engine.get_result(self.far)
            if word:
                self.reference[word]=now;self.engine.reset_stream(self.far)
        self.near.accept_waveform(16000,pcm)
        while self.engine.is_ready(self.near):
            self.engine.decode_stream(self.near);word=self.engine.get_result(self.near)
            if word:
                if word in WORDS and self.pending is None and now-self.last_emitted>=self.c['refractory_s']:
                    self.pending=(word,now,now+self.c['confirmation_s'])
                self.engine.reset_stream(self.near)
        if self.pending is None or now<self.pending[2]:return
        word,detected,_=self.pending;self.pending=None
        text,text_at=self.latest_text
        reason=('playback_reference' if abs(detected-self.reference.get(word,-math.inf))<=self.c['reference_guard_s']
                else 'negation_or_question' if 0<=now-text_at<=1. and blocked(text) else None)
        # Acoustic keyword similarity alone confused 解决 with 结束 and even
        # produced 停止 in near-silence onsite. Only accelerate a COMPLETE local
        # streaming control hypothesis; never turn an unconfirmed hit into a
        # cancellation. The primary final path remains independent of this gate.
        expected='standby' if word=='待机' else 'interaction'
        if reason is None and (not 0<=now-text_at<=1. or self.router.parse(text,state='task')['kind']!=expected):
            reason='unconfirmed_streaming_intent'
        if reason:
            self.emit(dict(kind='asr_control_suppressed',text=word,reason=reason,
                           streaming_text=text,captured_end_monotonic=now));return
        self.sequence+=1;self.last_emitted=now
        self.emit(dict(kind='asr_control',control_sequence=self.sequence,text=word,
            captured_end_monotonic=now,keyword_detected_monotonic=detected,
            streaming_text=text if 0<=now-text_at<=1. else '',
            source='local_keyword_control',recognizer='keyword_spotter'))
