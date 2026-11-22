"""Local speaker candidates only. No identity/permission/business mutation API."""
import hashlib
import json
import os
from pathlib import Path
import queue
import time
import wave

import numpy as np

from vivis.runtime.best_effort import BestEffortJob


def normalized(value):
    value=np.asarray(value,np.float32).reshape(-1);norm=float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm<1e-6:raise ValueError('Invalid speaker feature')
    return value/norm


def audio_quality(pcm,far,minimum_seconds=2.5):
    x=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
    ref=np.frombuffer(far,dtype='<i2').astype(np.float32)/32768
    if len(x)<16000*minimum_seconds:return None,'too_short'
    if np.mean(np.abs(x)>.98)>.005:return None,'clipping'
    rms=float(np.sqrt(np.mean(x*x)))
    if rms<.008:return None,'quiet'
    energy=np.sqrt(np.mean(x[:len(x)//320*320].reshape(-1,320)**2,axis=1))
    if np.count_nonzero(energy>.01)*.02<minimum_seconds:return None,'insufficient_speech'
    # Enrollment/observation during strong robot playback is deliberately skipped
    # even with AEC: one visible face does not prove whose voice remains.
    if len(ref) and float(np.sqrt(np.mean(ref*ref)))>.004:return None,'robot_playback'
    return x[-192000:],'accepted'


class SpeakerObserver:
    @classmethod
    def from_config(cls,path,root,emit,log):
        if not path:return None
        c=json.loads((Path(root)/path).read_text())
        if c.get('schema_version')!='vivis-speaker-observation-1.0' or c.get('observation_only') is not True:
            raise ValueError('Speaker recognition must remain observation only')
        if not c.get('enabled'):return None
        return cls(c,root,emit,log)

    def __init__(self,c,root,emit,log,extractor=None):
        if (c.get('observation_only') is not True or not .3<=c.get('threshold',0)<=.95
                or not .01<=c.get('minimum_margin',0)<=.5
                or not 3<=c.get('enrollment_samples',0)<=c.get('maximum_samples',0)<=16
                or not 2<=c.get('minimum_seconds',0)<=12):raise ValueError('Invalid speaker observation policy')
        self.c=c;self.root=Path(root);self.emit=emit;self.log=log;self.results=queue.SimpleQueue();self.job=BestEffortJob('speaker-observation');self.labels={}
        self.directory=self.root/c['directory'];self.model=None;self.error=None
        if not self.directory.resolve().is_relative_to((self.root/'private').resolve()):raise ValueError('Private speaker directory required')
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.extractor=extractor
        self.gallery_path=self.directory/'gallery.json'
        self.gallery={}
        if self.gallery_path.exists():
            data=json.loads(self.gallery_path.read_text())
            if data.get('model_sha256')!=c['sha256'] or not isinstance(data.get('people'),dict):raise ValueError('Speaker gallery/model mismatch')
            from vivis.interaction.memory import person_key
            for pid,rows in data['people'].items():
                person_key(pid)
                if not isinstance(rows,list) or len(rows)>c['maximum_samples']:raise ValueError('Invalid speaker reference count')
                for row in rows:normalized(row['vector'])
            self.gallery=data['people']

    def feature(self,samples):
        if self.extractor:return normalized(self.extractor(samples))
        if self.model is None:
            import sherpa_onnx
            path=self.root/self.c['model']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.c['sha256']:raise ValueError('Speaker model hash mismatch')
            self.model=sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path),num_threads=1,provider='cpu'))
        stream=self.model.create_stream();stream.accept_waveform(16000,samples);stream.input_finished()
        if not self.model.is_ready(stream):raise ValueError('Speaker sample not ready')
        return normalized(self.model.compute(stream))

    def offer(self,clip):
        # Snapshot all fields; a late result never reads the current face/person.
        pcm=b''.join(clip['near']);far=b''.join(clip['far'])
        source=dict(turn_id=clip['rid'],session=clip['session'],item_id=clip['item'],
            logical_person=clip['person'],visual_person=(clip.get('evidence') or {}).get('person_id'),
            enrollment_allowed=bool((clip.get('evidence') or {}).get('confirmed') and clip['alignment']=='aligned'))
        return self.job.offer(self.infer,pcm,far,source)

    def infer(self,pcm,far,source):
        result=dict(**source,speaker_candidate=None,observation_only=True,score=None,reason='unavailable')
        try:
            samples,reason=audio_quality(pcm,far,self.c['minimum_seconds']);result['reason']=reason
            if samples is None:return
            vector=self.feature(samples)
            # Independent beginning/end windows must sound like the same speaker.
            if len(samples)>=80000:
                if float(self.feature(samples[:40000])@self.feature(samples[-40000:]))<.6:
                    result['reason']='mixed_or_unstable_speech';return
            ranked=[]
            for pid,rows in self.gallery.items():
                if len(rows)>=self.c['enrollment_samples']:
                    center=normalized(np.mean([r['vector'] for r in rows],axis=0));ranked.append((float(center@vector),pid))
            ranked.sort(reverse=True)
            if ranked:
                score,pid=ranked[0];margin=score-ranked[1][0] if len(ranked)>1 else score
                result.update(score=score,margin=margin)
                if score>=self.c['threshold'] and margin>=self.c['minimum_margin']:result['speaker_candidate']=pid
            person=source['visual_person']
            if source['enrollment_allowed'] and person and person==source['logical_person']:
                rows=self.gallery.get(person,[])
                if rows and float(normalized(np.mean([r['vector'] for r in rows],axis=0))@vector)<.65:
                    result['reason']='face_voice_conflict';return
                if result['speaker_candidate'] not in (None,person):
                    result['reason']='known_other_voice';return
                if not any(row['turn_id']==source['turn_id'] for row in rows):
                    folder=self.directory/person;folder.mkdir(exist_ok=True,mode=0o700)
                    path=folder/(source['turn_id']+'.wav')
                    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
                    with os.fdopen(fd,'wb') as stream:
                        with wave.open(stream,'wb') as w:
                            w.setparams((1,2,16000,0,'NONE','not compressed'))
                            w.writeframes(np.rint(samples*32767).astype('<i2').tobytes())
                    previous=list(rows)
                    rows=[*rows,dict(turn_id=source['turn_id'],session=source['session'],created=time.time(),vector=vector.tolist())][-self.c['maximum_samples']:]
                    self.gallery[person]=rows
                    tmp=self.gallery_path.with_suffix('.tmp');fd=os.open(tmp,os.O_CREAT|os.O_WRONLY|os.O_TRUNC,0o600)
                    with os.fdopen(fd,'w') as f:json.dump(dict(model_sha256=self.c['sha256'],people=self.gallery),f)
                    tmp.replace(self.gallery_path)
                    retained={r['turn_id'] for r in rows}
                    for old in previous:
                        if old['turn_id'] not in retained:(folder/(old['turn_id']+'.wav')).unlink(missing_ok=True)
                    result['enrollment_samples']=len(rows)
                    if len(rows)<self.c['enrollment_samples']:result['reason']='collecting_reference_samples'
        except Exception as exc:result['reason']='speaker_unavailable:'+type(exc).__name__
        finally:self.results.put(result)

    def poll(self):
        while True:
            try:result=self.results.get_nowait()
            except queue.Empty:return
            self.log('speaker_observation',**result)
            def name(pid):
                if not pid or str(pid).startswith('anonymous:'):return 'other'
                return self.labels.get(pid,'other')
            label=name(result['logical_person'])+'-'+name(result['visual_person'])+'/图-'+name(result['speaker_candidate'])+'/声'
            self.emit(dict(kind='voice_warning',message='声纹观察［原话轮'+result['turn_id'][:8]+'］：'+label+'；'+result['reason'],speaker_observation=result))
