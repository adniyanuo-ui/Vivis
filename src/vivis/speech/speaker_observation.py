"""Local speaker candidates only. No identity/permission/business mutation API."""
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import stat
import time
import wave

import numpy as np

from vivis.runtime.best_effort import BestEffortJob

CONSISTENCY_THRESHOLD=.6


def normalized(value):
    value=np.asarray(value,np.float32).reshape(-1);norm=float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm<1e-6:raise ValueError('Invalid speaker feature')
    return value/norm


def audio_quality(pcm,far,minimum_seconds=0.6,statistics=None):
    x=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
    ref=np.frombuffer(far,dtype='<i2').astype(np.float32)/32768
    if statistics is not None:statistics.update(duration_s=len(x)/16000,minimum_speech_s=minimum_seconds)
    if len(x)<16000*minimum_seconds:return None,'too_short'
    if np.mean(np.abs(x)>.98)>.005:return None,'clipping'
    rms=float(np.sqrt(np.mean(x*x)))
    if statistics is not None:statistics['rms']=rms
    if rms<.008:return None,'quiet'
    energy=np.sqrt(np.mean(x[:len(x)//320*320].reshape(-1,320)**2,axis=1))
    speech_s=float(np.count_nonzero(energy>.01)*.02)
    if statistics is not None:statistics['effective_speech_s']=speech_s
    if speech_s<minimum_seconds:return None,'insufficient_speech'
    # Enrollment/observation during strong robot playback is deliberately skipped
    # even with AEC: one visible face does not prove whose voice remains.
    if len(ref) and float(np.sqrt(np.mean(ref*ref)))>.004:return None,'robot_playback'
    return x[-192000:],'accepted'


def consistency_windows(samples,minimum_seconds):
    """Two non-overlapping voiced-edge windows, never two copies of one short word.

    Only the consistency probes move. The main query embedding and stored WAV
    retain their original samples and the same gallery/model contract.
    """
    energy=np.sqrt(np.mean(samples[:len(samples)//320*320].reshape(-1,320)**2,axis=1))
    voiced=np.flatnonzero(energy>.01)
    if not len(voiced):return None,dict(reason='no_voiced_frames')
    start=max(0,int(voiced[0])*320-1600);end=min(len(samples),(int(voiced[-1])+1)*320+1600)
    size=min(40000,(end-start)//2)
    if size<320:return None,dict(reason='window_too_short')
    left=samples[start:start+size];right=samples[end-size:end]
    def speech_s(x):
        rms=np.sqrt(np.mean(x[:len(x)//320*320].reshape(-1,320)**2,axis=1))
        return float(np.count_nonzero(rms>.01)*.02)
    evidence=dict(start_sample=start,end_sample=end,window_samples=size,
        left_speech_s=speech_s(left),right_speech_s=speech_s(right))
    if min(evidence['left_speech_s'],evidence['right_speech_s'])<minimum_seconds:
        return None,dict(evidence,reason='insufficient_independent_speech')
    return (left,right),evidence


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
                or not 3<=c.get('enrollment_samples',0)<=c.get('maximum_samples',0)<=32
                or not .6<=c.get('minimum_seconds',0)<=12
                or c.get('consistency_window_mode','fixed') not in ('fixed','speech_aligned')):raise ValueError('Invalid speaker observation policy')
        weak=c.get('suspect_threshold',c['threshold'])
        if type(weak) not in (int,float) or not np.isfinite(weak) or not 0<=weak<=c['threshold']:
            raise ValueError('Invalid speaker suspect threshold')
        self.c=c;self.root=Path(root);self.emit=emit;self.log=log;self.results=queue.SimpleQueue();self.job=BestEffortJob('speaker-observation');self.labels={}
        self.directory=self.root/c['directory'];self.model=None;self.error=None
        if not self.directory.resolve().is_relative_to((self.root/'private').resolve()):raise ValueError('Private speaker directory required')
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.extractor=extractor
        self.gallery_path=self.directory/'gallery.json'
        self.gallery={};self.gallery_stamp=None;self.gallery_revision=0
        self.collector=None;self.collection_epoch=0;self.collection_housekeeping=0.
        self.result_handler=None;self.recognition_window=None
        if (c.get('short_recognition') or {}).get('enabled'):
            from vivis.speech.speaker_window import ShortRecognitionWindow
            self.recognition_window=ShortRecognitionWindow(c['short_recognition'])
        if (c.get('short_collection') or {}).get('enabled'):
            from vivis.speech.speaker_sampling import ShortSpeechCollector
            self.collector=ShortSpeechCollector(c['short_collection'])
        self.refresh_gallery()

    def refresh_gallery(self):
        """Manual WAV deletions remove only those references, even while running."""
        stamp=self.gallery_path.stat().st_mtime_ns if self.gallery_path.exists() else None
        data=self.gallery
        if stamp is not None and stamp!=self.gallery_stamp:
            from vivis.speech.speaker_gallery import read_gallery
            document=read_gallery(self.gallery_path,self.c['sha256']);self.gallery_revision=document['revision']
            data=document
            if data.get('model_sha256')!=self.c['sha256'] or not isinstance(data.get('people'),dict):raise ValueError('Speaker gallery/model mismatch')
            from vivis.interaction.memory import person_key
            for pid,rows in data['people'].items():
                person_key(pid)
                if not isinstance(rows,list) or len(rows)>self.c['maximum_samples']:raise ValueError('Invalid speaker reference count')
                turns=[]
                for row in rows:
                    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}',row['turn_id']):raise ValueError('Invalid speaker sample ID')
                    turns.append(row['turn_id']);normalized(row['vector'])
                if len(set(turns))!=len(turns):raise ValueError('Duplicate speaker sample')
            data=data['people']
        kept={};changed=False
        for pid,rows in data.items():
            valid=[]
            for row in rows:
                try:exists=stat.S_ISREG((self.directory/pid/(row['turn_id']+'.wav')).stat().st_mode)
                except FileNotFoundError:exists=False
                if exists:valid.append(row)
            kept[pid]=valid;changed|=len(valid)!=len(rows)
        if changed:self.save_gallery(kept)
        else:self.gallery=data;self.gallery_stamp=stamp

    def save_gallery(self,people):
        from vivis.speech.speaker_gallery import save_gallery
        updated=save_gallery(self.gallery_path,self.c['sha256'],people,self.gallery_revision)
        self.gallery_revision=updated['revision']
        self.gallery=people
        # An administrator may commit immediately after our rename. Do not
        # label our older in-memory rows with that newer file's timestamp.
        self.gallery_stamp=None

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
        source.update(audio_started=clip.get('speaker_start',clip.get('start')),audio_ended=clip.get('end'),
            sample_epoch=clip.get('sampling_epoch'),voice_context=(clip.get('evidence') or {}).get('voice_context'),
            source_generation=(clip.get('evidence') or {}).get('source_generation'))
        source.update(source_visual_evidence=(clip.get('evidence') or {}).get('identity_evidence'),
            sampling_conflict=clip.get('sampling_conflict',False))
        if self.collector:
            source.update(_collection_managed=True,_sampling_ranges=clip.get('sampling_ranges',[]),
                _sampling_conflict=clip.get('sampling_conflict',False),_reference_end_aligned=True,
                _source_start=clip.get('speaker_start',clip.get('start')))
        return self.job.offer(self.infer,pcm,far,source)

    def set_collection_epoch(self,epoch):self.collection_epoch=epoch

    def _identity_evidence(self,result,source,turns=None):
        from vivis.interaction.identity_fusion import unavailable
        stamp=source.get('audio_ended');state='unavailable';person=None
        if type(stamp) not in (float,int) or not np.isfinite(stamp):return unavailable('no_audio_endpoint')
        decision=result.get('recognition_reason');score=result.get('score');candidate=result.get('best_candidate')
        if decision=='matched' and result.get('speaker_candidate'):
            state='confirmed';person=result['speaker_candidate']
        elif (decision in ('low_similarity','ambiguous_candidates') and candidate
                and type(score) in (int,float) and score>=self.c.get('suspect_threshold',self.c['threshold'])):
            state='suspected';person=candidate
        elif decision=='low_similarity' and candidate:state='other'
        if state=='unavailable':return unavailable(result.get('reason','no_usable_voice'))
        return dict(state=state,person_id=person,stamp=stamp,evidence_id=source['turn_id'],score=score,
            margin=result.get('margin'),source_turn_ids=turns or [source['turn_id']],reason=decision)

    def _enrich_recognition(self,pcm,far,source,result):
        result['identity_evidence']=self._identity_evidence(result,source)
        window=self.recognition_window
        if not window:return
        if result['identity_evidence']['state']=='confirmed':window.clear();return
        group=window.add(pcm,far,source,result,self.collection_epoch)
        if not group:return
        samples=group['samples'];windows,detail=consistency_windows(samples,self.c['minimum_seconds'])
        similarity=float(self.feature(windows[0])@self.feature(windows[1])) if windows else None
        result['window_match']=dict(source_turn_ids=group['source_turn_ids'],
            effective_speech_s=group['effective_speech_s'],consistency=similarity,**detail)
        if similarity is None or similarity<CONSISTENCY_THRESHOLD:window.defer();return
        self.refresh_gallery()
        vector=self.feature(samples);ranked=[]
        for pid,rows in self.gallery.items():
            active=[r for r in rows if r.get('active',True)]
            if len(active)>=self.c['enrollment_samples']:
                ranked.append((float(normalized(np.mean([r['vector'] for r in active],axis=0))@vector),pid))
        ranked.sort(reverse=True)
        if not ranked:window.clear();return
        score,pid=ranked[0];margin=score-ranked[1][0] if len(ranked)>1 else score
        latest=self.feature(group['latest_samples'])
        support=float(vector@latest)
        result['window_match']['latest_speaker_support']=support
        if support<self.c.get('suspect_threshold',self.c['threshold']):
            result['window_match']['reason']='latest_fragment_not_supported';window.clear();return
        decision=('low_similarity' if score<self.c['threshold'] else
            'ambiguous_candidates' if margin<self.c['minimum_margin'] else 'matched')
        result['single_query']={key:result.get(key) for key in ('speaker_candidate','score','margin','reason','recognition_reason','best_candidate')}
        result.update(score=score,margin=margin,best_candidate=pid,recognition_reason=decision,
            speaker_candidate=pid if decision=='matched' else None,reason='short_window')
        result['identity_evidence']=self._identity_evidence(result,source,group['source_turn_ids'])
        window.clear()  # Published windows are disjoint, never two votes from the same speech.

    def _store_reference(self,person,samples,vector,turn_id,session,metadata=None):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}',turn_id):raise ValueError('Invalid speaker sample ID')
        from vivis.interaction.memory import person_key
        person_key(person)
        rows=self.gallery.get(person,[])
        if len(rows)>=self.c['maximum_samples'] or any(r['turn_id']==turn_id for r in rows):return False
        folder=self.directory/person;folder.mkdir(exist_ok=True,mode=0o700);path=folder/(turn_id+'.wav')
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        try:
            with os.fdopen(fd,'wb') as stream:
                with wave.open(stream,'wb') as w:
                    w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(np.rint(samples*32767).astype('<i2').tobytes())
                stream.flush();os.fsync(stream.fileno())
            row=dict(turn_id=turn_id,session=session,created=time.time(),vector=vector.tolist(),active=True,locked=False)
            if metadata:row.update(metadata)
            self.save_gallery(dict(self.gallery,**{person:[*rows,row]}))
        except Exception:path.unlink(missing_ok=True);raise
        return True

    def _collect_short(self,pcm,far,source,query):
        try:
            self.refresh_gallery()
            spans=source.get('_sampling_ranges',[]);pid=spans[0]['person_id'] if spans else None
            used=[t for rows in self.gallery.values() for r in rows for t in r.get('source_turn_ids',[r['turn_id']])]
            group,notice=self.collector.accept(pcm,far,source,query,self.collection_epoch,
                self.c['maximum_samples']-len(self.gallery.get(pid,[])),used)
            if notice:
                if notice['status'] not in ('already_used','pool_full'):self.results.put(dict(collection_event=True,**notice))
                return
            if group is None:return
            pid=group['person_id'];samples=group['samples'];vector=self.feature(samples)
            # A composite reference must contain two independent, consistent speech portions.
            if len(group['metadata']['source_turn_ids'])>1 or len(samples)>=80000:
                windows,evidence=consistency_windows(samples,self.c['minimum_seconds'])
                if windows is None or float(self.feature(windows[0])@self.feature(windows[1]))<CONSISTENCY_THRESHOLD:
                    self.collector.defer()
                    self.results.put(dict(collection_event=True,status='voice_inconsistent',person_id=pid));return
            rows=self.gallery.get(pid,[]);active=[r for r in rows if r.get('active',True)]
            if active and float(normalized(np.mean([r['vector'] for r in active],axis=0))@vector)<.65:
                self.collector.defer()
                self.results.put(dict(collection_event=True,status='face_voice_conflict',person_id=pid));return
            ranked=[]
            for other,refs in self.gallery.items():
                refs=[r for r in refs if r.get('active',True)]
                if len(refs)>=self.c['enrollment_samples']:ranked.append((float(normalized(np.mean([r['vector'] for r in refs],axis=0))@vector),other))
            ranked.sort(reverse=True)
            if ranked:
                score,other=ranked[0];margin=score-ranked[1][0] if len(ranked)>1 else score
                if other!=pid and score>=self.c['threshold'] and margin>=self.c['minimum_margin']:
                    self.collector.clear()
                    self.results.put(dict(collection_event=True,status='known_other_voice',person_id=pid));return
            if spans and spans[0]['epoch']!=self.collection_epoch:self.collector.clear();return
            saved=self._store_reference(pid,samples,vector,group['turn_id'],group['session'],group['metadata'])
            self.collector.clear()
            if saved:self.results.put(dict(collection_event=True,status='saved',person_id=pid,
                enrollment_samples=len(self.gallery[pid]),source_turn_ids=group['metadata']['source_turn_ids']))
        except Exception as exc:
            self.collector.clear();self.results.put(dict(collection_event=True,status='unavailable:'+type(exc).__name__))

    def infer(self,pcm,far,source):
        result=dict(**{k:v for k,v in source.items() if not k.startswith('_')},speaker_candidate=None,observation_only=True,score=None,reason='unavailable',
            audio_quality_accepted=False,reference_sample_saved=False,enrollment_skip_reason=None,
            best_candidate=None,score_threshold=self.c['threshold'],minimum_margin=self.c['minimum_margin'],
            recognition_reason='quality_rejected')
        try:
            statistics={};samples,reason=audio_quality(pcm,far,self.c['minimum_seconds'],statistics)
            result.update(reason=reason,audio_quality=statistics)
            if samples is None:return
            result['audio_quality_accepted']=True
            if not source.get('_collection_managed'):
                if not source['enrollment_allowed']:result['enrollment_skip_reason']='visual_evidence_unconfirmed'
                elif not source['visual_person'] or source['visual_person']!=source['logical_person']:
                    result['enrollment_skip_reason']='source_identity_mismatch'
            self.refresh_gallery()
            vector=self.feature(samples)
            ranked=[]
            for pid,rows in self.gallery.items():
                rows=[r for r in rows if r.get('active',True)]
                if len(rows)>=self.c['enrollment_samples']:
                    center=normalized(np.mean([r['vector'] for r in rows],axis=0));ranked.append((float(center@vector),pid))
            ranked.sort(reverse=True)
            if ranked:
                score,pid=ranked[0];margin=score-ranked[1][0] if len(ranked)>1 else score
                result.update(score=score,margin=margin,best_candidate=pid,
                    recognition_reason='low_similarity' if score<self.c['threshold'] else
                        'ambiguous_candidates' if margin<self.c['minimum_margin'] else 'matched')
            else:result['recognition_reason']='insufficient_reference_samples'
            # Matching diagnostics may be computed now, but no identity candidate
            # is published until independent-speaker consistency also passes.
            if len(samples)>=80000:
                mode=self.c.get('consistency_window_mode','fixed')
                if mode=='speech_aligned':windows,evidence=consistency_windows(samples,self.c['minimum_seconds'])
                else:windows,evidence=(samples[:40000],samples[-40000:]),{}
                result['consistency']=dict(mode=mode,threshold=CONSISTENCY_THRESHOLD,**evidence)
                similarity=float(self.feature(windows[0])@self.feature(windows[1])) if windows else None
                result['consistency']['similarity']=similarity
                if similarity is None or similarity<CONSISTENCY_THRESHOLD:
                    result.update(reason='mixed_or_unstable_speech',recognition_reason='consistency_rejected');return
            if result['recognition_reason']=='matched':result['speaker_candidate']=result['best_candidate']
            person=source['visual_person']
            if not source.get('_collection_managed') and source['enrollment_allowed'] and person and person==source['logical_person']:
                rows=self.gallery.get(person,[])
                active_rows=[r for r in rows if r.get('active',True)]
                if active_rows and float(normalized(np.mean([r['vector'] for r in active_rows],axis=0))@vector)<.65:
                    result['reason']='face_voice_conflict';return
                if result['speaker_candidate'] not in (None,person):
                    result['reason']='known_other_voice';return
                result.update(enrollment_samples=len(rows),reference_collection_complete=len(rows)>=self.c['maximum_samples'])
                if len(rows)>=self.c['maximum_samples']:
                    result['reason']='reference_pool_full';return
                if not any(row['turn_id']==source['turn_id'] for row in rows):
                    self._store_reference(person,samples,vector,source['turn_id'],source['session'])
                    result['enrollment_samples']=len(self.gallery[person])
                    result['reference_collection_complete']=len(self.gallery[person])>=self.c['maximum_samples']
                    result['reference_sample_saved']=True
                    if len(self.gallery[person])<self.c['enrollment_samples']:result['reason']='collecting_reference_samples'
        except Exception as exc:result['reason']='speaker_unavailable:'+type(exc).__name__
        finally:
            try:self._enrich_recognition(pcm,far,source,result)
            except Exception as exc:
                from vivis.interaction.identity_fusion import unavailable
                result['identity_evidence']=unavailable('recognition_window:'+type(exc).__name__)
            self.results.put(result)
            if self.collector and source.get('_collection_managed'):self._collect_short(pcm,far,source,result)

    def poll(self):
        now=time.monotonic()
        if self.collector and self.collector.batch and now-self.collection_housekeeping>=1:
            self.collection_housekeeping=now;self.job.offer(self.collector.expire,now,self.collection_epoch)
        while True:
            try:result=self.results.get_nowait()
            except queue.Empty:return
            if result.get('collection_event'):
                self.log('speaker_collection',**result)
                name=self.labels.get(result.get('person_id'),'other');status=result['status']
                if status=='accumulating':message=f'声纹短句暂存：{name}，有效发声{result["effective_s"]:.2f}/{result["target_s"]:.2f}秒，来自{result["turns"]}轮；尚未入库'
                elif status=='saved':message=f'声纹聚合参考已入库：{name}，{result["enrollment_samples"]}/{self.c["maximum_samples"]}，来源{len(result["source_turn_ids"])}轮'
                else:message='声纹聚合参考未入库：'+name+'；'+status
                self.emit(dict(kind='voice_warning',message=message,speaker_collection=result));continue
            self.log('speaker_observation',**result)
            if self.result_handler:
                try:self.result_handler(result)
                except Exception as exc:self.log('speaker_result_rejected',error_type=type(exc).__name__,turn_id=result['turn_id'])
            def name(pid):
                if not pid or str(pid).startswith('anonymous:'):return 'other'
                return self.labels.get(pid,'other')
            if self.result_handler:
                from vivis.interaction.identity_fusion import evidence_label
                visual=result.get('source_visual_evidence') or dict(state='unavailable')
                profiles={pid:dict(preferred_address=label) for pid,label in self.labels.items()}
                label=name(result['logical_person'])+'-'+evidence_label(visual,profiles,'图')+'-'+evidence_label(result['identity_evidence'],profiles,'声')
            else:label=name(result['logical_person'])+'-'+name(result['visual_person'])+'/图-'+name(result['speaker_candidate'])+'/声'
            message='声纹观察［原话轮'+result['turn_id'][:8]+'］：'+label+'；'+result['reason']
            if result.get('reference_sample_saved'):
                message+='；参考已入库（'+str(result['enrollment_samples'])+'/'+str(self.c['maximum_samples'])+'）'
            elif result.get('enrollment_skip_reason'):
                message+='；未入库：原话轮起止身份未确认或不一致'
            if result['reason']=='insufficient_speech':
                stats=result.get('audio_quality',{});effective=stats.get('effective_speech_s')
                if effective is not None:message+=f'；有效发声{effective:.2f}秒，需要{self.c["minimum_seconds"]:.2f}秒'
            if result['reason']=='face_voice_conflict':message+='；与该人已有声纹参考不一致，未新增样本'
            decision=result.get('recognition_reason');candidate=name(result.get('best_candidate'))
            if decision=='low_similarity':message+=f'；声纹未确认：候选{candidate}，相似度{result["score"]:.3f}低于{self.c["threshold"]:.3f}'
            elif decision=='ambiguous_candidates':message+=f'；声纹未确认：候选差距{result["margin"]:.3f}不足{self.c["minimum_margin"]:.3f}'
            elif decision=='matched':message+=f'；声纹匹配通过：{name(result["speaker_candidate"])}'+('（提交身份融合）' if self.result_handler else '（仅观察）')
            elif decision=='insufficient_reference_samples':message+='；暂无至少3段活动参考的人物可比对'
            elif decision=='consistency_rejected':
                detail=result.get('consistency',{});score=detail.get('similarity')
                message+=f'；分段声音一致性{score:.3f}低于{CONSISTENCY_THRESHOLD:.3f}' if score is not None else '；两段独立有效发声不足，暂不确认'
            if result.get('window_match'):
                message+='；短句识别窗口来自'+str(len(result['window_match']['source_turn_ids']))+'轮'
            self.emit(dict(kind='voice_warning',message=message,speaker_observation=result))
