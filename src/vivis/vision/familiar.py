"""Optional local identity matching of the SAME selected face; no actuator I/O."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

import numpy as np

from vivis.runtime.best_effort import BestEffortJob
from vivis.vision.identity_config import backend_config


def feature(model,image,row):
    vector=np.asarray(model.feature(model.alignCrop(image,np.asarray(row,dtype=np.float32))),dtype=np.float32).reshape(-1)
    norm=float(np.linalg.norm(vector))
    if vector.size!=128 or not np.isfinite(vector).all() or not math.isfinite(norm) or norm<1e-6:
        raise ValueError('Invalid SFace embedding')
    return vector/norm


def match(vector,vectors,ids,threshold,margin):
    scores=np.asarray(vectors)@vector
    return match_scores(scores,ids,threshold,margin)


def match_scores(scores,ids,threshold,margin,details=None):
    scores=np.asarray(scores,dtype=float)
    if not np.isfinite(scores).all():raise ValueError('Non-finite identity score')
    by_person={pid:float(np.max(scores[np.asarray(ids)==pid])) for pid in set(ids)}
    ranked=sorted(by_person.items(),key=lambda p:p[1],reverse=True)
    if not ranked:return None,0.
    pid,score=ranked[0];gap=score-ranked[1][1] if len(ranked)>1 else 1.
    if details is not None:
        details.update(candidate_id=str(pid),score_margin=gap,score_threshold=threshold,minimum_margin=margin,
            decision_reason='below_threshold' if score<threshold else 'ambiguous_candidates' if gap<margin else 'accepted_frame')
    return (pid if score>=threshold and gap>=margin else None),score


class FamiliarFaces:
    def __init__(self,config,root,model=None,gallery=None):
        config=backend_config(config)
        self.c=config;self.root=Path(root);self.owner=uuid.uuid4().hex
        self.model=None;self._closing=False;self.photo_pool=None;self.photo_error=None;self.photo_epoch=None
        self.brightness_samples=None;self.brightness_error=None
        self.path=self.root/config['state_file'];self.epoch=0;self.last_submit=-math.inf
        self.lock=threading.Lock();self.job=BestEffortJob('familiar-face');self.candidate=None;self.hits=0;self.last_bbox=None
        self.confirmed=None;self.last_confirmed=-math.inf
        self.result=self.unknown();self.available=False;self.error=None;self.published=None
        if (config.get('schema_version')!='vivis-familiar-faces-1.0'
                or type(config.get('confirmation_samples')) is not int or not 2<=config['confirmation_samples']<=10
                or not .35<=config['score_threshold']<=.95 or not .01<=config['minimum_margin']<=.5
                or not .5<=config['hz']<=5 or not .3<=config['max_result_age_s']<=3
                or type(config.get('review_hold_s',5.)) not in (int,float) or not 1<=config.get('review_hold_s',5.)<=15
                or type(config.get('min_face_size_px')) is not int or not 16<=config['min_face_size_px']<=512):
            raise ValueError('Invalid familiar-face configuration')
        suspect=config.get('suspect_threshold',config['score_threshold'])
        if type(suspect) not in (int,float) or not math.isfinite(suspect) or not 0<=suspect<=config['score_threshold']:
            raise ValueError('Invalid familiar-face suspect threshold')
        if config['backend']=='seeta6' and (type(config.get('quality_gate',True)) is not bool or type(config.get('age_estimation',False)) is not bool):
            raise ValueError('Invalid SeetaFace options')
        if not config.get('enabled'):return
        try:self._load(model,gallery)
        except Exception as exc:
            # Identification assets are optional; gaze and interaction remain usable.
            self.error='Local '+config['backend']+' assets invalid: '+type(exc).__name__;self.available=False
        diagnostics=config.get('brightness_diagnostics',{})
        if self.available and diagnostics.get('enabled'):
            try:
                from vivis.vision.brightness_samples import BrightnessSamples
                brightness_range=config.get('photo_sampling',{}).get('brightness_range',[.15,.9])
                self.brightness_samples=BrightnessSamples(self.root,diagnostics,brightness_range)
            except Exception as exc:self.brightness_error=type(exc).__name__

    def _load(self,model,gallery):
        config=self.c
        photo_hashes=None
        if gallery is None:
            path=self.root/config['gallery']
            if not path.is_file():raise FileNotFoundError('Face gallery missing')
            with np.load(path,allow_pickle=False) as data:
                if config['backend']=='seeta6' and (str(data['backend'])!='seeta6' or str(data['model_sha256'])!=config['sha256']):
                    raise ValueError('SeetaFace gallery/model mismatch')
                if config['backend']=='seeta6' and str(data['embedding_contract'])!='seeta6-yunet-roi-pts5-v1':raise ValueError('SeetaFace alignment mismatch')
                gallery=(data['vectors'],data['ids'])
                if 'photo_sha256' in data:photo_hashes=data['photo_sha256'].copy()
        self.vectors,self.ids=gallery
        self.vectors=np.asarray(self.vectors,dtype=np.float32);self.ids=np.asarray(self.ids)
        dimension=1024 if config['backend']=='seeta6' else 128
        if (self.vectors.ndim!=2 or self.vectors.shape[1]!=dimension or len(self.vectors)!=len(self.ids) or not len(self.ids)
                or not np.isfinite(self.vectors).all() or np.any(np.linalg.norm(self.vectors,axis=1)<1e-6)
                or (config['backend']=='sface' and np.any(abs(np.linalg.norm(self.vectors,axis=1)-1)>1e-3))
                or any(not re.fullmatch(r'[a-z][a-z0-9_-]{1,39}',str(pid)) for pid in self.ids)):
            raise ValueError('Invalid local face gallery')
        if model is None:
            path=self.root/config['model']
            if not path.is_file():return
            if hashlib.sha256(path.read_bytes()).hexdigest()!=config['sha256']:raise ValueError('SFace model hash mismatch')
            if config['backend']=='seeta6':
                from vivis.vision.seeta import Seeta
                model=Seeta(root=self.root,age=config.get('age_estimation',False))
            else:
                import cv2
                cv2.setNumThreads(1);model=cv2.FaceRecognizerSF.create(str(path),'')
        self.model=model;self.available=True
        if config.get('photo_collection',False) and photo_hashes is not None:
            try:
                from vivis.vision.photo_pool import PhotoPool
                self.photo_pool=PhotoPool(self.root,config,self.vectors,self.ids,photo_hashes)
                self.vectors,self.ids=self.photo_pool.gallery()
            except Exception as exc:self.photo_error=type(exc).__name__

    def unknown(self):
        return dict(schema_version='vivis-familiar-person-1.0',person_id=None,state='unknown',
                    observed_monotonic=0.,similarity=None,owner=self.owner,
                    identity_evidence=dict(state='unavailable',person_id=None,stamp=None,evidence_id=None,reason='no_usable_face'))

    def _reset_locked(self):
        self.epoch+=1;self.candidate=None;self.hits=0;self.result=self.unknown()
        self.confirmed=None;self.last_confirmed=-math.inf

    def reset(self):
        with self.lock:self._reset_locked()

    def infer(self,image,row,stamp,epoch,collect=False):
        try:
            begin=time.perf_counter();observation={}
            if self.photo_pool:
                try:
                    self.photo_pool.refresh();self.vectors,self.ids=self.photo_pool.gallery()
                    self.photo_error=self.photo_pool.error
                except Exception as exc:self.photo_error=type(exc).__name__
            if self.c['backend']=='seeta6':
                observation,vector=self.model.evaluate_selected(image,row,gate=self.c.get('quality_gate',True))
                if vector is None:
                    pid,score=None,None;observation['decision_reason']='quality_low' if observation.get('quality_rejected') else 'no_feature'
                else:
                    scores=[self.model.compare(vector,v) for v in self.vectors]
                    pid,score=match_scores(scores,self.ids,self.c['score_threshold'],self.c['minimum_margin'],observation)
            else:
                vector=feature(self.model,image,row)
                pid,score=match_scores(self.vectors@vector,self.ids,self.c['score_threshold'],self.c['minimum_margin'],observation)
            observation['worker_ms']=(time.perf_counter()-begin)*1000
            with self.lock:
                if self.job.closed.is_set() or epoch!=self.epoch:return
                self.hits=self.hits+1 if pid is not None and pid==self.candidate else int(pid is not None)
                self.candidate=pid
                observation.update(confirmation_hits=self.hits,confirmation_required=self.c['confirmation_samples'])
                if pid is not None:observation['decision_reason']='confirmed' if self.hits>=self.c['confirmation_samples'] else 'confirming'
                # A credible different candidate invalidates the previous identity
                # immediately. Low confidence alone enters bounded same-face review;
                # consumers may retain an already established interaction identity.
                if pid is not None and pid!=self.confirmed:self.confirmed=None
                if pid is not None and self.hits>=self.c['confirmation_samples']:
                    self.confirmed=pid;self.last_confirmed=stamp;state='recognized'
                elif self.confirmed and 0<=stamp-self.last_confirmed<self.c.get('review_hold_s',5.):
                    state='reviewing'
                else:
                    self.confirmed=None;state='unknown'
                self.result=dict(schema_version='vivis-familiar-person-1.0',person_id=self.confirmed,
                    state=state,observed_monotonic=stamp,similarity=score,owner=self.owner,
                    backend=self.c['backend'],observation=observation)
                weak=observation.get('candidate_id');evidence_state='unavailable';evidence_person=None
                if vector is not None:
                    if pid is not None and self.hits>=self.c['confirmation_samples']:
                        evidence_state='confirmed';evidence_person=str(pid)
                    elif weak and score>=self.c.get('suspect_threshold',self.c['score_threshold']):
                        evidence_state='suspected';evidence_person=weak
                    else:evidence_state='other'
                self.result['identity_evidence']=dict(state=evidence_state,person_id=evidence_person,
                    stamp=stamp,evidence_id=self.owner+':'+str(stamp)+':'+str(epoch),score=score,
                    reason=observation.get('decision_reason'),owner=self.owner)
                if self.confirmed:self.result['confirmed_monotonic']=self.last_confirmed
                self.error=None
            # Diagnostic capture is independent of photo enrollment: rejected
            # unknown faces and full personal pools still provide useful evidence.
            if self.brightness_samples:
                try:self.brightness_samples.offer(image,row,observation,stamp)
                except Exception as exc:self.brightness_error=type(exc).__name__
            if self.photo_pool:
                try:
                    if self.photo_epoch!=epoch:self.photo_pool.observe_identity(None,stamp)
                    self.photo_epoch=epoch
                    eligible=collect and state=='recognized' and pid is not None and vector is not None
                    self.photo_pool.observe_identity(str(pid) if eligible else None,stamp)
                    if eligible:
                        compare=self.model.compare if self.c['backend']=='seeta6' else lambda a,b:float(a@b)
                        self.photo_pool.consider(str(pid),image,row,vector,observation,compare,stamp)
                    if self.photo_pool.error:self.photo_error=self.photo_pool.error
                except Exception as exc:self.photo_error=type(exc).__name__
        except Exception:
            # Optional identification must not stop gaze/motion or reveal raw
            # source details. Treat a failed inference as unknown.
            self.error='Local face recognition failed';self.reset()

    def step(self,sample,row,now=None,*,faces=1,collect=False):
        now=time.monotonic() if now is None else now
        # Invalidate old evidence BEFORE submitting a new asynchronous sample.
        # Otherwise a fast result could reuse confirmation hits from a stale face.
        with self.lock:
            result=self.result
            if result['observed_monotonic']>0 and (
                    not 0<=now-result['observed_monotonic']<=self.c['max_result_age_s']
                    or (result['state']=='reviewing' and now-result['confirmed_monotonic']>=self.c.get('review_hold_s',5.))):
                self._reset_locked()
        if row is None or faces!=1 or not self.available:
            self.reset();self.last_bbox=None
        else:
            x,y,w,h=map(float,row[:4]);bbox=np.array([x,y,w,h])
            if self.last_bbox is not None and np.linalg.norm(bbox[:2]-self.last_bbox[:2])>max(w,h):self.reset()
            self.last_bbox=bbox
            if (min(w,h)>=self.c['min_face_size_px'] and now-self.last_submit>=1/self.c['hz']
                    and 0<=now-sample[1]<=self.c['max_result_age_s']):
                epoch=self.epoch
                arguments=(sample[2].copy(),np.asarray(row,dtype=np.float32).copy(),sample[1],epoch)
                if collect:arguments=(*arguments,True)
                if self.job.offer(self.infer,*arguments):self.last_submit=now
        with self.lock:result=dict(self.result)
        if (not 0<=now-result['observed_monotonic']<=self.c['max_result_age_s']
                or (result['state']=='reviewing' and now-result['confirmed_monotonic']>=self.c.get('review_hold_s',5.))):
            result=self.unknown()
        return result

    def publish(self,result):
        if result==self.published:return
        temporary=self.path.with_name(self.path.name+'.'+self.owner+'.tmp')
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            with os.fdopen(fd,'w') as stream:json.dump(result,stream,ensure_ascii=False)
            temporary.replace(self.path);self.published=dict(result)
        except OSError:
            self.error='Cannot publish optional local face identity'
            try:temporary.unlink(missing_ok=True)
            except OSError:pass

    def close(self):
        if self.brightness_samples:self.brightness_samples.close()
        self.job.close();self.reset()
        if self.model is not None and not self._closing:
            self._closing=True;model=self.model
            def release():
                self.job.wait()
                if self.photo_pool:self.photo_pool.close()
                if self.c['backend']=='seeta6':model.close()
            if self.job.thread is not None and self.job.thread.is_alive():
                threading.Thread(target=release,daemon=True,name='identity-release').start()
            else:release()
        try:
            if json.loads(self.path.read_text()).get('owner')==self.owner:self.path.unlink()
        except (OSError,ValueError):pass
