"""Optional local SFace matching of the SAME selected face; no actuator I/O."""
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


def feature(model,image,row):
    vector=np.asarray(model.feature(model.alignCrop(image,np.asarray(row,dtype=np.float32))),dtype=np.float32).reshape(-1)
    norm=float(np.linalg.norm(vector))
    if vector.size!=128 or not np.isfinite(vector).all() or not math.isfinite(norm) or norm<1e-6:
        raise ValueError('Invalid SFace embedding')
    return vector/norm


def match(vector,vectors,ids,threshold,margin):
    scores=np.asarray(vectors)@vector
    by_person={pid:float(np.max(scores[np.asarray(ids)==pid])) for pid in set(ids)}
    ranked=sorted(by_person.items(),key=lambda p:p[1],reverse=True)
    if not ranked:return None,0.
    pid,score=ranked[0];gap=score-ranked[1][1] if len(ranked)>1 else 1.
    return (pid if score>=threshold and gap>=margin else None),score


class FamiliarFaces:
    def __init__(self,config,root,model=None,gallery=None):
        self.c=config;self.root=Path(root);self.owner=uuid.uuid4().hex
        self.path=self.root/config['state_file'];self.epoch=0;self.last_submit=-math.inf
        self.lock=threading.Lock();self.job=BestEffortJob('familiar-face');self.candidate=None;self.hits=0;self.last_bbox=None
        self.result=self.unknown();self.available=False;self.error=None;self.published=None
        if (config.get('schema_version')!='vivis-familiar-faces-1.0'
                or type(config.get('confirmation_samples')) is not int or not 2<=config['confirmation_samples']<=10
                or not .35<=config['cosine_threshold']<=.95 or not .01<=config['minimum_margin']<=.5
                or not .5<=config['hz']<=5 or not .3<=config['max_result_age_s']<=3
                or type(config.get('min_face_size_px')) is not int or not 16<=config['min_face_size_px']<=512):
            raise ValueError('Invalid familiar-face configuration')
        if not config.get('enabled'):return
        try:self._load(model,gallery)
        except Exception:
            # Identification assets are optional; gaze and interaction remain usable.
            self.error='Local face recognition assets invalid';self.available=False

    def _load(self,model,gallery):
        config=self.c
        if gallery is None:
            path=self.root/config['gallery']
            if not path.is_file():return
            with np.load(path,allow_pickle=False) as data:gallery=(data['vectors'],data['ids'])
        self.vectors,self.ids=gallery
        self.vectors=np.asarray(self.vectors,dtype=np.float32);self.ids=np.asarray(self.ids)
        if (self.vectors.ndim!=2 or self.vectors.shape[1]!=128 or len(self.vectors)!=len(self.ids) or not len(self.ids)
                or not np.isfinite(self.vectors).all() or np.any(abs(np.linalg.norm(self.vectors,axis=1)-1)>1e-3)
                or any(not re.fullmatch(r'[a-z][a-z0-9_-]{1,39}',str(pid)) for pid in self.ids)):
            raise ValueError('Invalid local face gallery')
        if model is None:
            path=self.root/config['model']
            if not path.is_file():return
            if hashlib.sha256(path.read_bytes()).hexdigest()!=config['sha256']:raise ValueError('SFace model hash mismatch')
            import cv2
            cv2.setNumThreads(1);model=cv2.FaceRecognizerSF.create(str(path),'')
        self.model=model;self.available=True

    def unknown(self):
        return dict(schema_version='vivis-familiar-person-1.0',person_id=None,state='unknown',
                    observed_monotonic=0.,similarity=None,owner=self.owner)

    def reset(self):
        with self.lock:
            self.epoch+=1;self.candidate=None;self.hits=0;self.result=self.unknown()

    def infer(self,image,row,stamp,epoch):
        try:
            vector=feature(self.model,image,row)
            pid,score=match(vector,self.vectors,self.ids,self.c['cosine_threshold'],self.c['minimum_margin'])
            with self.lock:
                if self.job.closed.is_set() or epoch!=self.epoch:return
                self.hits=self.hits+1 if pid is not None and pid==self.candidate else int(pid is not None)
                self.candidate=pid
                confirmed=pid if self.hits>=self.c['confirmation_samples'] else None
                self.result=dict(schema_version='vivis-familiar-person-1.0',person_id=confirmed,
                    state='recognized' if confirmed else 'unknown',observed_monotonic=stamp,similarity=score,owner=self.owner)
        except Exception:
            # Optional identification must not stop gaze/motion or reveal raw
            # source details. Treat a failed inference as unknown.
            self.error='Local face recognition failed';self.reset()

    def step(self,sample,row,now=None):
        now=time.monotonic() if now is None else now
        if row is None or not self.available:
            self.reset();self.last_bbox=None
        else:
            x,y,w,h=map(float,row[:4]);bbox=np.array([x,y,w,h])
            if self.last_bbox is not None and np.linalg.norm(bbox[:2]-self.last_bbox[:2])>max(w,h):self.reset()
            self.last_bbox=bbox
            if (min(w,h)>=self.c['min_face_size_px'] and now-self.last_submit>=1/self.c['hz']
                    and 0<=now-sample[1]<=self.c['max_result_age_s']):
                epoch=self.epoch
                if self.job.offer(self.infer,sample[2].copy(),np.asarray(row,dtype=np.float32).copy(),sample[1],epoch):self.last_submit=now
        with self.lock:result=dict(self.result)
        if not 0<=now-result['observed_monotonic']<=self.c['max_result_age_s']:result=self.unknown()
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
        self.job.close();self.reset()
        try:
            if json.loads(self.path.read_text()).get('owner')==self.owner:self.path.unlink()
        except (OSError,ValueError):pass
