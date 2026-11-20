"""Local body presence, independent of face identity and actuator targeting.

NanoDet decoding/preprocessing adapted from OpenCV Zoo nanodet.py/demo.py
(Apache-2.0, pinned provenance and license in assets/models/nanodet).
Changes: person class only, stable softmax, clipped original-image boxes and
a single latest-frame worker so inference never blocks face tracking.
"""
import hashlib
import json
import math
import threading
import time

import numpy as np

from vivis.paths import ROOT


def configuration(root=ROOT):
    c=json.loads((root/'configs/perception/person_detection.json').read_text())
    if (c.get('schema_version')!='vivis-person-detection-1.0' or type(c.get('enabled')) is not bool
            or not isinstance(c.get('model_manifest'),str)):
        raise ValueError('Invalid body presence configuration')
    for key,low,high in (('hz',1,10),('score_threshold',.1,1),('nms_threshold',.1,1),('min_area_ratio',.001,.5)):
        v=c.get(key)
        if type(v) not in (int,float) or not math.isfinite(v) or not low<=v<=high:
            raise ValueError('Invalid body presence setting: '+key)
    return c


def verify_model(c,root=ROOT):
    manifest=json.loads((root/c['model_manifest']).read_text())
    path=root/manifest['destination']
    if (not path.is_file() or path.stat().st_size!=manifest['size']
            or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']):
        raise ValueError('Bundled NanoDet model missing or changed: '+str(path))
    return path


class PersonDetector:
    def __init__(self,c,root=ROOT):
        import cv2
        self.cv=cv2;self.c=c
        self.net=cv2.dnn.readNet(str(verify_model(c,root)))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.anchors=[]
        for stride in (8,16,32):
            x,y=np.meshgrid(np.arange(416//stride),np.arange(416//stride))
            self.anchors.append(np.column_stack((x.ravel(),y.ravel()))*stride+(stride-1)*.5)

    def detect(self,frame):
        begin=time.monotonic();height,width=frame.shape[:2]
        # Same RGB/normalization and black letterbox as the official demo.
        scale=416/max(height,width)
        nw,nh=max(1,int(width*scale)),max(1,int(height*scale))
        left,top=(416-nw)//2,(416-nh)//2
        resized=self.cv.resize(self.cv.cvtColor(frame,self.cv.COLOR_BGR2RGB),(nw,nh),interpolation=self.cv.INTER_AREA)
        image=np.zeros((416,416,3),dtype=np.float32)
        image[top:top+nh,left:left+nw]=resized
        image=(image-np.array([103.53,116.28,123.675],np.float32))/np.array([57.375,57.12,58.395],np.float32)
        self.net.setInput(self.cv.dnn.blobFromImage(image))
        outputs=self.net.forward(self.net.getUnconnectedOutLayersNames())
        boxes=[];scores=[]
        if len(outputs)!=6:raise ValueError('Unexpected NanoDet output count')
        for stride,anchors,cls,reg in zip((8,16,32),self.anchors,outputs[::2],outputs[1::2]):
            cls=np.asarray(cls).reshape(-1,80);reg=np.asarray(reg).reshape(-1,4,8)
            if len(cls)!=len(anchors) or len(reg)!=len(anchors):raise ValueError('Unexpected NanoDet output shape')
            # A high score for a chair/dog must never count as a person.
            keep=np.flatnonzero((cls[:,0]>=self.c['score_threshold']) & (np.argmax(cls,axis=1)==0))
            if not len(keep):continue
            logits=reg[keep];probs=np.exp(logits-logits.max(axis=2,keepdims=True))
            distance=(probs/probs.sum(axis=2,keepdims=True))@np.arange(8)*stride
            corners=np.concatenate((anchors[keep]-distance[:,:2],anchors[keep]+distance[:,2:]),axis=1)
            corners[:,[0,2]]=np.clip((corners[:,[0,2]]-left)*width/nw,0,width)
            corners[:,[1,3]]=np.clip((corners[:,[1,3]]-top)*height/nh,0,height)
            for box,index in zip(corners,keep):
                x,y,x2,y2=map(float,box);w,h=x2-x,y2-y
                score=float(cls[index,0])
                if all(math.isfinite(v) for v in (x,y,w,h,score)) and w*h/(width*height)>=self.c['min_area_ratio'] and w>0 and h>0:
                    boxes.append([x,y,w,h]);scores.append(score)
        indices=np.asarray(self.cv.dnn.NMSBoxes(boxes,scores,self.c['score_threshold'],self.c['nms_threshold'])).reshape(-1)
        people=[dict(bbox=boxes[int(i)],score=scores[int(i)]) for i in indices]
        return dict(persons=len(people),boxes=people,processing_ms=(time.monotonic()-begin)*1000)


class PersonWorker:
    """At most one running inference and one replaceable pending frame; no FIFO."""
    def __init__(self,c,root=ROOT,detector=None):
        self.c=c;self.condition=threading.Condition();self.pending=None;self.result=None
        self.stopped=False;self.error=None;self.last=-float('inf');self.thread=None
        self.detector=None
        if not c['enabled']:return
        try:self.detector=detector or PersonDetector(c,root)
        except Exception as exc:self.error=str(exc);return
        self.thread=threading.Thread(target=self._run,name='body-presence',daemon=True);self.thread.start()

    def submit(self,sample,now=None):
        now=time.monotonic() if now is None else now
        with self.condition:
            if self.stopped or self.error or not self.detector or now-self.last<1/self.c['hz']:return
            self.last=now;self.pending=sample;self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.stopped or self.pending is not None)
                if self.stopped:return
                sample=self.pending;self.pending=None
            try:
                result=self.detector.detect(sample[2])
                result.update(observed_monotonic=sample[1],sequence=sample[0])
                with self.condition:self.result=result
            except Exception as exc:
                with self.condition:self.error=str(exc);self.result=None;self.pending=None
                return  # Local fault disables only body-based greeting admission.

    def observation(self):
        with self.condition:
            return dict(enabled=self.c['enabled'],available=self.detector is not None and not self.error,
                        error=self.error,**(self.result or dict(persons=0,boxes=[],observed_monotonic=None)))

    def close(self):
        with self.condition:self.stopped=True;self.pending=None;self.condition.notify_all()
        if self.thread:self.thread.join(timeout=1)
