"""Bounded private examples of rejected brightness; never enroll or alter recognition."""
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from vivis.runtime.best_effort import BestEffortJob


class BrightnessSamples:
    def __init__(self,root,config,brightness_range):
        self.directory=(Path(root)/config['directory']).resolve()
        private=(Path(root)/'private').resolve()
        self.maximum=config['maximum_images'];self.interval=config['sample_interval_s']
        self.range=list(brightness_range)
        if (not self.directory.is_relative_to(private) or self.directory==private
                or type(self.maximum) is not int or not 1<=self.maximum<=100
                or type(self.interval) not in (int,float) or not math.isfinite(self.interval) or not .5<=self.interval<=60
                or len(self.range)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in self.range)
                or not 0<=self.range[0]<self.range[1]<=1):
            raise ValueError('Invalid private brightness diagnostics policy')
        self.job=BestEffortJob('brightness-diagnostic-writer');self.last_stamp=-math.inf
        self.status=dict(saved_this_process=0,retained_images=None,last_file=None,error=None)

    def offer(self,image,row,observation,stamp):
        if image is None or not isinstance(image,np.ndarray) or not math.isfinite(stamp):return False
        if stamp-self.last_stamp<self.interval:return False
        import cv2
        x,y,w,h=map(int,row[:4]);left=max(0,x);top=max(0,y)
        right=min(image.shape[1],x+w);bottom=min(image.shape[0],y+h)
        if right<=left or bottom<=top:return False
        mean=float(cv2.cvtColor(image[top:bottom,left:right],cv2.COLOR_BGR2GRAY).mean())/255
        reasons=[]
        if not self.range[0]<mean<self.range[1]:reasons.append('sampling_brightness')
        if (observation.get('brightness') or {}).get('level')==0:reasons.append('native_brightness')
        if not reasons:return False
        metadata=dict(observed_monotonic=stamp,face_bbox=list(map(float,row[:4])),
            brightness_mean_0_1=mean,sampling_brightness_range=self.range,rejection_reasons=reasons,
            native_brightness=observation.get('brightness'),native_diagnostics=observation.get('brightness_diagnostics'),
            diagnostic_only=True,identity_authority=False)
        accepted=self.job.offer(self._save,image,metadata)
        if accepted:self.last_stamp=stamp
        return accepted

    def _save(self,image,metadata):
        written=[]
        try:
            import cv2
            self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
            with (self.directory/'.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:return
                count=sum(p.suffix.lower() in ('.jpg','.jpeg','.png','.webp') for p in self.directory.iterdir())
                self.status={**self.status,'retained_images':count}
                if self.job.closed.is_set() or count>=self.maximum:return
                ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,92])
                if not ok:raise ValueError('Cannot encode brightness diagnostic')
                now=datetime.now().astimezone()
                reason='native-low' if 'native_brightness' in metadata['rejection_reasons'] else 'range-low' if metadata['brightness_mean_0_1']<=self.range[0] else 'range-high'
                name=f"{metadata['brightness_mean_0_1']:.4f}_{reason}_{now:%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}"
                metadata=dict(metadata,created_at=now.isoformat(),image=name+'.jpg')
                for suffix,raw in (('.json',json.dumps(metadata,ensure_ascii=False,indent=2).encode()),('.jpg',encoded.tobytes())):
                    path=self.directory/(name+suffix)
                    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);written.append(path)
                    with os.fdopen(fd,'wb') as stream:stream.write(raw)
                self.status=dict(saved_this_process=self.status['saved_this_process']+1,
                    retained_images=count+1,last_file=name+'.jpg',error=None)
        except Exception as exc:
            for path in written:
                try:path.unlink(missing_ok=True)
                except OSError:pass
            self.status={**self.status,'error':type(exc).__name__}

    def close(self):self.job.close()
