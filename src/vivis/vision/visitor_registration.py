"""Small ephemeral seed cache in the existing selected-face inference worker.

Only an explicit, current familiar introduction can persist these images. Native
feature extraction and quality gates stay owned by FamiliarFaces' single job.
"""
import hashlib
import json
import time
import uuid

import numpy as np

from vivis.interaction.introductions import Introductions,atomic


class VisitorRegistration:
    def __init__(self,root,config,owner):
        self.root=root;self.c=config;self.owner=owner;self.cache={};self.token=None
        self.last=0.;self.started=0.;self.bbox=None;self.loaded=set();self.revision=None
        self.request_signature=None;self.pending=False;self.next_check=0.
        self.store=Introductions(root)

    def track(self,row,stamp,faces):
        if row is None or faces!=1:
            self.token=None;self.bbox=None;return
        bbox=np.asarray(row[:4],float)
        if (self.token is None or stamp-self.last>1. or self.bbox is None
                or np.linalg.norm(bbox[:2]-self.bbox[:2])>max(bbox[2:])):
            self.token=uuid.uuid4().hex;self.started=stamp
        self.last=stamp;self.bbox=bbox

    def observe(self,token,image,row,vector,observation,stamp,person,compare):
        self.cache={k:v for k,v in self.cache.items() if 0<=stamp-v['stamp']<=120}
        if not token or vector is None or observation.get('quality_rejected'):return token
        existing=self.cache.get(token)
        if (existing and existing['seeds'] and
                max(compare(vector,s['vector']) for s in existing['seeds'])<self.c['score_threshold']):
            # A different person may take exactly the same bounding box without
            # an intervening empty frame. Geometry alone cannot lock a visitor.
            replacement=uuid.uuid4().hex
            if self.token==token:self.token=replacement;self.started=stamp
            token=replacement
        # Reacquire the original unknown face after a short head glance. This
        # restores only a visit token, never a claimed familiar identity.
        if token not in self.cache:
            ranked=sorted(((max(compare(vector,s['vector']) for s in v['seeds']),k)
                for k,v in self.cache.items() if v['seeds']),reverse=True)
            if ranked and ranked[0][0]>=self.c['score_threshold'] and (
                    len(ranked)==1 or ranked[0][0]-ranked[1][0]>=self.c['minimum_margin']):
                old=ranked[0][1]
                if self.token==token:self.token=old
                token=old
        entry=self.cache.setdefault(token,dict(stamp=stamp,seeds=[],person_id=person))
        entry.update(stamp=stamp,person_id=person)
        if len(self.cache)>4:
            oldest=min(self.cache,key=lambda k:self.cache[k]['stamp'])
            if oldest!=token:self.cache.pop(oldest)
        if person:return token
        # Bootstrap uses exactly the same collection quality criteria, separately
        # from recognition. Poor frames cannot become trusted registration bases.
        import cv2
        x,y,w,h=map(int,row[:4]);crop=image[max(0,y):max(0,y)+h,max(0,x):max(0,x)+w]
        if not crop.size:return token
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        brightness=float(gray.mean()/255);sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        sampling=self.c.get('photo_sampling') or {};lo,hi=sampling.get('brightness_range',[.1,.95])
        if (min(w,h)<sampling.get('min_face_size_px',70) or not lo<=brightness<=hi
                or sharpness<sampling.get('min_laplacian_variance',25)):return token
        if entry['seeds'] and stamp-entry['seeds'][-1]['stamp']<.75:return token
        if len(entry['seeds'])<4:
            if any(np.array_equal(s['image'],image) for s in entry['seeds']):return token
            entry['seeds'].append(dict(stamp=stamp,image=image.copy(),vector=vector.copy(),brightness=brightness))
        return token

    def status(self,stamp,person=None):
        if not self.token:return None
        entry=self.cache.get(self.token,{})
        # Publishing a fresh geometry timestamp at camera60Hz would turn the
        # existing4Hz identity state writer into60Hz disk I/O unnecessarily.
        return dict(token=self.token,owner=self.owner,stamp=entry.get('stamp',self.started),person_id=person,
            seed_count=len(entry.get('seeds',[])))

    def complete(self,stamp):
        directory=self.store.directory/'requests'
        try:signature=directory.stat().st_mtime_ns
        except FileNotFoundError:return
        if signature==self.request_signature and not self.pending:return
        if self.pending and signature==self.request_signature and stamp<self.next_check:return
        self.request_signature=signature;self.next_check=stamp+.5
        requests=self.store.statuses();self.pending=any(r['status']=='pending_face' for r in requests)
        for request in requests:
            if request['status']!='pending_face':continue
            path=self.store.directory/'requests'/f"{request['id']}.json"
            guest=request['guest'];token=guest['token']
            manifest=self.store.directory/'face_anchors.json'
            anchors=json.loads(manifest.read_text()) if manifest.exists() else dict(schema_version='vivis-online-face-1.0',people={})
            committed=anchors['people'].get(request['person_id'],{})
            if committed.get('source')==request['id']:
                request.update(status='ready',registered_monotonic=stamp);atomic(path,request);continue
            if guest['owner']!=self.owner or not 0<=stamp-request['created_monotonic']<=120:
                request.update(status='failed',error='访客跟踪已结束或注册超时，请重新介绍');atomic(path,request);continue
            entry=self.cache.get(token)
            if token!=self.token or not entry or not entry['seeds']:continue
            if entry.get('person_id'):
                request.update(status='failed',error='访客已匹配已有档案，不能重复注册');atomic(path,request);continue
            pid=request['person_id'];folder=self.root/self.c['people_directory']/pid
            info=request['information']
            profile=dict(schema_version='vivis-person-1.0',allow_cloud_context=True,
                display_name=info['display_name'] or info['preferred_address'],preferred_address=info['preferred_address'] or info['display_name'],
                organization='',role='',professional_summary=info['background'],source='familiar_introduction',
                introduction=dict(introducer=request['introducer'],date=request['created_at'],source=request['source'],
                    background_verified=False,guest_token=token))
            from vivis.interaction.knowledge import professional_profile
            professional_profile(profile)  # Identical cloud-profile validation.
            rows=[]
            import cv2
            for seed in entry['seeds']:
                ok,encoded=cv2.imencode('.jpg',seed['image'])
                if not ok:raise ValueError('Cannot encode registration photo')
                digest=hashlib.sha256(encoded.tobytes()).hexdigest()
                if any(r['hash']==digest for r in rows):continue
                target=folder/'photo_pool'/f'{digest}.jpg';target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                target.write_bytes(encoded.tobytes());target.chmod(0o600)
                rows.append(dict(hash=digest,path=str(target.relative_to(folder)),vector=seed['vector'].tolist(),
                    origin='introduction',locked=True,lock_reason='introduced_base',quality=1.,
                    coverage=[0.,0.,seed['brightness']],dhash=None))
            pool=dict(schema_version='vivis-photo-pool-1.0',person_id=pid,model_sha256=self.c['sha256'],
                revision=1,rows=rows,active=[r['hash'] for r in rows])
            # Publish registry last: incomplete files are invisible to matching.
            atomic(folder/'profile.json',profile);atomic(folder/'photo_pool.json',pool)
            anchors['people'][pid]=dict(model_sha256=self.c['sha256'],source=request['id'],rows=rows)
            atomic(manifest,anchors)
            request.update(status='ready',registered_monotonic=stamp);atomic(path,request)

    def refresh(self,pool):
        if pool is None:return
        manifest=self.store.directory/'face_anchors.json'
        if not manifest.exists():return
        revision=manifest.stat().st_mtime_ns
        if revision==self.revision:return
        data=json.loads(manifest.read_text())
        if data.get('schema_version')!='vivis-online-face-1.0':raise ValueError('Invalid online face registry')
        for pid,entry in data['people'].items():
            if pid in self.loaded:continue
            if entry['model_sha256']!=self.c['sha256']:raise ValueError('Online model mismatch')
            rows=entry['rows'];pool.add_anchors(pid,rows);self.loaded.add(pid)
        self.revision=revision
