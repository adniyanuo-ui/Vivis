"""Bounded, anchor-verified photo pool; never enroll from held identity alone."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def choose(rows,limit=16):
    locked=[r for r in rows if r['locked']]
    if len(locked)>4:raise ValueError('At most four locked base photos')
    selected=list(locked);remaining=[r for r in rows if not r['locked'] and not r.get('excluded',False)]
    while remaining and len(selected)<limit:
        def merit(row):
            distance=min((np.linalg.norm(np.asarray(row['coverage'])-r['coverage']) for r in selected),default=1.)
            return row['quality']+.8*min(float(distance),2.)
        best=max(remaining,key=merit);selected.append(best);remaining.remove(best)
    return [r['hash'] for r in selected]


class PhotoPool:
    def __init__(self,root,config,vectors,ids,hashes=None):
        self.root=Path(root);self.c=config
        self.directory=self.root/config['people_directory']
        if not self.directory.resolve().is_relative_to((self.root/'private').resolve()):raise ValueError('Private photo pool required')
        self.anchors=np.asarray(vectors).copy();self.anchor_ids=np.asarray(ids).copy()
        self.hashes=hashes;self.last={};self.version=0;self.last_refresh=0.
        self.pools={};self.signatures={}
        self.load()

    def load(self):
        from vivis.vision.enrollment import registration_photos
        for pid in sorted(set(self.anchor_ids)):
            pid=str(pid);folder=self.directory/pid
            if not (folder/'profile.json').is_file():continue
            path=folder/'photo_pool.json'
            if path.is_file():
                data=json.loads(path.read_text());self.validate(data,pid)
                if data.get('model_sha256')!=self.c.get('sha256'):raise ValueError('Pool/model mismatch')
                # Explicitly rebuilt manual galleries may add supplied photos.
                # Merge them on restart; never let an old pool hide those inputs.
                originals={hashlib.sha256(p.read_bytes()).hexdigest():p for p in registration_photos(folder)}
                known={r['hash'] for r in data['rows']};changed=False
                for index in np.where(self.anchor_ids==pid)[0]:
                    digest=str(self.hashes[index]) if self.hashes is not None else None
                    if digest in known or digest not in originals:continue
                    locked=sum(r['locked'] for r in data['rows'])<4
                    data['rows'].append(dict(hash=digest,path=str(originals[digest].relative_to(folder)),locked=locked,
                        origin='manual',quality=1.,coverage=[0.,0.,.5],vector=self.anchors[index].tolist(),dhash=None))
                    known.add(digest);changed=True
                if changed:
                    data['active']=choose(data['rows'])
                    removed=[]
                    while len(data['rows'])>32:
                        candidates=[r for r in data['rows'] if not r['locked'] and r['hash'] not in data['active']]
                        old=min(candidates,key=lambda r:r['quality']);data['rows'].remove(old);removed.append(old)
                    data['revision']+=1;self.save(pid,data)
                    for old in removed:
                        if old['origin']=='automatic':(folder/old['path']).unlink(missing_ok=True)
            else:
                rows=[];indices=np.where(self.anchor_ids==pid)[0]
                photos=registration_photos(folder)
                by_hash={hashlib.sha256(p.read_bytes()).hexdigest():p for p in photos}
                for j,index in enumerate(indices):
                    digest=str(self.hashes[index]) if self.hashes is not None else None
                    if digest not in by_hash:continue  # No invented original/feature linkage.
                    rows.append(dict(hash=digest,path=str(by_hash[digest].relative_to(folder)),locked=j<4,
                        origin='manual',quality=1.,coverage=[0.,0.,.5],vector=self.anchors[index].tolist(),dhash=None))
                if not rows:continue
                rows=rows[:32];data=dict(person_id=pid,rows=rows,active=choose(rows),revision=1,model_sha256=self.c.get('sha256'))
                self.save(pid,data)
            self.pools[pid]=data;self.signatures[pid]=path.stat().st_mtime_ns
            profile=json.loads((folder/'profile.json').read_text());name=profile.get('display_name')
            if isinstance(name,str) and name and '/' not in name and name not in ('.','..'):
                aliases=self.directory.parent/'people_by_name';aliases.mkdir(exist_ok=True,mode=0o700)
                link=aliases/name
                target=os.path.relpath(folder,aliases)
                if link.is_symlink() and os.readlink(link)!=target:link.unlink()
                if not link.exists() and not link.is_symlink():link.symlink_to(target,target_is_directory=True)

    @staticmethod
    def validate(data,pid):
        if (data.get('person_id')!=pid or not isinstance(data.get('rows'),list) or len(data['rows'])>32
                or not 1<=len(data.get('active',[]))<=16):raise ValueError('Invalid photo pool')
        hashes={r['hash'] for r in data['rows']}
        if (len(hashes)!=len(data['rows']) or len(set(data['active']))!=len(data['active'])
                or not set(data['active'])<=hashes):raise ValueError('Duplicate or missing pool images')
        locked={r['hash'] for r in data['rows'] if r['locked']}
        if not 1<=len(locked)<=4 or not locked<=set(data['active']):raise ValueError('Locked bases must stay active')
        for row in data['rows']:
            if (Path(row['path']).is_absolute() or '..' in Path(row['path']).parts
                    or len(row['vector']) not in (128,1024) or not np.isfinite(row['vector']).all()):raise ValueError('Invalid pool sample')
            if row.get('origin')=='automatic' and (Path(row['path']).parent!=Path('photo_pool') or Path(row['path']).suffix!='.jpg'):
                raise ValueError('Automatic photos must stay in the managed subdirectory')

    def save(self,pid,data,expected_revision=None):
        self.validate(data,pid);path=self.directory/pid/'photo_pool.json';tmp=path.with_suffix('.tmp')
        import fcntl
        with (path.parent/'photo_pool.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if expected_revision is not None and json.loads(path.read_text())['revision']!=expected_revision:
                raise ValueError('Photo pool changed during collection; retained administrator update')
            fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            with os.fdopen(fd,'w') as stream:json.dump(data,stream,ensure_ascii=False)
            tmp.replace(path);self.version+=1

    def gallery(self):
        vectors=[];ids=[]
        for pid in sorted(set(self.anchor_ids)):
            pid=str(pid);data=self.pools.get(pid)
            if data:
                for row in data['rows']:
                    if row['hash'] in data['active']:vectors.append(row['vector']);ids.append(pid)
            else:
                for vector in self.anchors[self.anchor_ids==pid]:vectors.append(vector);ids.append(pid)
        return np.asarray(vectors,np.float32),np.asarray(ids)

    def refresh(self):
        # Administrator edits are loaded as a complete snapshot between inference
        # jobs. Invalid/partial files retain the last valid gallery.
        updates=[]
        for pid,old in list(self.pools.items()):
            path=self.directory/pid/'photo_pool.json';stamp=path.stat().st_mtime_ns
            if stamp==self.signatures.get(pid):continue
            data=json.loads(path.read_text());self.validate(data,pid)
            if data.get('model_sha256')!=self.c.get('sha256'):raise ValueError('Pool/model mismatch; rebuild explicitly')
            original={r['hash'] for r in old['rows'] if r['locked']}
            if not original<={r['hash'] for r in data['rows'] if r['locked']}:raise ValueError('Cannot unlock supplied base photos online')
            updates.append((pid,data,stamp))
        for pid,data,stamp in updates:
            self.pools[pid]=data;self.signatures[pid]=stamp;self.version+=1

    def consider(self,pid,image,row,vector,observation,compare,stamp):
        from vivis.vision.familiar import match_scores
        data=self.pools.get(pid)
        if not data or stamp-self.last.get(pid,-1e9)<5:return False
        if observation.get('quality_rejected') or observation.get('quality_low'):return False
        # Frozen manually supplied anchors, never auto-enrolled vectors, decide
        # eligibility. This prevents a feedback loop gradually adopting a stranger.
        accepted,_=match_scores([compare(vector,v) for v in self.anchors],self.anchor_ids,
            self.c['score_threshold'],self.c['minimum_margin'])
        if accepted!=pid:return False
        import cv2
        x,y,w,h=map(int,row[:4]);crop=image[max(0,y):min(image.shape[0],y+h),max(0,x):min(image.shape[1],x+w)]
        if min(crop.shape[:2],default=0)<90:return False
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);brightness=float(gray.mean())/255
        clarity=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        if not .15<brightness<.9 or clarity<45:return False
        small=cv2.resize(gray,(9,8));bits=(small[:,1:]>small[:,:-1]).reshape(-1)
        dhash=sum(int(v)<<i for i,v in enumerate(bits))
        if any(r.get('dhash') is not None and (int(r['dhash'])^dhash).bit_count()<8 for r in data['rows']):return False
        ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,92])
        if not ok:return False
        raw=encoded.tobytes();digest=hashlib.sha256(raw).hexdigest()
        if any(r['hash']==digest for r in data['rows']):return False
        pts=np.asarray(row[4:14]).reshape(5,2)
        eyes=(pts[0]+pts[1])/2
        coverage=[float((pts[2,0]-eyes[0])/max(w,1)),float((pts[2,1]-eyes[1])/max(h,1)),brightness]
        item=dict(hash=digest,path='photo_pool/'+digest+'.jpg',locked=False,origin='automatic',
            quality=min(clarity/300,1.),coverage=coverage,vector=np.asarray(vector).tolist(),dhash=str(dhash),
            face_bbox=list(map(float,row[:4])),created_monotonic=stamp)
        updated=dict(data,rows=[*data['rows'],item],revision=data['revision']+1)
        updated['active']=choose(updated['rows'])
        removed=[]
        while len(updated['rows'])>32:
            candidates=[r for r in updated['rows'] if r['hash'] not in updated['active'] and not r['locked']]
            worst=min(candidates,key=lambda r:r['quality']);updated['rows'].remove(worst);removed.append(worst)
        if digest not in {r['hash'] for r in updated['rows']}:return False
        path=self.directory/pid/item['path'];path.parent.mkdir(exist_ok=True,mode=0o700)
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(raw)
        try:self.save(pid,updated,expected_revision=data['revision'])
        except Exception:
            path.unlink(missing_ok=True);raise
        self.pools[pid]=updated;self.last[pid]=stamp
        for old in removed:
            if old['origin']=='automatic':(self.directory/pid/old['path']).unlink(missing_ok=True)
        return True
