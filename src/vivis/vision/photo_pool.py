"""Bounded, anchor-verified photo pool; never enroll from held identity alone."""
import hashlib
import json
import os
from pathlib import Path
from datetime import date
from functools import wraps
import threading

import numpy as np
from vivis.vision.photo_sampling import plan,apply_batch


def synchronized(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self.lock:return method(self,*args,**kwargs)
    return call


def opportunistic(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        if not self.lock.acquire(blocking=False):return False
        try:return method(self,*args,**kwargs)
        finally:self.lock.release()
    return call


def diverse(rows,context,count):
    selected=[];remaining=list(rows)
    while remaining and len(selected)<count:
        def merit(row):
            distance=min((np.linalg.norm(np.asarray(row['coverage'])-r['coverage']) for r in [*context,*selected]),default=1.)
            return row['quality']+.8*min(float(distance),2.)
        best=max(remaining,key=merit);selected.append(best);remaining.remove(best)
    return selected


def choose(rows,limit=16):
    locked=[r for r in rows if r['locked']]
    if len(locked)>limit:raise ValueError('Locked photos must fit the sixteen active slots')
    remaining=[r for r in rows if not r['locked'] and not r.get('excluded',False)]
    selected=[*locked,*diverse(remaining,locked,limit-len(locked))]
    return [r['hash'] for r in selected]


class PhotoPool:
    def __init__(self,root,config,vectors,ids,hashes=None,*,day=None,timer_factory=None):
        self.root=Path(root);self.c=config
        self.directory=self.root/config['people_directory']
        if not self.directory.resolve().is_relative_to((self.root/'private').resolve()):raise ValueError('Private photo pool required')
        self.anchors=np.asarray(vectors).copy();self.anchor_ids=np.asarray(ids).copy()
        self.hashes=hashes;self.version=0
        self.lock=threading.RLock();self.closed=False;self.pending={};self.error=None
        self.day=day or (lambda:date.today().isoformat());self.timer_factory=timer_factory or threading.Timer
        self.sampling=dict(stable_s=2.,batch_window_s=30.,sample_interval_s=.75,
            min_face_size_px=90,brightness_range=[.15,.9],min_laplacian_variance=45.)
        self.sampling.update(config.get('photo_sampling') or {})
        for name,low,high in (('stable_s',1.,10.),('batch_window_s',5.,60.),('sample_interval_s',.25,5.)):
            value=self.sampling.get(name)
            if type(value) not in (int,float) or not np.isfinite(value) or not low<=value<=high:raise ValueError('Invalid photo sampling configuration')
        size=self.sampling['min_face_size_px'];brightness=self.sampling['brightness_range'];clarity=self.sampling['min_laplacian_variance']
        if (type(size) is not int or not 40<=size<=512 or not isinstance(brightness,(list,tuple)) or len(brightness)!=2
                or any(type(v) not in (int,float) or not np.isfinite(v) for v in brightness)
                or not 0<=brightness[0]<brightness[1]<=1 or type(clarity) not in (int,float)
                or not np.isfinite(clarity) or not 1<=clarity<=10000):raise ValueError('Invalid photo collection quality thresholds')
        self.stable_person=None;self.stable_since=0.;self.stable_last=0.
        self.pools={};self.signatures={};self.last_attempt=None
        self.load()

    @synchronized
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
                # A filesystem deletion becomes effective on the next restart.
                # The administrator command also updates the live manifest.
                missing={r['hash'] for r in data['rows'] if r['origin']=='automatic' and not (folder/r['path']).is_file()}
                if missing:
                    data['rows']=[r for r in data['rows'] if r['hash'] not in missing]
                    data['active']=[h for h in data['active'] if h not in missing]
                known={r['hash'] for r in data['rows']};changed=bool(missing)
                for index in np.where(self.anchor_ids==pid)[0]:
                    digest=str(self.hashes[index]) if self.hashes is not None else None
                    if digest in known or digest not in originals:continue
                    data['rows'].append(dict(hash=digest,path=str(originals[digest].relative_to(folder)),locked=True,lock_reason='manual',
                        origin='manual',quality=1.,coverage=[0.,0.,.5],vector=self.anchors[index].tolist(),dhash=None))
                    known.add(digest);changed=True
                for sample in data['rows']:
                    if sample['origin']=='manual' and not sample['locked']:
                        sample.update(locked=True,lock_reason='manual',excluded=False);changed=True
                    elif sample['origin']=='automatic' and not sample['locked'] and sample.get('excluded') is False and sample.get('lock_reason')!='administrator_unlocked':
                        # The previous administrator tool wrote explicit False
                        # only on activation; ordinary automatic rows omit it.
                        sample.update(locked=True,lock_reason='administrator');changed=True
                if changed:
                    data['active']=choose([r for r in data['rows'] if r['locked'] or r['hash'] in data['active']])
                    removed=[]
                    while len(data['rows'])>32:
                        candidates=[r for r in data['rows'] if not r['locked'] and r['hash'] not in data['active']]
                        old=min(candidates,key=lambda r:r['quality']);data['rows'].remove(old);removed.append(old)
                    from vivis.vision.photo_sampling import administrator_change
                    administrator_change(data)
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
                    rows.append(dict(hash=digest,path=str(by_hash[digest].relative_to(folder)),locked=True,lock_reason='base' if j<4 else 'manual',
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
        if not 1<=len(locked)<=16 or not locked<=set(data['active']):raise ValueError('Locked photos must stay active')
        if any(r.get('excluded',False) for r in data['rows'] if r['locked']):raise ValueError('Locked photos cannot be excluded')
        state=data.get('collection')
        if state is not None:
            if state.get('policy')!='fill-once-1' or type(state.get('completed')) is not bool:
                raise ValueError('Invalid photo collection state')
            if state.get('last_batch_day') is not None:date.fromisoformat(state['last_batch_day'])
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
            with os.fdopen(fd,'w') as stream:
                json.dump(data,stream,ensure_ascii=False);stream.flush();os.fsync(stream.fileno())
            tmp.replace(path);self.version+=1

    def gallery(self):
        # A timer may be writing files. Recognition reads the last published
        # immutable per-person snapshots without waiting for that disk work.
        vectors=[];ids=[];pools=dict(self.pools)
        for pid in sorted(set(self.anchor_ids)):
            pid=str(pid);data=pools.get(pid)
            if data:
                for row in data['rows']:
                    if row['hash'] in data['active']:vectors.append(row['vector']);ids.append(pid)
            else:
                for vector in self.anchors[self.anchor_ids==pid]:vectors.append(vector);ids.append(pid)
        return np.asarray(vectors,np.float32),np.asarray(ids)

    @synchronized
    def add_anchors(self,pid,rows):
        """Append an explicitly introduced base without changing old anchors."""
        import re
        if not re.fullmatch(r'[a-z][a-z0-9_-]{1,39}',pid):raise ValueError('Invalid introduced person ID')
        if pid in self.anchor_ids:return
        data=json.loads((self.directory/pid/'photo_pool.json').read_text());self.validate(data,pid)
        if data['model_sha256']!=self.c['sha256']:raise ValueError('Introduced model mismatch')
        vectors=np.asarray([r['vector'] for r in rows],np.float32)
        if vectors.shape!=(len(rows),self.anchors.shape[1]) or not 1<=len(rows)<=4:
            raise ValueError('Invalid introduced anchor dimensions')
        self.anchors=np.concatenate([self.anchors,vectors])
        self.anchor_ids=np.concatenate([self.anchor_ids,np.asarray([pid]*len(rows))])
        self.hashes=np.concatenate([self.hashes,np.asarray([r['hash'] for r in rows])])
        self.load()

    @opportunistic
    def refresh(self):
        # Administrator edits are loaded as a complete snapshot between inference
        # jobs. Invalid/partial files retain the last valid gallery.
        updates=[]
        for pid,old in list(self.pools.items()):
            path=self.directory/pid/'photo_pool.json';stamp=path.stat().st_mtime_ns
            if stamp==self.signatures.get(pid):continue
            data=json.loads(path.read_text());self.validate(data,pid)
            if data.get('model_sha256')!=self.c.get('sha256'):raise ValueError('Pool/model mismatch; rebuild explicitly')
            original={r['hash'] for r in old['rows'] if r['origin']=='manual' and r['locked']}
            if not original<={r['hash'] for r in data['rows'] if r['locked']}:raise ValueError('Cannot unlock supplied base photos online')
            updates.append((pid,data,stamp))
        for pid,data,stamp in updates:
            self.pools[pid]=data;self.signatures[pid]=stamp;self.version+=1

    @opportunistic
    def observe_identity(self,pid,stamp):
        # This is collection evidence only. It never changes identity results.
        if pid is None or pid!=self.stable_person or not 0<=stamp-self.stable_last<=self.c.get('max_result_age_s',1.5):
            self.stable_person=pid;self.stable_since=stamp
        self.stable_last=stamp

    @opportunistic
    def consider(self,pid,image,row,vector,observation,compare,stamp):
        from vivis.vision.familiar import match_scores
        data=self.pools.get(pid)
        details={}
        def decision(reason,accepted=False,**values):
            self.last_attempt=dict(person_id=pid,observed_monotonic=stamp,reason=reason,
                staged=accepted,**details,**values)
            return accepted
        if (self.closed or not data or self.stable_person!=pid
                or not 0<=stamp-self.stable_last<=self.c.get('max_result_age_s',1.5)
                or stamp-self.stable_since<self.sampling['stable_s']):return decision('identity_not_stable')
        batch=self.pending.get(pid)
        if batch and stamp-batch['last_sample']<self.sampling['sample_interval_s']:return decision('sample_interval')
        work=batch['work'] if batch else plan(data,self.day())
        if not work:return decision('pool_full')
        target=work['active']+work['reserve']
        if batch and len(batch['samples'])>=target:return decision('batch_full')
        if observation.get('quality_rejected') or observation.get('quality_low'):return decision('native_quality_low')
        # Frozen manually supplied anchors, never auto-enrolled vectors, decide
        # eligibility. This prevents a feedback loop gradually adopting a stranger.
        accepted,_=match_scores([compare(vector,v) for v in self.anchors],self.anchor_ids,
            self.c['score_threshold'],self.c['minimum_margin'])
        if accepted!=pid:return decision('manual_anchor_mismatch')
        import cv2
        x,y,w,h=map(int,row[:4]);crop=image[max(0,y):min(image.shape[0],y+h),max(0,x):min(image.shape[1],x+w)]
        details.update(crop_size_px=list(crop.shape[:2]))
        minimum_size=self.sampling['min_face_size_px']
        if min(crop.shape[:2],default=0)<minimum_size:return decision('face_too_small',minimum_size_px=minimum_size)
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);brightness=float(gray.mean())/255
        clarity=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        details.update(brightness=brightness,laplacian_variance=clarity)
        low,high=self.sampling['brightness_range'];minimum_clarity=self.sampling['min_laplacian_variance']
        if not low<brightness<high or clarity<minimum_clarity:
            return decision('image_quality_low',rejected_by=(
                ([] if low<brightness<high else ['brightness'])+([] if clarity>=minimum_clarity else ['clarity'])),
                brightness_range=[low,high],minimum_laplacian_variance=minimum_clarity)
        small=cv2.resize(gray,(9,8));bits=(small[:,1:]>small[:,:-1]).reshape(-1)
        dhash=sum(int(v)<<i for i,v in enumerate(bits))
        existing=[*data['rows'],*(s[0] for s in batch['samples'])] if batch else data['rows']
        if any(r.get('dhash') is not None and (int(r['dhash'])^dhash).bit_count()<8 for r in existing):return decision('near_duplicate')
        ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,92])
        if not ok:return decision('encoding_failed')
        raw=encoded.tobytes();digest=hashlib.sha256(raw).hexdigest()
        if any(r['hash']==digest for r in existing):return decision('duplicate_hash')
        pts=np.asarray(row[4:14]).reshape(5,2)
        eyes=(pts[0]+pts[1])/2
        coverage=[float((pts[2,0]-eyes[0])/max(w,1)),float((pts[2,1]-eyes[1])/max(h,1)),brightness]
        item=dict(hash=digest,path='photo_pool/'+digest+'.jpg',locked=False,origin='automatic',
            quality=min(clarity/300,1.),coverage=coverage,vector=np.asarray(vector).tolist(),dhash=str(dhash),
            face_bbox=list(map(float,row[:4])),created_monotonic=stamp,created_day=work['day'])
        if batch is None:
            token=object()
            timer=self.timer_factory(self.sampling['batch_window_s'],self._finish_safely,args=(pid,token))
            timer.daemon=True
            batch=dict(work=work,revision=data['revision'],samples=[],last_sample=stamp,timer=timer,token=token)
            self.pending[pid]=batch;timer.start()
        batch['samples'].append((item,raw));batch['last_sample']=stamp
        if len(batch['samples'])>=target:
            batch['timer'].cancel()
            timer=self.timer_factory(0,self._finish_safely,args=(pid,batch['token']))
            timer.daemon=True;batch['timer']=timer;timer.start()
        return decision('staged_for_batch',accepted=True,pending_samples=len(batch['samples']))

    def _finish_safely(self,pid,token=None):
        try:self.finish_batch(pid,token)
        except Exception as exc:self.error='Photo batch update failed: '+type(exc).__name__

    @synchronized
    def finish_batch(self,pid,token=None):
        batch=self.pending.get(pid)
        if batch is None or token is not None and batch['token'] is not token:return False
        self.pending.pop(pid)
        batch['timer'].cancel();self.refresh();data=self.pools[pid]
        if data['revision']!=batch['revision']:return False  # Keep any administrator change.
        work=batch['work'];count=min(work['active']+work['reserve'],len(batch['samples']))
        selected=diverse([s[0] for s in batch['samples']],data['rows'],count)
        updated=apply_batch(data,selected,work)
        if updated is None:return False
        raw_by_hash={r['hash']:raw for r,raw in batch['samples']};written=[]
        try:
            for item in selected:
                path=self.directory/pid/item['path'];path.parent.mkdir(exist_ok=True,mode=0o700)
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                written.append(path)
                with os.fdopen(fd,'wb') as stream:
                    stream.write(raw_by_hash[item['hash']]);stream.flush();os.fsync(stream.fileno())
            self.save(pid,updated,expected_revision=data['revision'])
        except Exception:
            for path in written:path.unlink(missing_ok=True)
            raise
        self.pools[pid]=updated
        self.signatures[pid]=(self.directory/pid/'photo_pool.json').stat().st_mtime_ns
        self.error=None
        return True

    @synchronized
    def close(self):
        self.closed=True
        for pid in list(self.pending):self._finish_safely(pid)
