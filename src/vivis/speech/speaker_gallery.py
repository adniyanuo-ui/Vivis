"""Bounded reference metadata with atomic revisions shared by UI and observer."""
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import time

from vivis.interaction.memory import person_key


class RevisionConflict(ValueError):pass


def sample_key(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}',value):raise ValueError('无效录音编号')
    return value


def read_gallery(path,model_hash):
    path=Path(path)
    data=json.loads(path.read_text()) if path.exists() else dict(model_sha256=model_hash,revision=0,people={})
    if (data.get('model_sha256')!=model_hash or not isinstance(data.get('people'),dict)
            or type(data.get('revision',0)) is not int or data.get('revision',0)<0):raise ValueError('Speaker gallery/model mismatch')
    data.setdefault('revision',0)
    for pid,rows in data['people'].items():
        person_key(pid)
        if not isinstance(rows,list) or len(rows)>32:raise ValueError('Invalid speaker reference count')
        ids=[]
        for row in rows:
            ids.append(sample_key(row['turn_id']))
            if not isinstance(row.get('vector'),list) or not row['vector']:raise ValueError('Invalid speaker feature')
            if type(row.get('active',True)) is not bool or type(row.get('locked',False)) is not bool:raise ValueError('Invalid speaker metadata')
        if len(set(ids))!=len(ids):raise ValueError('Duplicate speaker reference')
    return data


def commit(path,data):
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.speaker-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(data,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


def save_gallery(path,model_hash,people,expected_revision):
    path=Path(path)
    with (path.parent/'gallery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=read_gallery(path,model_hash)
        if current['revision']!=expected_revision:raise RevisionConflict('声纹样本已变化，请刷新后重试')
        updated=dict(model_sha256=model_hash,revision=expected_revision+1,people=people)
        commit(path,updated)
        return updated


def edit_reference(path,model_hash,person,sample,action,expected_revision):
    path=Path(path);person_key(person);sample_key(sample)
    if action not in ('activate','deactivate','delete','lock','unlock'):raise ValueError('无效声纹操作')
    with (path.parent/'gallery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=read_gallery(path,model_hash)
        if data['revision']!=expected_revision:raise RevisionConflict('声纹样本已变化，请刷新后重试')
        rows=data['people'].get(person,[]);row=next((r for r in rows if r['turn_id']==sample),None)
        if row is None:raise ValueError('录音已不存在，请刷新')
        media=(path.parent/person/(sample+'.wav')).resolve()
        if not media.is_relative_to(path.parent.resolve()):raise ValueError('录音路径越界')
        if action=='activate':
            row.update(active=True,locked=True,administrator_verified=True,reviewed_at=time.time(),candidate_reason=None)
            if row.get('origin')=='onsite_label_pending_review':row['origin']='manual_review'
        elif action=='deactivate':row.update(active=False,locked=False)
        elif action=='lock':
            if not row.get('active',True):raise ValueError('请先激活该录音')
            row['locked']=True
        elif action=='unlock':row['locked']=False
        elif action=='delete':rows.remove(row)
        data['revision']+=1;commit(path,data)
        if action=='delete':media.unlink(missing_ok=True)
        return data
