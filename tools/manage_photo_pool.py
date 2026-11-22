#!/usr/bin/env python3
"""Explicit local administrator replacement within the 32/16 photo pool."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT
from vivis.interaction.memory import person_key
from vivis.vision.photo_pool import PhotoPool


def replace_active(path,person,activate,replace):
    import fcntl,os
    with (path.parent/'photo_pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(path.read_text());PhotoPool.validate(data,person)
        locked={r['hash'] for r in data['rows'] if r['locked']};hashes={r['hash'] for r in data['rows']}
        if replace in locked:raise ValueError('基础照片锁定，不能替换')
        if replace not in data['active'] or activate not in hashes or activate in data['active']:
            raise ValueError('请选择活动照片和候选照片的完整hash')
        data['active']=[activate if h==replace else h for h in data['active']];data['revision']+=1
        for row in data['rows']:
            if row['hash']==replace:row['excluded']=True
            elif row['hash']==activate:row['excluded']=False
        PhotoPool.validate(data,person);tmp=path.with_suffix('.tmp')
        fd=os.open(tmp,os.O_CREAT|os.O_WRONLY|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as f:json.dump(data,f,ensure_ascii=False)
        tmp.replace(path)
        return data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--person',required=True)
    p.add_argument('--activate');p.add_argument('--replace');args=p.parse_args();person_key(args.person)
    path=ROOT/'private/people'/args.person/'photo_pool.json';data=json.loads(path.read_text())
    PhotoPool.validate(data,args.person)
    if args.activate or args.replace:
        if not args.activate or not args.replace:p.error('Provide both --activate HASH and --replace HASH')
        try:data=replace_active(path,args.person,args.activate,args.replace)
        except ValueError as exc:p.error(str(exc))
    print(json.dumps(dict(person_id=args.person,active=data['active'],photos=[{k:r[k] for k in ('hash','path','locked','origin')} for r in data['rows']]),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
