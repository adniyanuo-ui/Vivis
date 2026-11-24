#!/usr/bin/env python3
"""Read-only continuity/photo/speaker status. Does not open hardware or cloud."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--person');p.add_argument('--last',type=int,default=3);args=p.parse_args()
    if args.person:
        from vivis.interaction.memory import person_key
        person_key(args.person)
    if not 1<=args.last<=20:p.error('--last must be 1..20')
    path=ROOT/'private/continuity/rounds.sqlite3';data=dict(continuity_exists=path.exists(),photos=[],speakers={})
    if path.exists():
        db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        data['round_counts']=[dict(r) for r in db.execute('SELECT person,count(*) rounds,sum(archived) archived FROM rounds GROUP BY person')]
        data['jobs']=[dict(r) for r in db.execute('SELECT id,requested,state,error,notice FROM jobs ORDER BY requested DESC LIMIT 5')]
        clause=' WHERE person=?' if args.person else '';params=(args.person,args.last) if args.person else (args.last,)
        rows=[dict(r) for r in db.execute('SELECT * FROM rounds'+clause+' ORDER BY created DESC LIMIT ?',params)]
        for r in rows:r['replies']=[dict(a) for a in db.execute('SELECT id,text,samples,recorded_samples,audio_status,outcome,audio FROM replies WHERE round_id=?',(r['id'],))]
        data['recent_rounds']=rows;db.close()
    for file in sorted((ROOT/'private/people').glob('*/photo_pool.json')):
        c=json.loads(file.read_text());data['photos'].append(dict(person_id=c['person_id'],originals=len(c['rows']),
            locked=sum(r['locked'] for r in c['rows']),active=len(c['active']),candidates=len(c['rows'])-len(c['active']),
            sampling_complete=len(c['rows'])>=32,vacant_slots=max(0,32-len(c['rows'])),manifest=str(file)))
    path=ROOT/'private/speakers/gallery.json'
    if path.exists():data['speakers']={p:dict(samples=len(v),active=sum(r.get('active',True) for r in v),locked=sum(r.get('locked',False) for r in v),sampling_complete=len(v)>=32,vacant_slots=max(0,32-len(v)),observation_only=True) for p,v in json.loads(path.read_text())['people'].items()}
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
