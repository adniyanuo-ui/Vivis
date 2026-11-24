#!/usr/bin/env python3
"""Local administrator activation, deactivation and replacement of photo samples."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT
from vivis.interaction.memory import person_key
from vivis.vision.photo_pool import PhotoPool
from vivis.vision.photo_sampling import administrator_change


from vivis.vision.photo_management import edit_active


def replace_active(path,person,activate,replace):
    return edit_active(path,person,activate=activate,replace=replace)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--person',required=True)
    p.add_argument('--activate');p.add_argument('--replace');p.add_argument('--deactivate');p.add_argument('--delete');p.add_argument('--lock',dest='lock_photo');args=p.parse_args();person_key(args.person)
    path=ROOT/'private/people'/args.person/'photo_pool.json';data=json.loads(path.read_text())
    PhotoPool.validate(data,args.person)
    if args.activate or args.replace or args.deactivate or args.delete or args.lock_photo:
        try:data=edit_active(path,args.person,activate=args.activate,replace=args.replace,deactivate=args.deactivate,delete=args.delete,lock_photo=args.lock_photo)
        except ValueError as exc:p.error(str(exc))
    print(json.dumps(dict(person_id=args.person,active=data['active'],collection=data.get('collection'),photos=[{**{k:r[k] for k in ('hash','path','locked','origin')},'excluded':r.get('excluded',False)} for r in data['rows']]),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
