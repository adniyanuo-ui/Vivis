#!/usr/bin/env python3
"""Append local photo templates to an existing person; preserve profile and other IDs."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT
from vivis.vision.enrollment import registration_image,registration_photos,extractor,extract,save_gallery,SEETA_CONTRACT
from vivis.vision.identity_config import backend_config


def main():
    import cv2
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id',required=True);p.add_argument('--photo',type=Path,action='append',default=[])
    p.add_argument('--check',action='store_true',help='Extract/check only; do not change gallery or photos')
    args=p.parse_args();c=json.loads((ROOT/'configs/perception/familiar_faces.json').read_text());selected=backend_config(c)
    folder=ROOT/c['people_directory']/args.id
    if folder.parent!=ROOT/c['people_directory'] or not (folder/'profile.json').is_file():p.error('Existing registered person required')
    gallery=ROOT/selected['gallery']
    with np.load(gallery,allow_pickle=False) as old:
        if str(old['backend'])!=selected['backend'] or str(old['model_sha256'])!=selected['sha256']:
            raise ValueError('Gallery backend/model mismatch')
        expected=SEETA_CONTRACT if selected['backend']=='seeta6' else 'opencv-sface-128'
        if str(old['embedding_contract'])!=expected:raise ValueError('Gallery alignment mismatch')
        vectors=list(old['vectors']);ids=list(map(str,old['ids']));hashes=list(map(str,old['photo_sha256']))
    cv2.setNumThreads(1);model=extractor(c,ROOT);added=[]
    try:
        for photo in [*registration_photos(folder),*args.photo]:
            digest=hashlib.sha256(photo.read_bytes()).hexdigest()
            if any(pid==args.id and h==digest for pid,h in zip(ids,hashes)):continue
            image,row=registration_image(photo,ROOT);vector,quality=extract(model,c,image,row)
            vectors.append(vector);ids.append(args.id);hashes.append(digest)
            added.append(dict(source=str(photo.resolve()),sha256=digest,quality=quality))
        # Keep all old vectors byte-identical and compare supplied photos against
        # every person, using the same per-person max and margin as runtime.
        from vivis.vision.familiar import match_scores
        checks=[]
        for photo in [*registration_photos(folder),*args.photo]:
            digest=hashlib.sha256(photo.read_bytes()).hexdigest()
            index=next(i for i,(pid,h) in enumerate(zip(ids,hashes)) if pid==args.id and h==digest)
            v=vectors[index]
            scores=np.array([model.compare(v,np.asarray(w,np.float32)) for w in vectors]) if selected['backend']=='seeta6' else np.asarray(vectors)@v
            details={};matched,score=match_scores(scores,np.array(ids),selected['score_threshold'],c['minimum_margin'],details)
            check=dict(photo=photo.name,matched_id=matched,score=score,details=details)
            if ids.count(args.id)>1:
                mask=np.arange(len(ids))!=index;held={}
                held_id,held_score=match_scores(scores[mask],np.array(ids)[mask],selected['score_threshold'],c['minimum_margin'],held)
                check['without_self']=dict(matched_id=held_id,score=held_score,details=held)
                if held_id!=args.id:raise ValueError('Photo not confirmed by other supplied templates: '+photo.name)
            checks.append(check)
            if matched!=args.id:raise ValueError('Supplied photo conflicts with gallery: '+photo.name)
        if not args.check and added:
            extra=folder/'photos';extra.mkdir(exist_ok=True,mode=0o700)
            for record in added:
                source=Path(record['source']);target=extra/(record['sha256']+source.suffix.lower())
                if source.resolve().is_relative_to(folder.resolve()):continue
                shutil.copy2(source,target);os.chmod(target,0o600)
            save_gallery(gallery,c,vectors,ids,hashes)
        report=dict(person_id=args.id,backend=selected['backend'],check_only=args.check,added=len(added),
            templates=dict(Counter(ids)),other_vectors_unchanged=True,checks=checks,sources=added,
            raw_files_uploaded=False,onsite_verified=False)
        if not args.check:
            target=folder/'multiphoto_enrollment.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');os.chmod(target,0o600)
        print(json.dumps(report,ensure_ascii=False))
    finally:
        if selected['backend']=='seeta6':model.close()

if __name__=='__main__':main()
