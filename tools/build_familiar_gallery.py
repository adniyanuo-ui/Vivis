#!/usr/bin/env python3
"""Build one local identity backend's gallery from existing supplied photos."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vivis.paths import ROOT
from vivis.vision.identity_config import backend_config
from vivis.vision.enrollment import registration_image,extractor,extract,save_gallery

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('sface','seeta6'),default='seeta6')
    args=p.parse_args();c=json.loads((ROOT/'configs/perception/familiar_faces.json').read_text());c['backend']=args.backend
    selected=backend_config(c);model=extractor(c,ROOT);vectors=[];ids=[];hashes=[];quality=[]
    try:
        for profile in sorted((ROOT/c['people_directory']).glob('*/profile.json')):
            photos=[p for p in profile.parent.glob('photo.*') if p.suffix.lower() in ('.jpg','.jpeg','.png')]
            if len(photos)!=1:raise ValueError('Expected one original photo for '+profile.parent.name)
            image,row=registration_image(photos[0],ROOT);vector,observation=extract(model,c,image,row)
            vectors.append(vector);ids.append(profile.parent.name);hashes.append(hashlib.sha256(photos[0].read_bytes()).hexdigest())
            quality.append(dict(person_id=profile.parent.name,quality=observation))
        save_gallery(ROOT/selected['gallery'],c,vectors,ids,hashes)
    finally:
        if args.backend=='seeta6':model.close()
    out=ROOT/'outputs/face_ab';out.mkdir(parents=True,exist_ok=True)
    (out/('enrollment_'+args.backend+'.json')).write_text(json.dumps(quality,ensure_ascii=False,indent=2))
    print(json.dumps(dict(backend=args.backend,people=len(ids),gallery=selected['gallery'],other_galleries_unchanged=True,no_devices_opened=True),ensure_ascii=False))
if __name__=='__main__':main()
