#!/usr/bin/env python3
"""Enroll a supplied labelled photo and selected work background locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import cv2
    import numpy as np
    from vivis.interaction.knowledge import professional_profile
    from vivis.vision.familiar import feature
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id',required=True);p.add_argument('--name',required=True);p.add_argument('--address',required=True)
    p.add_argument('--age-years',type=int);p.add_argument('--role',required=True);p.add_argument('--organization',required=True)
    p.add_argument('--photo',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--professional-summary',default='');p.add_argument('--allow-cloud-context',action='store_true')
    args=p.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_-]{1,39}',args.id):p.error('Invalid local person ID')
    c=json.loads((ROOT/'configs/perception/familiar_faces.json').read_text());model=ROOT/c['model']
    if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest()!=c['sha256']:p.error('Run python3 tools/setup_familiar_faces.py first')
    profile=dict(schema_version='vivis-person-1.0',display_name=args.name,preferred_address=args.address,
        organization=args.organization,role=args.role,professional_summary=args.professional_summary,
        allow_cloud_context=args.allow_cloud_context,source='user-supplied labelled photograph and professional context')
    previous=ROOT/c['people_directory']/args.id/'profile.json'
    age=args.age_years
    if age is None and previous.is_file():age=json.loads(previous.read_text()).get('age_years')
    if age is not None:profile['age_years']=age
    professional_profile(profile)
    cv2.setNumThreads(1)
    image=cv2.imread(str(args.photo))
    if image is None:p.error('Cannot decode supplied photo')
    h,w=image.shape[:2];scale=min(1.,1280/max(h,w));image=cv2.resize(image,(round(w*scale),round(h*scale)))
    detector=cv2.FaceDetectorYN.create(str(ROOT/'assets/models/yunet/face_detection_yunet_2023mar.onnx'),'',(image.shape[1],image.shape[0]),.85,.3,5000)
    _,faces=detector.detect(image)
    if faces is None or len(faces)!=1:p.error('Enrollment requires exactly one visible face; use a single-person photo')
    recognizer=cv2.FaceRecognizerSF.create(str(model),'');vector=feature(recognizer,image,faces[0])
    base=ROOT/c['people_directory'];base.mkdir(parents=True,exist_ok=True);os.chmod(base,0o700)
    folder=base/args.id;folder.mkdir(exist_ok=True);os.chmod(folder,0o700)
    for source,name in [(args.photo,'photo'+args.photo.suffix.lower()),(args.resume,'resume'+args.resume.suffix.lower() if args.resume else '')]:
        if source is not None:
            target=folder/name;shutil.copy2(source,target);os.chmod(target,0o600)
    if args.resume and args.resume.suffix.lower()=='.docx':
        with zipfile.ZipFile(args.resume) as z:root=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text='\n'.join(''.join(t.text or '' for t in row.findall('.//w:t',ns)) for row in root.findall('.//w:p',ns))
        target=folder/'resume.local.txt';target.write_text(text);os.chmod(target,0o600)
    target=folder/'profile.json';target.write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n');os.chmod(target,0o600)
    gallery=ROOT/c['gallery'];vectors=[];ids=[]
    if gallery.exists():
        with np.load(gallery,allow_pickle=False) as old:
            for pid,v in zip(old['ids'],old['vectors']):
                if pid!=args.id:ids.append(str(pid));vectors.append(v)
    vectors.append(vector);ids.append(args.id)
    temp=gallery.with_suffix('.tmp.npz');np.savez(temp,vectors=np.asarray(vectors,np.float32),ids=np.asarray(ids));os.chmod(temp,0o600);temp.replace(gallery)
    print(json.dumps(dict(enrolled_id=args.id,visible_faces=1,local_gallery_size=len(ids),raw_files_uploaded=False,cloud_professional_context=args.allow_cloud_context),ensure_ascii=False))


if __name__=='__main__':main()
