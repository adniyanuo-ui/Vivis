"""Local backend-specific enrollment; original photos and other galleries stay intact."""
import hashlib
import os
from pathlib import Path
import re
import shutil
from datetime import datetime

import numpy as np
from vivis.vision.identity_config import backend_config

SEETA_CONTRACT='seeta6-yunet-roi-pts5-v1'

def registration_image(path,root):
    import cv2
    image=cv2.imread(str(path))
    if image is None:raise ValueError('Cannot decode registration photo')
    h,w=image.shape[:2];scale=min(1.,1280/max(h,w))
    image=cv2.resize(image,(round(w*scale),round(h*scale)))
    detector=cv2.FaceDetectorYN.create(str(root/'assets/models/yunet/face_detection_yunet_2023mar.onnx'),'',
        (image.shape[1],image.shape[0]),.85,.3,5000)
    _,faces=detector.detect(image)
    if faces is None or len(faces)!=1:raise ValueError('Registration requires exactly one face')
    return image,faces[0]

def extractor(config,root):
    import cv2
    c=backend_config(config);model=root/c['model']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Registration model hash mismatch')
    if c['backend']=='seeta6':
        from vivis.vision.seeta import Seeta
        return Seeta(root=root,age=False)
    return cv2.FaceRecognizerSF.create(str(model),'')

def extract(model,config,image,row):
    c=backend_config(config)
    if c['backend']=='seeta6':
        # Retain supplied historical enrollment photos for a fair migration.
        # Record their quality but do not silently discard registered colleagues.
        observation,vector=model.evaluate_selected(image,row,gate=False)
        if vector is None:raise ValueError('Registration feature extraction failed')
        return vector,observation
    from vivis.vision.familiar import feature
    return feature(model,image,row),{}

def save_gallery(path,config,vectors,ids,photo_hashes):
    c=backend_config(config);path=Path(path)
    if not ids or len(vectors)!=len(ids) or len(photo_hashes)!=len(ids):raise ValueError('Invalid gallery rows')
    if len(set(ids))!=len(ids) or any(not re.fullmatch(r'[a-z][a-z0-9_-]{1,39}',pid) for pid in ids):raise ValueError('Invalid gallery IDs')
    vectors=np.asarray(vectors,np.float32);dimension=1024 if c['backend']=='seeta6' else 128
    if vectors.shape!=(len(ids),dimension) or not np.isfinite(vectors).all():raise ValueError('Invalid gallery dimensions')
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if path.exists():
        backup=path.parent/'.backups';backup.mkdir(exist_ok=True,mode=0o700)
        target=backup/(path.stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.npz')
        shutil.copy2(path,target);os.chmod(target,0o600)
    temporary=path.with_name(path.stem+'.tmp.npz')
    try:
        np.savez(temporary,vectors=vectors,ids=np.asarray(ids),backend=np.asarray(c['backend']),
            model_sha256=np.asarray(c['sha256']),photo_sha256=np.asarray(photo_hashes),
            embedding_contract=np.asarray(SEETA_CONTRACT if c['backend']=='seeta6' else 'opencv-sface-128'))
        os.chmod(temporary,0o600);temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)
