"""Pinned local SeetaFace6 transport; no camera, cloud or actuator access."""
import ctypes
import hashlib
import json
from pathlib import Path
import numpy as np
from vivis.paths import ROOT

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()

class Seeta:
    def __init__(self,threads=1,*,root=ROOT,age=False):
        root=Path(root);manifest=json.loads((root/'native/vision/seeta/sources.json').read_text())
        for row in manifest['models']:
            if row.get('optional')=='age_estimation' and not age:continue
            if digest(root/'outputs/face_ab/models'/row['name'])!=row['sha256']:raise ValueError('SeetaFace model hash mismatch')
        self.lib=ctypes.CDLL(str(root/'outputs/face_ab/build/probe/libvivis_seeta_probe.so'))
        lib=self.lib;pointer=ctypes.c_void_p
        lib.sf6_create.argtypes=[ctypes.c_char_p,ctypes.c_int];lib.sf6_create.restype=pointer
        lib.sf6_error.restype=ctypes.c_char_p
        lib.sf6_destroy.argtypes=[pointer]
        lib.sf6_destroy.restype=None
        lib.sf6_dimension.argtypes=[pointer];lib.sf6_dimension.restype=ctypes.c_int
        lib.sf6_eval.argtypes=[pointer,pointer,ctypes.c_int,ctypes.c_int,ctypes.c_int,pointer,ctypes.c_int,pointer]
        lib.sf6_similarity.argtypes=[pointer,pointer,pointer];lib.sf6_similarity.restype=ctypes.c_float
        lib.sf6_eval_face.argtypes=[pointer,pointer,ctypes.c_int,ctypes.c_int,pointer,ctypes.c_int,pointer,ctypes.c_int,pointer,pointer]
        lib.sf6_enable_age.argtypes=[pointer];lib.sf6_enable_age.restype=ctypes.c_int
        self.handle=lib.sf6_create(str(root/'outputs/face_ab/models').encode(),threads)
        if not self.handle:raise RuntimeError(lib.sf6_error().decode())
        self.dimension=lib.sf6_dimension(self.handle)
        if age and lib.sf6_enable_age(self.handle):
            reason=lib.sf6_error().decode();self.close();raise RuntimeError(reason)

    @staticmethod
    def observation(meta,gate):
        return dict(faces=int(meta[0]),bbox=meta[1:5].tolist() if meta[0] else None,
            detection_score=float(meta[5]) if meta[0] else None,
            brightness=dict(score=float(meta[6]),level=int(meta[7])) if meta[0] else None,
            clarity=dict(score=float(meta[8]),level=int(meta[9])) if meta[0] else None,
            timing_ms=dict(zip(('detection','landmarks','quality','recognition','native_total'),map(float,meta[10:15]))),
            quality_low=bool(meta[0] and (meta[7]==0 or meta[9]==0)),
            quality_rejected=bool(gate and meta[0] and (meta[7]==0 or meta[9]==0)))

    def evaluate(self,image,gate=False):
        image=np.ascontiguousarray(image,dtype=np.uint8);meta=np.zeros(16,dtype=np.float64)
        features=np.zeros(self.dimension,dtype=np.float32)
        code=self.lib.sf6_eval(self.handle,image.ctypes.data,image.shape[1],image.shape[0],int(gate),
                              features.ctypes.data,self.dimension,meta.ctypes.data)
        if code:raise RuntimeError(self.lib.sf6_error().decode())
        if not np.isfinite(meta).all() or not np.isfinite(features).all():raise ValueError('Non-finite SeetaFace output')
        result=self.observation(meta,gate)
        return result,features if meta[15] else None

    def evaluate_selected(self,image,row,gate=True):
        image=np.ascontiguousarray(image,dtype=np.uint8);box=np.ascontiguousarray(row[:4],dtype=np.float64)
        if box.shape!=(4,) or not np.isfinite(box).all():raise ValueError('Invalid selected-face rectangle')
        features=np.zeros(self.dimension,np.float32);meta=np.zeros(16,np.float64);attributes=np.array([-1.,0.],np.float64)
        code=self.lib.sf6_eval_face(self.handle,image.ctypes.data,image.shape[1],image.shape[0],box.ctypes.data,
            int(gate),features.ctypes.data,self.dimension,meta.ctypes.data,attributes.ctypes.data)
        if code:raise RuntimeError(self.lib.sf6_error().decode())
        if not all(np.isfinite(a).all() for a in (features,meta,attributes)):raise ValueError('Non-finite selected-face result')
        result=self.observation(meta,gate);result.update(face_source='yunet_selected',detection_score=None)
        if attributes[0]>=0:
            result['age_estimate']=dict(years=int(attributes[0]),estimated=True,source='seeta6_age',not_identity_evidence=True)
            result['timing_ms']['age']=float(attributes[1])
        return result,features if meta[15] else None

    def compare(self,a,b):
        return float(self.lib.sf6_similarity(self.handle,a.ctypes.data,b.ctypes.data))

    def close(self):
        if self.handle:self.lib.sf6_destroy(self.handle);self.handle=None
