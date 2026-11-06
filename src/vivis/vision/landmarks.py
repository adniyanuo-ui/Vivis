"""One Face Landmarker for the already-selected single face; no actuator I/O."""
import hashlib
import math
from pathlib import Path


class FaceLandmarkerDetector:
    def __init__(self,config,root):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions,vision
        path=Path(root)/config['model']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=config['sha256']:
            raise ValueError('Face Landmarker model hash mismatch')
        self.mp=mp
        self.model=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),running_mode=vision.RunningMode.VIDEO,
            num_faces=1,output_face_blendshapes=True,min_face_detection_confidence=.5,
            min_face_presence_confidence=.5,min_tracking_confidence=.5))
        self.timestamp_ms=-1

    def observe(self,sample,target):
        import cv2
        sequence,stamp,image=sample
        observation=dict(schema_version='vivis-face-expression-1.0',sequence=sequence,
                         timestamp_monotonic=stamp,valid=False,landmark_count=0,weights={})
        if target is None:return observation
        x,y,w,h=target['bbox'];height,width=image.shape[:2]
        x0,y0=max(0,int(x-.15*w)),max(0,int(y-.15*h))
        x1,y1=min(width,math.ceil(x+1.15*w)),min(height,math.ceil(y+1.15*h))
        if x1<=x0 or y1<=y0:return observation
        roi=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)
        self.timestamp_ms=max(self.timestamp_ms+1,int(stamp*1000))
        result=self.model.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=roi),self.timestamp_ms)
        if len(result.face_blendshapes)==1 and len(result.face_landmarks)==1:
            observation.update(valid=True,landmark_count=len(result.face_landmarks[0]),
                weights={v.category_name:float(v.score) for v in result.face_blendshapes[0]})
        return observation

    def close(self):self.model.close()
