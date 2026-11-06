"""Strict named A2F adaptation and calibration-derived social arbitration."""
import math
import numpy as np

from vivis.motion.mapping import CompiledFaceMapping


class A2FAdapter:
    def __init__(self, canonical, playback_id):
        self.canonical=list(canonical)
        if len(self.canonical)!=52 or len(set(self.canonical))!=52:
            raise ValueError('Need canonical52 names')
        self.playback_id=playback_id
        self.order=None
        self.raw_count=None
        self.last_sequence=self.last_sample=-1

    def bind(self, packet):
        if self.order is not None or packet.get('schema_version')!='vivis-a2f-bindings-1.0' or packet.get('playback_id')!=self.playback_id:
            raise ValueError('Invalid or repeated A2F bindings')
        skin=packet['names']['skin'];tongue=packet['names']['tongue']
        if len(skin)!=52 or any(not isinstance(n,str) or not n for n in skin+tongue):
            raise ValueError('Invalid A2F names')
        names=[n[0].upper()+n[1:] for n in skin]
        if len(set(names))!=52 or set(names)!=set(self.canonical):
            raise ValueError('A2F semantic mapping does not cover canonical52')
        self.order=[names.index(n) for n in self.canonical]
        self.raw_count=len(skin)+len(tongue)

    def adapt(self, packet):
        if self.order is None or packet.get('schema_version')!='vivis-a2f-raw-1.0' or packet.get('playback_id')!=self.playback_id:
            raise ValueError('Invalid A2F frame/instance')
        seq,stamp,rate=packet['frame_seq'],packet['audio_sample_index'],packet['sample_rate_hz']
        if (type(seq) is not int or seq!=self.last_sequence+1 or type(stamp) is not int
                or stamp<0 or stamp<=self.last_sample or rate!=16000 or type(rate) is not int):
            raise ValueError('Invalid A2F sequence or audio sample timestamp')
        values=packet['weights']
        if (not isinstance(values,list) or len(values)!=self.raw_count
                or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values)):
            raise ValueError('Invalid raw A2F coefficients')
        self.last_sequence,self.last_sample=seq,stamp
        return dict(schema_version='vivis-animation-1.0',channel_mapping_version='a2f-skin52-v1',
                    playback_id=self.playback_id,frame_seq=seq,audio_sample_index=stamp,
                    sample_rate_hz=rate,weights=[values[i] for i in self.order])


class FrameMapper:
    def __init__(self, order, shapes, profile, ids, smile_strength=.15):
        if not math.isfinite(smile_strength) or not 0<=smile_strength<=.5:
            raise ValueError('Social strength must be0..0.5')
        self.order,self.shapes,self.profile,self.ids=list(order),shapes,profile,list(ids)
        self.smile_strength=smile_strength
        self.critical=('JawOpen','MouthClose','MouthFunnel','MouthPucker')
        self.social=('MouthSmileLeft','MouthSmileRight')
        self.neutral=np.array([next(m for m in profile['motors'] if m['id']==sid)['neutral'] for sid in ids],float)
        self.mapping=CompiledFaceMapping(self.order,self.shapes,self.profile,scope='face_eyes')
        self.last_fused=None
        self.protected_ids=[]

    def map_weights(self, weights, smile=0.):
        if (len(weights)!=52 or not all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in weights)
                or type(smile) not in (int,float) or not math.isfinite(smile) or not 0<=smile<=1):
            raise ValueError('Invalid canonical frame or smile envelope')
        combine=smile>1e-6
        value=np.zeros((2 if combine else 1,61));value[:,:52]=weights
        self.last_fused=list(weights);self.protected_ids=[]
        if combine:
            for channel in self.social:
                i=self.order.index(channel)
                value[1,i]=min(1.,value[1,i]+smile*self.smile_strength)
            self.last_fused=value[1,:52].tolist()
        # Batch base and social candidates through the SAME existing mapper.
        # Repeating the Python calibration traversal twice costs a control tick.
        source_ids=self.mapping.active
        mapped,_=self.mapping.map(value)
        result=self.neutral.copy()
        for i,sid in enumerate(source_ids):
            if sid in self.ids:result[self.ids.index(sid)]=mapped[0,i]
        if not combine:return result
        protected=set()
        for name in self.critical:
            if value[0,self.order.index(name)]>.01:
                protected.update(self.shapes[name])
        self.protected_ids=sorted(protected)
        # Derive shared resources from calibration, never a fixed motor-ID list.
        for i,sid in enumerate(source_ids):
            if sid in self.ids and sid not in protected:result[self.ids.index(sid)]=mapped[1,i]
        return result
