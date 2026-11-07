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


def speech_channel_gains(settings,order):
    """Absolute source gains; round lips override, never multiply a second time."""
    mouth=settings.get('mouth_channel_gain',1.);round_lip=settings.get('round_lip_channel_gain',1.)
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<v<=3 for v in (mouth,round_lip)):
        raise ValueError('Speech channel gains must be finite in (0,3]')
    gains={name:mouth for name in order[:52] if name.startswith(('Mouth','Jaw','Tongue'))}
    gains.update(MouthFunnel=round_lip,MouthPucker=round_lip)
    return gains


class FrameMapper:
    def __init__(self, order, shapes, profile, ids, smile_strength=.15,scope='face_eyes',channel_gains=None,social_channels=None):
        if not math.isfinite(smile_strength) or not 0<=smile_strength<=.5:
            raise ValueError('Social strength must be0..0.5')
        self.order,self.shapes,self.profile,self.ids=list(order),shapes,profile,list(ids)
        self.smile_strength=smile_strength
        self.channel_gains=dict(channel_gains or {})
        if (not set(self.channel_gains)<=set(self.order[:52]) or any(type(v) not in (int,float)
                or not math.isfinite(v) or not 0<v<=3 for v in self.channel_gains.values())):
            raise ValueError('Invalid source channel gains')
        self.gains=np.array([self.channel_gains.get(name,1.) for name in self.order[:52]],float)
        self.critical=('JawOpen','MouthClose','MouthFunnel','MouthPucker')
        self.social=tuple(social_channels if social_channels is not None else ('MouthSmileLeft','MouthSmileRight'))
        allowed={'MouthSmileLeft','MouthSmileRight','CheekSquintLeft','CheekSquintRight','EyeSquintLeft','EyeSquintRight'}
        if not self.social or len(set(self.social))!=len(self.social) or not set(self.social)<=allowed:
            raise ValueError('Invalid social smile channels')
        self.neutral=np.array([next(m for m in profile['motors'] if m['id']==sid)['neutral'] for sid in ids],float)
        self.mapping=CompiledFaceMapping(self.order,self.shapes,self.profile,scope=scope)
        # Reserve articulation resources for the whole song. A .01 coefficient
        # threshold switched the entire social target on/off at shared joints,
        # even when source coefficients and the smile envelope were continuous.
        self.speech_owned_ids=sorted(set(self.ids)&{sid for name in self.critical for sid in self.shapes[name]})
        self.last_fused=None
        self.protected_ids=[]
        self.last_social_delta=np.zeros(len(ids))

    def map_weights(self, weights, smile=0.):
        if (len(weights)!=52 or not all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in weights)
                or type(smile) not in (int,float) or not math.isfinite(smile) or not 0<=smile<=1):
            raise ValueError('Invalid canonical frame or smile envelope')
        combine=smile>1e-6
        value=np.zeros((2 if combine else 1,61));value[:,:52]=np.clip(np.asarray(weights)*self.gains,0,1)
        self.last_fused=value[0,:52].tolist();self.protected_ids=[];self.last_social_delta.fill(0)
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
        baseline=result.copy()
        protected=set(self.speech_owned_ids)
        self.protected_ids=self.speech_owned_ids.copy()
        # Derive shared resources from calibration, never a fixed motor-ID list.
        for i,sid in enumerate(source_ids):
            if sid in self.ids and sid not in protected:result[self.ids.index(sid)]=mapped[1,i]
        self.last_social_delta=result-baseline
        return result
