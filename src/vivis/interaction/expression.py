"""Named MediaPipe coefficients to the shared, bounded face mapper; no device I/O."""
import math
import numpy as np


class MediaPipeAdapter:
    """MediaPipe has _neutral plus51 expression scores; TongueOut is unavailable."""
    def __init__(self,canonical,mirror=False):
        self.canonical=list(canonical)
        if len(self.canonical)!=52 or len(set(self.canonical))!=52 or 'TongueOut' not in self.canonical:
            raise ValueError('Need canonical52 expression channels')
        self.names={name[0].lower()+name[1:]:name for name in self.canonical if name!='TongueOut'}
        if len(self.names)!=51 or type(mirror) is not bool:raise ValueError('Invalid expression adapter')
        self.indices=[]
        for name in self.canonical:
            other=(name[:-4]+'Right' if name.endswith('Left') else
                   name[:-5]+'Left' if name.endswith('Right') else name)
            self.indices.append(self.canonical.index(other if mirror else name))

    def adapt(self,observation):
        if (observation.get('schema_version')!='vivis-face-expression-1.0'
                or observation.get('valid') is not True or observation.get('landmark_count')!=478):
            raise ValueError('No verified single-face expression observation')
        raw=observation['weights']
        if not isinstance(raw,dict) or set(raw)!=set(self.names)|{'_neutral'}:
            raise ValueError('MediaPipe channel names differ from the verified51+neutral binding')
        if any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in raw.values()):
            raise ValueError('Invalid MediaPipe expression score')
        weights=[0. if name=='TongueOut' else raw[name[0].lower()+name[1:]] for name in self.canonical]
        return np.array(weights)[self.indices]


class ExpressionFrames:
    """Latest observation, short coefficient filter, bounded face-loss release.

    Camera observations are logged by the perception service. This controller
    keeps only current weights and summary ranges; it never queues old frames.
    """
    def __init__(self,receiver,mapper,settings,publisher=None,duration=60.,tracking_mode='face'):
        self.receiver,self.mapper,self.publisher=receiver,mapper,publisher
        self.duration=duration
        if tracking_mode not in ('face','eyes-neck'):raise ValueError('Invalid expression tracking mode')
        self.tracking_mode=tracking_mode
        self.adapter=MediaPipeAdapter(mapper.order[:52],settings['mirror_expression'])
        self.gain=settings['gain'];self.tau=settings['filter_tau_s']
        self.hold=settings['lost_hold_s'];self.release=settings['release_s']
        for value,low,high in ((self.gain,.1,2.),(self.tau,.001,1.),(self.hold,0.,1.),(self.release,.05,2.)):
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError('Invalid expression gain/filter/loss setting')
        self.filtered=np.zeros(52);self.raw=np.zeros(52);self.last_valid=None
        self.last_time=None;self.last_sequence=-1;self.last_publish=-math.inf
        self.samples=self.valid_observations=self.rejected_observations=self.sequence_skips=0
        self.channel_max=np.zeros(52);self.goal_min=np.full(len(mapper.ids),np.inf)
        self.goal_max=np.full(len(mapper.ids),-np.inf);self.state='waiting_for_face'
        self.error=None

    def sample(self,t,now):
        if not math.isfinite(now) or self.last_time is not None and now<self.last_time:
            raise ValueError('Expression clock must be monotonic')
        dt=0. if self.last_time is None else now-self.last_time
        self.last_time=now
        packet=self.receiver.poll(now);observation=None
        if packet is not None:
            candidate=packet.get('face_expression')
            if (isinstance(candidate,dict) and candidate.get('sequence')==packet.get('sequence')
                    and type(candidate.get('sequence')) is int
                    and candidate['sequence']>self.last_sequence):
                self.sequence_skips+=max(0,candidate['sequence']-self.last_sequence-1) if self.last_sequence>=0 else 0
                self.last_sequence=candidate['sequence']
                observation=candidate
        if observation is not None:
            stamp=observation.get('timestamp_monotonic')
            try:
                if (type(stamp) not in (int,float) or not math.isfinite(stamp)
                        or stamp!=packet.get('timestamp_monotonic') or not 0<=now-stamp<=self.receiver.max_age
                        or packet.get('target') is None):
                    raise ValueError('Stale, future or unassociated expression observation')
                weights=self.adapter.adapt(observation)
            except (ValueError,KeyError,TypeError,AttributeError):
                self.rejected_observations+=1
            else:
                self.raw=np.clip(weights*self.gain,0,1)
                self.last_valid=stamp;self.valid_observations+=1
                self.channel_max=np.maximum(self.channel_max,self.raw)
        age=math.inf if self.last_valid is None else now-self.last_valid
        # Freshness is measured from CAMERA CAPTURE, never from repeated polls.
        if age<=self.receiver.max_age:
            desired=self.raw;self.state='following'
        elif age<=self.receiver.max_age+self.hold:
            desired=self.raw;self.state='holding'
        else:
            fade=max(0.,1-(age-self.receiver.max_age-self.hold)/self.release)
            desired=self.raw*fade;self.state='releasing' if fade>0 else 'neutral'
        self.filtered+=(desired-self.filtered)*(-math.expm1(-dt/self.tau))
        goals=self.mapper.map_weights(self.filtered.tolist())
        self.samples+=1;self.goal_min=np.minimum(self.goal_min,goals);self.goal_max=np.maximum(self.goal_max,goals)
        if self.publisher is not None and now-self.last_publish>=.1:
            self.publisher.submit(dict(active=True,program='expression-follow',state=self.state,
                updated_monotonic=now,remaining_seconds=max(0.,self.duration-t),
                command_ids=self.mapper.ids,eye_neck_held=self.tracking_mode=='face',tracking_mode=self.tracking_mode,
                expression_top=sorted(zip(self.adapter.canonical,self.filtered.tolist()),key=lambda p:-p[1])[:8]))
            self.last_publish=now
        return goals

    def summary(self):
        return dict(source='camera_face_landmarker',samples=self.samples,valid_observations=self.valid_observations,
            rejected_observations=self.rejected_observations,camera_sequence_skips=self.sequence_skips,
            state=self.state,channel_max=dict(zip(self.adapter.canonical,self.channel_max.tolist())),
            nominal_goal_ranges={str(sid):[float(lo),float(hi)] for sid,lo,hi in
                zip(self.mapper.ids,self.goal_min,self.goal_max) if math.isfinite(lo)},
            tongue_out_available=False,eye_neck_held=self.tracking_mode=='face',tracking_mode=self.tracking_mode)
