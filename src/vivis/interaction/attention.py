"""Bounded attention intentions; never replace camera/encoder observations."""
import math
import random


class AttentionRhythm:
    def __init__(self,settings,seed=None):
        self.config=settings;self.rng=random.Random(seed)
        self.enabled=settings.get('enabled',False)
        if type(self.enabled) is not bool:raise ValueError('Invalid attention enabled switch')
        for key,default,low,high in [('contact_s',[4.,7.],2.,30.),('aversion_s',[.7,1.2],.4,3.)]:
            values=settings.get(key,default)
            if (not isinstance(values,list) or len(values)!=2
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
                    or not low<=values[0]<=values[1]<=high):raise ValueError('Invalid attention '+key)
            setattr(self,key,values)
        for key,default,low,high in [('eye_yaw_fraction',.16,0.,.25),('eye_pitch_fraction',.12,0.,.25),
                                   ('return_s',.25,.1,1.)]:
            value=settings.get(key,default)
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError('Invalid attention '+key)
            setattr(self,key,value)
        self.last=None;self.contact_at=self.next=self.started=self.abort_at=None
        self.duration=0.;self.blend=self.abort_origin=0.;self.count=0;self.direction=1
        self.previous_smile=False

    @staticmethod
    def smooth(u):
        u=max(0.,min(1.,u));return u**3*(10+u*(-15+6*u))

    def step(self,visible,now,smile=False):
        if type(now) not in (int,float) or not math.isfinite(now) or self.last is not None and now<self.last:
            raise ValueError('Invalid attention clock')
        self.last=now;began=False
        if not self.enabled:return dict(state='disabled',blend=0.,pause=False,started=False)
        if not visible:
            self.contact_at=self.next=None
            if self.started is not None and self.abort_at is None:
                self.abort_at=now;self.abort_origin=self.blend;self.started=None
        if self.abort_at is not None:
            u=(now-self.abort_at)/self.return_s;self.blend=self.abort_origin*(1-self.smooth(u))
            if u>=1:self.abort_at=None;self.blend=0.
            state='recovering'
        elif not visible:self.blend=0.;state='no_target'
        else:
            if self.contact_at is None:
                self.contact_at=now;self.next=now+self.rng.uniform(*self.contact_s)
            if self.started is None and smile and not self.previous_smile:
                self.next=min(max(self.next,now+.8),self.contact_at+self.contact_s[1]+1.)
            if self.started is None and now>=self.next:
                self.started=now;self.duration=self.rng.uniform(*self.aversion_s)
                self.direction=self.rng.choice((-1,1));self.count+=1;began=True
            if self.started is not None:
                u=(now-self.started)/self.duration
                if u>=1:
                    self.started=None;self.blend=0.;self.contact_at=now
                    self.next=now+self.rng.uniform(*self.contact_s);state='attending'
                else:
                    self.blend=self.smooth(u/.35) if u<.35 else 1-self.smooth((u-.35)/.65)
                    state='averting'
            else:self.blend=0.;state='attending'
        self.previous_smile=bool(smile)
        return dict(state=state,blend=float(self.blend),pause=state=='averting',started=began,
                    aversion_count=self.count)
