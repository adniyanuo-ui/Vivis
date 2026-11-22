"""Observable face patterns -> bounded social poses, not emotion diagnosis."""
import math


def observation(face,single=True):
    stamp=face.get('timestamp_monotonic') if isinstance(face,dict) else None
    valid=bool(single and isinstance(face,dict) and face.get('valid') is True)
    weights=face.get('weights',{}) if valid else {}
    names=('mouthSmileLeft','mouthSmileRight','browDownLeft','browDownRight','browInnerUp',
        'mouthFrownLeft','mouthFrownRight','eyeWideLeft','eyeWideRight','browOuterUpLeft','browOuterUpRight')
    if valid and any(type(weights.get(n)) not in (int,float) or not math.isfinite(weights[n])
            or not 0<=weights[n]<=1 for n in names):valid=False
    pair=lambda a,b:min(weights[a],weights[b])
    scores=(dict(joy=pair('mouthSmileLeft','mouthSmileRight'),listening=pair('browDownLeft','browDownRight'),
        concern=min(weights['browInnerUp'],pair('mouthFrownLeft','mouthFrownRight')),
        surprise=min(pair('eyeWideLeft','eyeWideRight'),max(weights['browInnerUp'],pair('browOuterUpLeft','browOuterUpRight'))))
        if valid else {})
    return dict(schema_version='vivis-social-face-pattern-1.0',timestamp_monotonic=stamp,
        valid=valid,observation_only=True,scores=scores)


class SocialExpressionResponse:
    def __init__(self,c,presets,attack_s,release_s,max_age_s=.3):
        if (type(c.get('enabled')) is not bool
                or any(type(c.get(k)) not in (int,float) or not math.isfinite(c[k]) for k in ('enter','exit'))
                or not 0<=c['exit']<c['enter']<=1
                or any(type(c[k]) not in (int,float) or not math.isfinite(c[k]) or not low<=c[k]<=high
                    for k,low,high in (('sustain_s',.1,3),('hold_s',0,3),('filter_tau_s',.01,1),('invalid_hold_s',0,1)))):
            raise ValueError('Invalid social expression response settings')
        if type(max_age_s) not in (int,float) or not math.isfinite(max_age_s) or not .01<=max_age_s<=3:
            raise ValueError('Invalid expression observation freshness')
        self.c=c;self.presets=presets;self.attack=attack_s;self.release=release_s
        self.max_age=max_age_s
        self.state='neutral';self.filtered={name:0. for name in presets};self.weights={}
        self.last=self.stamp=self.candidate_since=self.low_since=self.invalid_since=None;self.candidate=None
        self.diagnostic={}

    def step(self,event,now):
        dt=1/60 if self.last is None else max(0.,min(.1,now-self.last));self.last=now
        stamp=event.get('timestamp_monotonic') if isinstance(event,dict) else None
        scores=event.get('scores') if isinstance(event,dict) else None
        valid=(self.c['enabled'] and isinstance(event,dict) and event.get('schema_version')=='vivis-social-face-pattern-1.0'
            and event.get('valid') is True and event.get('observation_only') is True
            and type(stamp) in (int,float) and 0<=now-stamp<=self.max_age and isinstance(scores,dict)
            and set(scores)==set(self.presets) and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in scores.values()))
        if valid and (self.stamp is None or stamp>self.stamp):
            elapsed=dt if self.stamp is None else min(1.,stamp-self.stamp)
            if self.stamp is not None and stamp-self.stamp>self.max_age:
                self.candidate=self.candidate_since=self.low_since=None
                self.state='neutral';self.filtered={name:0. for name in self.presets}
            self.stamp=stamp;self.invalid_since=None
            alpha=-math.expm1(-elapsed/self.c['filter_tau_s'])
            for name in self.filtered:self.filtered[name]+=alpha*(scores[name]-self.filtered[name])
            name=max(self.filtered,key=self.filtered.get)
            if self.filtered[name]>=self.c['enter']:
                self.low_since=None
                if name!=self.candidate:self.candidate=name;self.candidate_since=stamp
                if stamp-self.candidate_since>=self.c['sustain_s']:self.state=name
            elif self.state!='neutral' and self.filtered[self.state]>=self.c['exit']:
                self.low_since=None;self.candidate=self.candidate_since=None
            else:
                self.candidate=self.candidate_since=None
                if self.low_since is None:self.low_since=stamp
                if stamp-self.low_since>=self.c['hold_s']:self.state='neutral'
        elif not valid:
            if self.invalid_since is None:self.invalid_since=now
            self.candidate=self.candidate_since=None
            if now-self.invalid_since>=self.c['invalid_hold_s']:
                self.state='neutral';self.filtered={name:0. for name in self.presets};self.stamp=None
        target=self.presets.get(self.state,{}) if self.c['enabled'] else {}
        for name in set(self.weights)|set(target):
            previous=self.weights.get(name,0.);goal=target.get(name,0.)
            tau=self.attack if goal>previous else self.release
            self.weights[name]=previous+(goal-previous)*(-math.expm1(-dt/tau))
        self.weights={name:v for name,v in self.weights.items() if v>1e-5}
        self.diagnostic=dict(state=self.state,valid=bool(valid),scores={k:round(v,3) for k,v in self.filtered.items()},
            weights={k:round(v,3) for k,v in self.weights.items()},observation_only=True)
        return self.weights
