"""Optional per-axis software acceleration; no actuator IDs or serial I/O."""
import math
import numpy as np

from vivis.motion.mapping import limit_step


class AccelerationLimiter:
    def __init__(self,profile,ids,command_ids=None,prelimited_ids=()):
        motors={m['id']:m for m in profile['motors']}
        commanded=set(ids if command_ids is None else command_ids)
        self.speeds=np.array([motors[sid]['max_speed_counts_s'] for sid in ids],dtype=float)
        self.axes=[]
        for j,sid in enumerate(ids):
            m=motors[sid];a=m.get('max_acceleration_counts_s2')
            if a is None:continue
            if type(a) not in (int,float) or not math.isfinite(a) or a<=0:
                raise ValueError(f'ID{sid}: invalid software acceleration')
            if sid in commanded and sid not in prelimited_ids:
                self.axes.append((j,sid,float(a),float(m['safe_min']),float(m['safe_max'])))
        self.velocity={sid:0. for _,sid,_,_,_ in self.axes}

    def at_rest(self):
        return all(abs(v)<1e-8 for v in self.velocity.values())

    def step(self,previous,target,dt):
        result=limit_step(previous,target,self.speeds,dt)
        for j,sid,a,low,high in self.axes:
            p=float(previous[j]);delta=float(target[j])-p;old=self.velocity[sid]
            # With p_next=p+v_next*dt, retain enough room to brake to a
            # fixed endpoint. Moving authored targets may be passed smoothly
            # while braking; they are not hard mechanical limits.
            def stopping_speed(distance):
                return math.sqrt((a*dt)**2+2*a*max(0.,distance))-a*dt
            requested=math.copysign(min(self.speeds[j],abs(delta)/dt,
                                        stopping_speed(abs(delta))),delta)
            requested=float(np.clip(requested,-stopping_speed(p-low),stopping_speed(high-p)))
            velocity=old+float(np.clip(requested-old,-a*dt,a*dt))
            goal=p+velocity*dt
            if not low-1e-7<=goal<=high+1e-7:
                raise RuntimeError(f'ID{sid}: software acceleration state exceeds position bounds')
            result[j]=float(np.clip(goal,low,high));self.velocity[sid]=velocity
        return result
