"""Optional single-axis reference planning; no serial or encoder control loop.

Tracking requests velocity. Neutral return requests one cached position profile.
All replans start from the previous floating-point reference p/v/a. Native
Ruckig computes jerk-limited profiles; independent extrema checks retain the
robot's configured position envelope, including future braking distance.
"""
import importlib
import math


def settings(neck):
    value=neck.get('upstream_trajectory',{'enabled':False})
    if not isinstance(value,dict) or type(value.get('enabled')) is not bool:
        raise ValueError('Invalid upstream trajectory enabled flag')
    if set(value)-{'enabled','max_acceleration_counts_s2','max_jerk_counts_s3'}:
        raise ValueError('Unknown upstream trajectory setting')
    result=dict(enabled=value['enabled'])
    for key,default in [('max_acceleration_counts_s2',800.),('max_jerk_counts_s3',8000.)]:
        number=value.get(key,default)
        if type(number) not in (int,float) or not math.isfinite(number) or not 0<number<=1e6:
            raise ValueError('Invalid upstream trajectory limit: '+key)
        result[key]=float(number)
    return result


def configure(config,mode,option=None,home_only=False):
    """Resolve CLI/config once, identically in the launcher and show runtime."""
    if option not in (None,'on','off'):
        raise ValueError('Upstream trajectory option must be on/off')
    active=mode=='eyes-neck' and not home_only
    if option=='on' and not active:
        raise ValueError('上游轨迹规划只能用于示例3的眼颈跟踪播放')
    plan=settings(config['neck'])
    if not active:plan['enabled']=False
    elif option is not None:plan['enabled']=option=='on'
    config['neck']['upstream_trajectory']=plan
    return plan


def require_engine():
    try:
        engine=importlib.import_module('ruckig')
    except ImportError as exc:
        raise RuntimeError("上游轨迹规划需要可选依赖：CMAKE_BUILD_PARALLEL_LEVEL=2 "
                           ".venv/bin/python -m pip install -e '.[trajectory]'；"
                           "或使用 --upstream-trajectory off") from exc
    if getattr(engine,'__version__',None)!='0.19.4':
        raise RuntimeError('上游轨迹规划需要ruckig==0.19.4；执行 ./tools/setup.sh --trajectory，'
                           '或使用 --upstream-trajectory off')
    return engine


class UpstreamTrajectory:
    """A bounded, continuous reference state, independent of integer packets."""
    def __init__(self,low,high,maximum_speed,plan,return_duration=3.,neutral=2047.):
        values=(low,high,maximum_speed,return_duration,neutral)
        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
                or not low<neutral<high or maximum_speed<=0 or return_duration<=0):
            raise ValueError('Invalid upstream trajectory bounds')
        self.r=require_engine();self.engine=self.r.Ruckig(1)
        self.low,self.high=float(low),float(high)
        self.maximum_speed=float(maximum_speed)
        self.maximum_acceleration=plan['max_acceleration_counts_s2']
        self.maximum_jerk=plan['max_jerk_counts_s3']
        self.neutral=float(neutral);self.minimum_return_duration=float(return_duration)
        self.position=None;self.velocity=self.acceleration=0.
        self.return_profile=None;self.return_elapsed=0.;self.boundary_direction=0
        self.last_diagnostic={}

    def reset(self,position):
        if not math.isfinite(position) or not self.low<=position<=self.high:
            raise ValueError('Invalid upstream initial position')
        self.position=float(position);self.velocity=self.acceleration=0.
        self.return_profile=None;self.return_elapsed=0.;self.boundary_direction=0

    def _profile(self,state,velocity=0.,position=None,minimum_duration=None):
        inp=self.r.InputParameter(1)
        inp.current_position=[state[0]];inp.current_velocity=[state[1]];inp.current_acceleration=[state[2]]
        inp.max_velocity=[self.maximum_speed];inp.max_acceleration=[self.maximum_acceleration]
        inp.max_jerk=[self.maximum_jerk]
        # Velocity control ignores target_position; retain a meaningful anchor.
        inp.target_position=[state[0]]
        inp.target_velocity=[velocity];inp.target_acceleration=[0.]
        if position is None:
            inp.control_interface=self.r.ControlInterface.Velocity
        else:
            inp.target_position=[position]
            if minimum_duration is not None:inp.minimum_duration=minimum_duration
        trajectory=self.r.Trajectory(1)
        result=self.engine.calculate(inp,trajectory)
        if result<0:raise RuntimeError('Upstream trajectory calculation failed: '+str(result))
        return trajectory

    def _bounded(self,trajectory,margin=0.):
        low,high=position_extrema(trajectory)
        return self.low+margin-1e-7<=low and high<=self.high-margin+1e-7

    @staticmethod
    def _sample(trajectory,elapsed):
        p,v,a=trajectory.at_time(elapsed)
        return float(p[0]),float(v[0]),float(a[0])

    @property
    def state(self):
        return self.position,self.velocity,self.acceleration

    def return_window(self,position):
        state=(float(position),0.,0.) if self.position is None else self.state
        profile=self._profile(state,position=self.neutral,minimum_duration=self.minimum_return_duration)
        return max(5.,profile.duration+1.)

    def step(self,requested_velocity,dt,returning=False):
        if (self.position is None or type(dt) not in (int,float) or not math.isfinite(dt) or dt<=0
                or type(requested_velocity) not in (int,float) or not math.isfinite(requested_velocity)):
            raise ValueError('Invalid upstream trajectory sample')
        requested=float(max(-self.maximum_speed,min(self.maximum_speed,requested_velocity)))
        old_acceleration=self.acceleration
        if returning:
            self.boundary_direction=0
            if self.return_profile is None:
                profile=self._profile(self.state,position=self.neutral,minimum_duration=self.minimum_return_duration)
                if not self._bounded(profile):raise RuntimeError('Upstream neutral trajectory leaves position bounds')
                self.return_profile=profile;self.return_elapsed=0.
            self.return_elapsed+=dt
            state=self._sample(self.return_profile,min(self.return_elapsed,self.return_profile.duration))
            mode='return'
        else:
            self.return_profile=None;self.return_elapsed=0.
            direction=1 if requested>0 else -1 if requested<0 else 0
            if direction!=self.boundary_direction:self.boundary_direction=0
            profile=self._profile(self.state,velocity=requested)
            candidate=self._sample(profile,dt)
            stop=self._profile(candidate)
            # Do not enter a state from which a jerk-limited stop would leave
            # the envelope. At a boundary, finish at the endpoint rather than
            # alternate velocity/stop requests on adjacent ticks.
            if self.boundary_direction or not self._bounded(profile) or not self._bounded(stop):
                bound=self.high if (direction or (1 if self.velocity>0 else -1))>0 else self.low
                endpoint=self._profile(self.state,position=bound)
                if direction and self._bounded(endpoint):
                    profile=endpoint;self.boundary_direction=direction;mode='boundary'
                else:
                    profile=self._profile(self.state);mode='braking'
                    if not self._bounded(profile):raise RuntimeError('Upstream reference cannot stop inside position bounds')
                state=self._sample(profile,min(dt,profile.duration))
            else:
                state=candidate;mode='velocity'
        p,v,a=state
        if (not all(math.isfinite(x) for x in state) or not self.low-1e-7<=p<=self.high+1e-7
                or abs(v)>self.maximum_speed+1e-6 or abs(a)>self.maximum_acceleration+1e-6
                or abs(a-old_acceleration)>self.maximum_jerk*dt+1e-5):
            raise RuntimeError('Upstream trajectory reference exceeds motion limits')
        self.position,self.velocity,self.acceleration=state
        self.last_diagnostic=dict(enabled=True,mode=mode,position=p,velocity=v,acceleration=a,
            jerk_average=(a-old_acceleration)/dt,dt=dt,requested_velocity=requested,
            return_duration_s=self.return_profile.duration if self.return_profile is not None else None)
        return p


def position_extrema(trajectory,axis=0):
    """Exact cubic-segment extrema, including zero-velocity interior points.

    Ruckig0.19.4 velocity profiles leave pf=0; native position_extrema includes
    that unused endpoint. Use actual piecewise p/v/a/j data instead, without
    treating the fictitious zero as a real position or ignoring real overshoot.
    """
    values=[trajectory.at_time(0.)[0][axis],trajectory.at_time(trajectory.duration)[0][axis]]
    for section in trajectory.profiles:
        profile=section[axis]
        for phases in (profile.brake,profile,profile.accel):
            for index,duration in enumerate(phases.t):
                if duration<=0:continue
                p,v,a,j=(getattr(phases,name)[index] for name in ('p','v','a','j'))
                times=[0.,duration]
                if abs(j)>1e-12:
                    discriminant=a*a-2*j*v
                    if discriminant>=0:
                        root=math.sqrt(discriminant)
                        times.extend(t for t in ((-a-root)/j,(-a+root)/j) if 0<t<duration)
                elif abs(a)>1e-12 and 0<-v/a<duration:
                    times.append(-v/a)
                values.extend(p+v*t+a*t*t/2+j*t*t*t/6 for t in times)
    return min(values),max(values)
