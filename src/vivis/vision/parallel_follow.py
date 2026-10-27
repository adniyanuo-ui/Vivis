"""Camera-relative pitch and light roll using a measured rotary parallel neck.

The existing eye/yaw controller is not modified. Encoder-derived platform pose
is a kinematic estimate, not an IMU measurement; eye-pan angle is not fabricated
from its servo counts. All motor commands still use the existing serial owner.
"""
import copy
import math
import numpy as np

from vivis.motion.parallel_neck import RotaryParallelNeck,UnreachablePose,rotation
from vivis.motion.upstream import require_engine,position_extrema


def configure(config,mode,option=None,home_only=False):
    plan=copy.deepcopy(config.get('parallel_neck',{'enabled':False}))
    if not isinstance(plan,dict) or type(plan.get('enabled')) is not bool:
        raise ValueError('Invalid parallel neck configuration')
    if option not in (None,'on','off'):raise ValueError('Invalid parallel neck option')
    active=mode=='eyes-neck' and not home_only
    if option=='on' and not active:raise ValueError('并联颈部只用于示例3的跟踪播放')
    if not active:plan['enabled']=False
    elif option is not None:plan['enabled']=option=='on'
    config['parallel_neck']=plan
    return plan


def validate(config,profile):
    plan=config.get('parallel_neck',{'enabled':False})
    if type(plan.get('enabled')) is not bool:raise ValueError('Invalid parallel neck enabled flag')
    if not plan['enabled']:return None
    if profile.get('sync_group_size',30)<2:raise ValueError('Parallel axes need a shared syncwrite group')
    if (plan.get('schema_version')!='vivis-parallel-neck-tracking-1.0'
            or plan.get('geometry_confirmed') is not True or plan.get('directions_confirmed') is not True
            or plan.get('yaw_above_platform') is not True):
        raise ValueError('Need confirmed CAD, encoder directions and above-platform yaw layout')
    k=RotaryParallelNeck(plan['geometry']);motors={m['id']:m for m in profile['motors']}
    for b in k.branches:
        m=motors[b['id']]
        if m['model_raw_expected']!=10760 or not m['device_min']<=b['low']<2047<b['high']<=m['device_max']:
            raise ValueError('Parallel tracking window exceeds configured device envelope')
    for section,limits in [('motion',{'max_speed_counts_s':(1,1000),'max_acceleration_counts_s2':(1,10000),
                                      'max_jerk_counts_s3':(1,1e6),'return_duration_s':(1,10)}),
                           ('control',{'pitch_gain_rad_s':(.01,1),'pitch_limit_rad':(.001,.3),
                                       'roll_follow_ratio':(0,.5),'roll_limit_rad':(.001,.15),
                                       'roll_gain_s':(.1,5),'image_deadband':(0,.2),'filter_tau_s':(.01,1)})]:
        for key,(low,high) in limits.items():
            value=plan[section][key]
            if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError('Invalid parallel '+section+' parameter: '+key)
    if any(plan['motion']['max_speed_counts_s']>motors[b['id']]['max_speed_counts_s'] for b in k.branches):
        raise ValueError('Parallel reference speed exceeds hardware speed configuration')
    require_engine()
    return k


def camera_axes(pose,yaw):
    """Above-platform yaw: R_head=Rx(pitch)Ry(roll)Rz(yaw)."""
    p,r=pose;c,s=math.cos(yaw),math.sin(yaw)
    head=rotation(p,r)@np.array([[c,-s,0],[s,c,0],[0,0,1]])
    axes=np.array([[1.,0.],[0.,math.cos(p)],[0.,math.sin(p)]])
    interaction=np.stack([head[:,0]@axes,head[:,1]@axes])
    camera_roll=math.atan2(-head[2,0],head[2,2])
    camera_pitch=math.atan2(head[2,1],math.hypot(head[0,1],head[1,1]))
    return interaction,camera_pitch,camera_roll


class CoupledJointReference:
    def __init__(self,kinematics,motion):
        self.k=kinematics;self.motion=motion;self.r=require_engine();self.engine=self.r.Ruckig(2)
        self.low=np.array([b['low'] for b in self.k.branches]);self.high=np.array([b['high'] for b in self.k.branches])
        self.reset([2047.,2047.])

    def reset(self,positions):
        self.p=np.array(positions,dtype=float);self.v=np.zeros(2);self.a=np.zeros(2)
        self.pose=self.k.forward(dict(zip((33,34),map(float,self.p))))
        self.return_profile=None;self.elapsed=0.;self.phase=None;self.diagnostic={}

    def profile(self,velocity=None,home=False,minimum=None):
        inp=self.r.InputParameter(2);inp.current_position=self.p.tolist();inp.current_velocity=self.v.tolist()
        inp.current_acceleration=self.a.tolist();inp.target_acceleration=[0.,0.]
        inp.max_velocity=[self.motion['max_speed_counts_s']]*2
        inp.max_acceleration=[self.motion['max_acceleration_counts_s2']]*2
        inp.max_jerk=[self.motion['max_jerk_counts_s3']]*2
        inp.target_position=[2047.,2047.] if home else self.p.tolist()
        inp.target_velocity=[0.,0.] if velocity is None else list(map(float,velocity))
        if not home:inp.control_interface=self.r.ControlInterface.Velocity
        if minimum is not None:inp.minimum_duration=minimum
        trajectory=self.r.Trajectory(2);result=self.engine.calculate(inp,trajectory)
        if result<0:raise RuntimeError('Coupled parallel trajectory calculation failed: '+str(result))
        return trajectory

    def bounded(self,trajectory):
        for j in range(2):
            low,high=position_extrema(trajectory,j)
            if low<self.low[j]-1e-7 or high>self.high[j]+1e-7:return False
        return True

    @staticmethod
    def sample(trajectory,dt):return tuple(np.array(x,dtype=float) for x in trajectory.at_time(dt))

    def step(self,velocity,dt,returning=False):
        old_a=self.a.copy();speed=self.motion['max_speed_counts_s']
        if type(dt) not in (int,float) or not math.isfinite(dt) or dt<=0:raise ValueError('Invalid parallel clock')
        requested=np.array(velocity,dtype=float)
        if requested.shape!=(2,) or not np.isfinite(requested).all():raise ValueError('Invalid parallel velocity')
        requested*=min(1.,speed/max(1e-12,float(np.max(np.abs(requested)))))
        if returning:
            if self.return_profile is None:
                profile=self.profile(home=True,minimum=self.motion['return_duration_s']);self.phase='home'
                if not self.bounded(profile):profile=self.profile();self.phase='brake_before_home'
                if not self.bounded(profile):raise RuntimeError('Parallel reference cannot return inside bounds')
                self.return_profile=profile;self.elapsed=0.
            self.elapsed+=dt;profile=self.return_profile
            state=self.sample(profile,min(self.elapsed,profile.duration))
            mode=self.phase
        else:
            self.return_profile=None;self.elapsed=0.;self.phase=None
            profile=self.profile(requested);candidate=self.sample(profile,dt)
            saved=self.p,self.v,self.a;self.p,self.v,self.a=candidate
            try:stop=self.profile()
            finally:self.p,self.v,self.a=saved
            if self.bounded(profile) and self.bounded(stop):state=candidate;mode='tracking'
            else:
                profile=self.profile()
                if not self.bounded(profile):raise RuntimeError('Parallel reference cannot brake inside bounds')
                state=self.sample(profile,min(dt,profile.duration));mode='workspace_braking'
        p,v,a=state
        if (not np.isfinite(np.r_[p,v,a]).all() or np.any(p<self.low-1e-7) or np.any(p>self.high+1e-7)
                or np.max(np.abs(v))>speed+1e-6
                or np.max(np.abs(a))>self.motion['max_acceleration_counts_s2']+1e-6
                or np.max(np.abs(a-old_a))>self.motion['max_jerk_counts_s3']*dt+1e-5):
            raise RuntimeError('Parallel reference exceeds motion bounds')
        pose=self.pose if np.array_equal(p,self.p) else self.k.forward(dict(zip((33,34),map(float,p))),self.pose)
        self.p,self.v,self.a,self.pose=p,v,a,pose
        self.diagnostic=dict(mode=mode,positions=p.tolist(),velocities=v.tolist(),accelerations=a.tolist(),
            jerk_average=((a-old_a)/dt).tolist(),reference_pose_rad=list(pose))
        if returning and self.phase=='brake_before_home' and self.elapsed>=self.return_profile.duration:
            self.return_profile=None
        return dict(zip((33,34),map(float,p)))


class ParallelFaceFollower:
    def __init__(self,config,profile):
        self.config=config;self.plan=config['parallel_neck'];self.k=validate(config,profile)
        self.reference=CoupledJointReference(self.k,self.plan['motion'])
        self.measurement_tolerance=max(m.get('position_feedback_tolerance_counts',0) for m in profile['motors'] if m['id'] in (33,34))
        self.observations={};self.measured_pose=(0.,0.);self.filtered_y=0.;self.filtered_roll=0.
        self.measured_counts=None
        self.stamp=None;self.diagnostic={}

    def observe(self,sid,position,now):
        if sid in (32,33,34):self.observations[sid]=(float(position),now)

    def reset(self,measured,now):
        self.reference.reset([measured[33],measured[34]])
        self.measured_pose=self.reference.pose;self.measured_counts=(measured[33],measured[34])
        self.stamp=None;self.filtered_y=self.filtered_roll=0.
        for sid in (32,33,34):self.observe(sid,measured[sid],now)

    def step(self,packet,state,dt,now,returning=False):
        c=self.plan['control'];max_age=self.config['transport']['max_age_s']
        should_return=returning or state in ('releasing','idle')
        fresh=all(s in self.observations and 0<=now-self.observations[s][1]<=max_age for s in (32,33,34))
        velocity=np.zeros(2);reason='holding'
        if fresh:
            counts=tuple(self.observations[s][0] for s in (33,34))
            if counts!=self.measured_counts:
                self.measured_pose=self.k.forward(dict(zip((33,34),counts)),self.measured_pose,self.measurement_tolerance)
                self.measured_counts=counts
        yaw=-self.config['neck']['sign']*(self.observations.get(32,(2047.,now))[0]-2047)*2*math.pi/4096
        interaction,pitch,roll=camera_axes(self.measured_pose,yaw)
        if fresh and state=='tracking' and not returning and packet and packet.get('target') is not None:
            target=packet['target'];stamp=packet['timestamp_monotonic']
            if 0<=now-stamp<=max_age:
                if self.stamp is None or stamp>self.stamp:
                    alpha=1. if self.stamp is None else -math.expm1(-(stamp-self.stamp)/c['filter_tau_s'])
                    self.filtered_y+=alpha*(float(target['y'])-self.filtered_y)
                    line=target.get('eye_line_roll_rad')
                    valid=type(line) in (int,float) and math.isfinite(line) and abs(line)<=math.pi/2
                    desired_roll=float(np.clip(c['roll_follow_ratio']*(line+roll),-c['roll_limit_rad'],c['roll_limit_rad'])) if valid else 0.
                    self.filtered_roll+=alpha*(desired_roll-self.filtered_roll);self.stamp=stamp
                error=math.copysign(max(0.,abs(self.filtered_y)-c['image_deadband']),self.filtered_y)
                vp=-c['pitch_gain_rad_s']*error
                if pitch*vp>0:
                    vp*=float(np.clip((c['pitch_limit_rad']-abs(pitch))/math.radians(.75),0.,1.))
                vr=c['roll_gain_s']*(self.filtered_roll-roll)
                pose_velocity=np.linalg.solve(interaction,np.array([vp,vr]))
                velocity=self.k.jacobian(*self.reference.pose)@pose_velocity
                reason='face_tracking'
        elif state in ('releasing','idle') or returning:
            self.stamp=None;self.filtered_y=self.filtered_roll=0.;reason='returning'
        if not fresh:reason='feedback_stale'
        goals=self.reference.step(velocity,dt,returning=should_return and fresh)
        self.diagnostic=dict(enabled=True,reason=reason,measured_pose_rad=list(self.measured_pose),
            head_yaw_rad=yaw,camera_pitch_rad=pitch,camera_roll_rad=roll,
            filtered_vertical_error=self.filtered_y,roll_target_rad=self.filtered_roll,
            feedback_age_s={str(s):now-self.observations[s][1] for s in self.observations},
            **self.reference.diagnostic)
        return goals
