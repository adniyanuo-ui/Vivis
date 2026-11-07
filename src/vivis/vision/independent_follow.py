"""Independent pitch/roll joints with a head-fixed camera; no parallel solver.

Pitch reuses visual-error eye/neck recruitment. Roll follows a small fraction
of the image eye-line tilt, not an invented absolute head-angle measurement.
Both joints reuse the bounded single-axis Ruckig reference and serial owner.
"""
import copy
import math

from vivis.motion.upstream import UpstreamTrajectory,settings
from vivis.vision.policy import axis_normalized,finite
from vivis.vision.pursuit import VisualNeckAssist,camera_axes


def configure(config,mode,home_only=False):
    plan=copy.deepcopy(config.get('independent_neck',{'enabled':False}))
    if not isinstance(plan,dict) or type(plan.get('enabled')) is not bool:
        raise ValueError('Invalid independent neck configuration')
    if mode!='eyes-neck' or home_only:plan['enabled']=False
    if plan['enabled'] and config.get('parallel_neck',{}).get('enabled'):
        raise ValueError('Independent joints cannot use the parallel neck option')
    config['independent_neck']=plan
    return plan


def validate(config,profile):
    plan=config.get('independent_neck',{'enabled':False})
    if type(plan.get('enabled')) is not bool:raise ValueError('Invalid independent neck enabled flag')
    if not plan.get('enabled'):return None
    if (plan.get('schema_version')!='vivis-independent-neck-tracking-1.0'
            or config.get('neck_topology')!='independent'
            or profile.get('schema_version')!='vivis-feetech-playback-1.0'
            or config['control']['camera_mount']!='fixed'
            or plan.get('directions_confirmed') is not True
            or config.get('parallel_neck',{}).get('enabled')):
        raise ValueError('Need confirmed independent neck axes and a fixed camera')
    motors={m['id']:m for m in profile['motors']}
    for role,sid in (('pitch',33),('roll',34)):
        a=plan[role];m=motors[sid]
        low,high=joint_limits(plan,m)
        if (a['id']!=sid or type(a['sign']) is not int or a['sign'] not in (-1,1)
                or m['model_raw_expected']!=11272
                or not m['device_min']<=low<2047<high<=m['device_max']
                or not 0<m['max_speed_counts_s']<=1000):
            raise ValueError('Invalid independent neck actuator: '+role)
    if not config['control'].get('vertical_enabled') or camera_axes(config)[1] is None:
        raise ValueError('Independent pitch needs the vertical eye reference')
    if not finite(plan['pitch']['gain_counts_s']) or not 1<=plan['pitch']['gain_counts_s']<=motors[33]['max_speed_counts_s']:
        raise ValueError('Invalid independent pitch gain')
    for key,low,high in (('roll_follow_ratio',0.,.5),('roll_reference_rad',.1,math.pi/2),
                         ('roll_gain_s',.1,5.),('roll_filter_tau_s',.01,1.),('return_duration_s',1.,10.)):
        if not finite(plan[key]) or not low<=plan[key]<=high:raise ValueError('Invalid independent neck '+key)
    motion=settings(plan)
    if plan.get('roll_control_mode','position') not in ('position','visual_velocity'):
        raise ValueError('Unknown independent roll control mode')
    if not motion['enabled']:raise ValueError('Independent pitch/roll need bounded trajectory planning')
    return motors


def joint_limits(plan,motor):
    mode=plan.get('range_mode','configured')
    if mode=='device':return motor['device_min'],motor['device_max']
    if mode=='configured':return motor['tracking_min'],motor['tracking_max']
    raise ValueError('Unknown independent neck range mode')


class IndependentReference:
    def __init__(self,plan,motors):
        self.planners={sid:UpstreamTrajectory(*joint_limits(plan,motors[sid]),
            motors[sid]['max_speed_counts_s'],settings(plan),plan['return_duration_s']) for sid in (33,34)}

    def reset(self,measured):
        for sid,p in self.planners.items():p.reset(measured[sid])

    def return_window(self):
        return max(p.return_window(p.position) for p in self.planners.values())

    def step(self,velocity,dt,returning):
        return {sid:p.step(velocity[sid],dt,returning=returning) for sid,p in self.planners.items()}


class IndependentFaceFollower:
    def __init__(self,config,profile):
        self.config=config;self.plan=config['independent_neck'];self.motors=validate(config,profile)
        self.reference=IndependentReference(self.plan,self.motors)
        self.eye=camera_axes(config)[1];self.eye_id=self.eye['id']
        # Reuse the yaw recruitment rule on vertical image error. No timer or
        # command-based subtraction of presumed neck motion is introduced.
        vertical=copy.deepcopy(config);vertical['neck'].update(self.plan['pitch'])
        vertical['camera']['width']=config['camera']['height']
        self.assist=VisualNeckAssist(vertical)
        self.observations={};self.eye_position=None;self.eye_motion_position=None;self.eye_motion=0.;self.eye_motion_at=-math.inf
        self.stamp=None;self.filtered_roll=0.;self.diagnostic={}

    def observe(self,sid,position,now):
        if sid in (self.eye_id,33,34):self.observations[sid]=(float(position),now)
        if sid==self.eye_id:
            if self.eye_motion_position is None:self.eye_motion_position=float(position)
            if abs(position-self.eye_motion_position)>=1:
                direction=1. if position>self.eye_motion_position else -1.
                self.eye_motion=direction if self.eye['positive']>self.eye['neutral'] else -direction
                self.eye_motion_at=now
                self.eye_motion_position=float(position)
            self.eye_position=float(position)

    def reset(self,measured,now):
        self.reference.reset(measured);self.observations={};self.eye_position=self.eye_motion_position=None
        self.eye_motion=0.;self.eye_motion_at=-math.inf;self.stamp=None;self.filtered_roll=0.;self.assist.reset()
        for sid in (self.eye_id,33,34):self.observe(sid,measured[sid],now)

    def step(self,packet,state,dt,now,returning=False):
        age=self.config['transport']['max_age_s'];velocity={33:0.,34:0.}
        fresh=all(sid in self.observations and 0<=now-self.observations[sid][1]<=age for sid in (self.eye_id,33,34))
        visible=(state=='tracking' and packet is not None and packet.get('target') is not None
                 and 0<=now-packet['timestamp_monotonic']<=age)
        should_return=returning or state in ('releasing','idle')
        if fresh and visible and not returning:
            image=copy.deepcopy(packet);image['target']['x']=packet['target']['y']
            eye=axis_normalized(self.eye,self.eye_position)
            span=abs(self.eye['positive' if eye>=0 else 'negative']-2047)
            motion=self.eye_motion if now-self.eye_motion_at<=age else 0.
            velocity[33]=self.assist.step(image,state,eye,motion,span,True)
            stamp=packet['timestamp_monotonic'];line=packet['target'].get('eye_line_roll_rad')
            roll_valid=finite(line) and abs(line)<=math.pi/2
            line=line if roll_valid else 0.
            if self.stamp is None or stamp>self.stamp:
                alpha=1. if self.stamp is None else -math.expm1(-(stamp-self.stamp)/self.plan['roll_filter_tau_s'])
                self.filtered_roll+=alpha*(line-self.filtered_roll);self.stamp=stamp
            ref=self.reference.planners[34];extent=min(2047-ref.low,ref.high-2047)
            offset=max(-1.,min(1.,self.filtered_roll/self.plan['roll_reference_rad']))
            target=2047+self.plan['roll']['sign']*extent*self.plan['roll_follow_ratio']*offset
            if self.plan.get('roll_control_mode','position')=='visual_velocity':
                # A head-fixed camera measures residual image tilt. Restoring
                # toward a neutral-anchored absolute position opposes the lean.
                # Feed image error into the existing bounded velocity planner;
                # do not pretend that encoder counts are measured head angles.
                velocity[34]=self.plan['roll_gain_s']*(target-2047) if roll_valid else 0.
            else:velocity[34]=self.plan['roll_gain_s']*(target-self.reference.planners[34].position)
            reason=self.assist.reason
        else:
            self.assist.reset()
            if should_return:self.stamp=None;self.filtered_roll=0.
            reason='feedback_stale' if not fresh else 'returning' if should_return else 'holding'
        goals=self.reference.step(velocity,dt,returning=should_return and fresh)
        self.diagnostic=dict(enabled=True,topology='independent',reason=reason,requested_velocity=velocity,
            roll_control_mode=self.plan.get('roll_control_mode','position'),
            image_roll_rad=self.filtered_roll,eye_reference_id=self.eye_id,
            eye_motion_source='encoder',feedback_age_s={str(s):now-v[1] for s,v in self.observations.items()},
            axes={str(s):dict(p.last_diagnostic) for s,p in self.reference.planners.items()})
        return goals
