"""Brief acoustic attention intents. No audio, camera or actuator I/O.

Positive bearings mean robot-left. Mounting and angular conversion are explicit
engineering settings; a serial sector is never identity or speech evidence.
"""
from collections import deque
import json
import math
from pathlib import Path
import threading
import time
import uuid

from vivis.paths import ROOT


CONFIG = ROOT/'configs/perception/sound_direction.json'


def configuration(path=CONFIG):
    c=json.loads(Path(path).read_text())['attention']
    if type(c.get('enabled')) is not bool:raise ValueError('Invalid acoustic attention switch')
    for key,low,high in [('budget_s',1,8),('dwell_s',0,3),('direction_difference_deg',30,120),
            ('cooldown_s',2,30),('lease_s',.2,1),('speech_run_s',.06,.3),
            ('camera_half_fov_deg',10,60),('yaw_counts_per_deg',1,30),
            ('search_s',.5,5),('minimum_clean_rms',.001,.1),('maximum_reference_rms',.0001,.1)]:
        value=c.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError('Invalid acoustic attention setting: '+key)
    if c['dwell_s']>c['budget_s']-.4:raise ValueError('Acoustic dwell must leave time to move and return')
    if type(c.get('no_face_search_enabled')) is not bool:raise ValueError('Invalid no-face search switch')
    m=c['mount']
    if (type(m.get('front_sector')) is not int or not 0<=m['front_sector']<12
            or type(m.get('sector_sign')) is not int or m['sector_sign'] not in (-1,1)
            or type(m.get('calibrated')) is not bool or type(m.get('allow_provisional')) is not bool):
        raise ValueError('Invalid acoustic mounting')
    return c


def angle(value):return (value+180)%360-180


def bearing(sector,c):
    if type(sector) is not int or not 0<=sector<12:raise ValueError('Invalid sector')
    return angle((sector-c['mount']['front_sector'])*30*c['mount']['sector_sign'])


def recent(packet,now,age):
    return (isinstance(packet,dict) and type(packet.get('updated_monotonic')) in (int,float)
        and 0<=now-packet['updated_monotonic']<=age)


class Mailbox:
    """Latest-only files off control/audio loops; bounded stale data fails closed."""
    def __init__(self,read_path,write_path):
        self.read_path,self.write_path=Path(read_path),Path(write_path)
        self.latest=None;self.pending=None;self.error=None
        self.stop=threading.Event();self.thread=None

    def start(self):
        self.thread=threading.Thread(target=self.run,daemon=True,name='acoustic-attention-mailbox')
        self.thread.start()
        return self

    def submit(self,packet):self.pending=packet

    def run(self):
        saved=None
        while not self.stop.is_set():
            try:
                if self.read_path.stat().st_size<=16384:
                    value=json.loads(self.read_path.read_text())
                    self.latest=value if isinstance(value,dict) else None
            except (OSError,ValueError):self.latest=None
            value=self.pending
            if value is not None and value is not saved:
                temporary=self.write_path.with_name(self.write_path.name+'.'+str(id(self))+'.tmp')
                try:
                    temporary.write_text(json.dumps(value,allow_nan=False)+'\n')
                    temporary.replace(self.write_path);saved=value;self.error=None
                except (OSError,ValueError) as exc:self.error=type(exc).__name__
            self.stop.wait(.02)

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=.5)


class SideSpeechGate:
    """VAD + fresh independent direction. Never promotes a side voice to actor.

    Reference playback is conservatively excluded. No reliable double-talk
    detector is assumed; source direction alone cannot defeat self-echo.
    """
    def __init__(self,c,vad):
        self.c,self.vad=c,vad;self.run=0.;self.last=None;self.candidate=None
        self.last_episode=-math.inf;self.intervals=deque(maxlen=64)
        self.suppressed_until=-math.inf;self.last_voice=-math.inf
        self.diagnostic=dict(reason='not_observed')

    def note(self,reason,**fields):
        self.diagnostic=dict(reason=reason,run_s=self.run,**fields)

    def reset(self,*,keep_history=False,now=None):
        self.run=0.;self.last=None;self.candidate=None
        self.suppressed_until=-math.inf;self.last_voice=-math.inf
        if keep_history:
            if self.intervals:self.intervals[-1][1]=min(self.intervals[-1][1],now)
        else:self.intervals.clear()

    def observe(self,stamp,pcm,metrics,direction,feedback,*,available,now):
        continuous=self.last is not None and .005<=stamp-self.last<=.035
        self.last=stamp
        if not continuous:self.run=0.
        # Suppress one bounded side utterance, including its tail after returning.
        if now<self.suppressed_until:
            new_main=(recent(direction,now,.3) and direction.get('frame_valid')
                and direction.get('sequence')!=getattr(self,'side_sequence',None)
                and abs(angle(bearing(direction['sector'],self.c)-self.main_anchor))<self.c['direction_difference_deg'])
            voiced=self.vad(pcm) and metrics['clean_rms']>=self.c['minimum_clean_rms']
            if voiced:self.last_voice=now
            if new_main or now-self.last_voice>.35:
                self.suppressed_until=-math.inf
                if self.intervals:self.intervals[-1][1]=stamp
            else:self.note('side_utterance_masked',admission=getattr(self,'mask_reason',None));return True,None
        reason=('direction_unavailable' if not recent(direction,now,.75) or not direction.get('frame_valid') else
            'gaze_feedback_unavailable' if not recent(feedback,now,self.c['lease_s']) else
            'gaze_busy' if feedback.get('phase')!='idle' else
            'main_face_unavailable' if feedback.get('single_face') is not True else
            'mount_unconfirmed' if not (self.c['mount']['calibrated'] or self.c['mount']['allow_provisional']) else None)
        if reason:
            self.run=0.;self.candidate=None;self.note(reason);return False,None
        azimuth=bearing(direction['sector'],self.c)
        difference=abs(angle(azimuth-feedback['main_bearing_deg']))
        # VAD is stateful: preserve its former call order even when logging a
        # same-direction rejection. Diagnostics must not retune speech history.
        audible=metrics['clean_rms']>=self.c['minimum_clean_rms']
        overlap=metrics['reference_rms']>=self.c['maximum_reference_rms']
        speech=self.vad(pcm) if audible and not overlap else False
        reason=('same_direction' if difference<self.c['direction_difference_deg'] else
            'quiet' if not audible else 'robot_playback_overlap' if overlap else 'not_speech' if not speech else None)
        if reason:
            self.run=0.;self.candidate=None
            self.note(reason,sector=direction['sector'],difference_deg=round(difference,1));return False,None
        # Quarantine from the first candidate frame, before cloud VAD/tools.
        if self.candidate!=direction['sector']:self.run=0.;self.candidate=direction['sector']
        self.run+=.02
        if self.run+1e-9<self.c['speech_run_s']:
            self.note('collecting_speech',sector=direction['sector']);return True,None
        may_glance=now-self.last_episode>=self.c['cooldown_s']
        may_start=available
        self.mask_reason='accepted' if may_start else 'main_busy'
        self.last_voice=now;self.suppressed_until=now+90.
        self.side_sequence=direction.get('sequence');self.main_anchor=feedback['main_bearing_deg']
        onset=stamp-self.run+.02
        self.intervals.append([onset-.25,now+90.])
        self.run=0.
        if not may_start:
            self.note('main_busy' if not available else 'cooldown',sector=direction['sector']);return True,None
        if may_glance:self.last_episode=now
        self.note('accepted',sector=direction['sector'],difference_deg=round(difference,1))
        return True,dict(id=uuid.uuid4().hex,owner=feedback['owner'],sector=direction['sector'],
            bearing_deg=azimuth,heard_monotonic=onset,created_monotonic=now,
            action='glance',glance_permitted=may_glance,mount_calibrated=self.c['mount']['calibrated'])

    def contains(self,stamp):
        return stamp is not None and any(start<=stamp<=end for start,end in self.intervals)


class NoFaceSpeechGate:
    """Motion request only; original PCM, speaker attribution and wake stay intact."""
    def __init__(self,c,vad):
        self.c,self.vad=c,vad;self.last=None;self.run=0.;self.candidate=None
        self.used=deque(maxlen=64);self.last_episode=-math.inf
        self.diagnostic=dict(reason='not_observed')

    def reset(self):self.last=None;self.run=0.;self.candidate=None

    def observe(self,stamp,pcm,metrics,direction,feedback,*,available,now):
        continuous=self.last is not None and .005<=stamp-self.last<=.035
        self.last=stamp
        if not continuous:self.run=0.
        reason=('direction_unavailable' if not recent(direction,now,.75) or not direction.get('frame_valid') else
            'gaze_feedback_unavailable' if not recent(feedback,now,self.c['lease_s']) or not feedback.get('visual_fresh') else
            'face_available' if feedback.get('face_count')!=0 else
            'gaze_busy' if feedback.get('phase')!='idle' else
            'main_busy' if not available else
            'mount_unconfirmed' if not (self.c['mount']['calibrated'] or self.c['mount']['allow_provisional']) else None)
        if reason:self.reset();self.diagnostic=dict(reason=reason);return None
        key=(direction.get('sequence'),direction['updated_monotonic'])
        reason=('already_used_direction' if key in self.used else
            'cooldown' if now-self.last_episode<self.c['cooldown_s'] else
            'robot_playback_overlap' if metrics['reference_rms']>=self.c['maximum_reference_rms'] else
            'quiet' if metrics['clean_rms']<self.c['minimum_clean_rms'] else
            'not_speech' if not self.vad(pcm) else None)
        if reason:self.run=0.;self.candidate=None;self.diagnostic=dict(reason=reason);return None
        if self.candidate!=direction['sector']:self.run=0.;self.candidate=direction['sector']
        self.run+=.02;self.diagnostic=dict(reason='collecting_speech',run_s=self.run)
        if self.run+1e-9<self.c['speech_run_s']:return None
        onset=stamp-self.run+.02
        self.run=0.;self.used.append(key);self.last_episode=now;self.diagnostic=dict(reason='no_face_seek')
        return dict(id=uuid.uuid4().hex,owner=feedback['owner'],sector=direction['sector'],
            bearing_deg=bearing(direction['sector'],self.c),heard_monotonic=onset,
            created_monotonic=now,action='seek',mount_calibrated=self.c['mount']['calibrated'])


class GlanceMotion:
    """One finite out/return, then bounded reacquisition; existing owner sends it.

    Quintic profiles start only from a settled target. Their analytic velocity,
    acceleration and jerk bounds limit excursion to fit the soft time budget.
    Native feedback, final command limits and watchdogs remain outside this layer.
    """
    def __init__(self,c,ids,low,high,speeds,neutral,axes,neck_plan):
        self.c=c;self.ids=ids;self.low=low;self.high=high;self.speeds=speeds
        self.neutral=neutral;self.axes=axes;self.plan=neck_plan
        self.owner=uuid.uuid4().hex;self.episode=None;self.seen=deque(maxlen=64)
        self.phase='idle';self.previous=None;self.last=None;self.settled_since=None
        self.completed=None;self.main_x=None

    def duration(self,deltas):
        times=[.16]
        for sid,delta in deltas.items():
            distance=abs(delta)
            times.append(1.875*distance/self.speeds[sid])
            if sid in (32,33,34):
                times.extend((math.sqrt(5.774*distance/self.plan['max_acceleration_counts_s2']),
                    (60*distance/self.plan['max_jerk_counts_s3'])**(1/3)))
        return max(times)

    @staticmethod
    def blend(u):
        u=max(0.,min(1.,u));return u*u*u*(10+u*(-15+6*u))

    def positions(self,values):return {sid:float(values[self.ids.index(sid)]) for sid in self.axes}

    def leg(self,start,goal,now,duration=None):
        self.origin=dict(start);self.goal=dict(goal);self.leg_start=now
        self.leg_seconds=max(self.duration({s:goal[s]-start[s] for s in start}),duration or 0.)

    def sample(self,now):
        u=(now-self.leg_start)/self.leg_seconds;weight=self.blend(u)
        return {s:self.origin[s]+weight*(self.goal[s]-self.origin[s]) for s in self.origin},u>=1

    def start(self,request,current,now):
        self.episode=dict(request);self.saved=self.positions(current);self.seen.append(request['id'])
        self.completed=None;self.phase='seek_out' if request['action']=='seek' else 'out'
        target=dict(self.saved)
        yaw=32;desired=2047+request['bearing_deg']*self.c['yaw_counts_per_deg']
        target[yaw]=max(self.low[yaw],min(self.high[yaw],desired))
        delta=target[yaw]-self.saved[yaw]
        for sid,axis in self.axes.items():
            if sid==32 or axis.get('coordinate')!='x':continue
            endpoint=axis['positive'] if delta>=0 else axis['negative']
            target[sid]=axis['neutral']+(endpoint-axis['neutral'])*min(1.,abs(request['bearing_deg'])/30)
        # Keep pitch/roll and vertical gaze at the original positions.
        remaining=max(.4,self.c['budget_s']-(now-request['heard_monotonic'])-.12)
        self.dwell=min(self.c['dwell_s'],max(0.,remaining-.4))
        available=max(.2,(remaining-self.dwell)/2)
        self.travel_seconds=available
        scaled={};fractions={}
        for sid in target:
            fraction=1.
            for _ in range(32):
                delta=fraction*(target[sid]-self.saved[sid])
                if self.duration({sid:delta})<=available:break
                fraction*=.85
            scaled[sid]=self.saved[sid]+fraction*(target[sid]-self.saved[sid]);fractions[sid]=fraction
        # Fast eyes need not inherit the slower neck's shortened excursion.
        self.excursion_fraction=fractions[32]
        # The configured budget is an actual out/dwell/back schedule, rather
        # than just an upper bound that can still produce a subsecond twitch.
        self.leg(self.saved,scaled,now,available)

    def step(self,request,current,packet,now):
        position=self.positions(current)
        fresh_packet=isinstance(packet,dict) and 0<=now-packet.get('timestamp_monotonic',-math.inf)<=.3
        fresh_face=fresh_packet and packet.get('target') is not None
        if self.previous is not None and self.last is not None and now>self.last:
            speed=max(abs(position[s]-self.previous[s])/(now-self.last) for s in position)
        else:speed=math.inf
        self.settled_since=(self.settled_since or now) if speed<2 else None
        settled=self.settled_since is not None and now-self.settled_since>=.12
        self.previous=position;self.last=now
        valid_request=(isinstance(request,dict) and isinstance(request.get('id'),str) and 0<len(request['id'])<=64
            and type(request.get('bearing_deg')) in (int,float) and math.isfinite(request['bearing_deg'])
            and -180<=request['bearing_deg']<=180
            and all(type(request.get(k)) in (int,float) and math.isfinite(request[k])
                    and 0<=now-request[k]<=12 for k in ('heard_monotonic','created_monotonic')))
        live=valid_request and recent(request,now,self.c['lease_s']) and request.get('owner')==self.owner
        if self.episode is None and live and request.get('action') in ('glance','seek') and request.get('id') not in self.seen:
            if request['action']=='seek' and fresh_face:
                self.seen.append(request['id']);self.phase='found'
                self.completed=dict(id=request['id'],elapsed_s=now-request['heard_monotonic'],excursion_fraction=0.)
            elif settled:self.start(request,current,now)
        output=None
        if self.episode:
            same=live and request.get('id')==self.episode['id'] and request.get('action')==self.episode['action']
            # Finish a smooth leg before returning, even if its lease disappears.
            # Never splice a new zero-velocity quintic into a moving trajectory.
            if self.phase in ('out','return','search','neutral','seek_out','seek_neutral'):
                output,done=self.sample(now)
                if done:
                    if self.phase=='out':
                        if same:self.phase='hold';self.hold_until=now+self.dwell
                        else:self.phase='return';self.leg(output,self.saved,now)
                    elif self.phase=='return':
                        self.completed=dict(id=self.episode['id'],elapsed_s=now-self.episode['heard_monotonic'],
                            excursion_fraction=self.excursion_fraction)
                        self.phase='check_main';self.check_until=now+.4
                    elif self.phase=='search':
                        if not same:
                            self.phase='return';self.leg(output,self.saved,now)
                        elif fresh_face:self.phase='found'
                        else:
                            self.phase='neutral';self.leg(output,{s:float(self.neutral[self.ids.index(s)]) for s in output},now)
                    elif self.phase=='seek_out':
                        if same and fresh_face:self.phase='found'
                        elif same:self.phase='seek_hold';self.hold_until=now+self.dwell
                        else:
                            self.phase='seek_neutral';self.leg(output,{s:float(self.neutral[self.ids.index(s)]) for s in output},now)
                    else:self.phase='empty'
            elif self.phase=='hold':
                output=dict(self.goal)
                if not same or now>=self.hold_until:
                    self.phase='return';self.leg(output,self.saved,now,self.travel_seconds if same else None)
            elif self.phase=='seek_hold':
                output=dict(self.goal)
                if same and fresh_face:self.phase='found'
                elif not same or now>=self.hold_until:
                    self.phase='seek_neutral'
                    self.leg(output,{s:float(self.neutral[self.ids.index(s)]) for s in output},now,
                        self.travel_seconds if same else None)
            elif self.phase=='check_main':
                output=dict(self.saved)
                if same and request.get('main_present') is True:self.phase='returned'
                elif now>=self.check_until:
                    if not same:self.phase='returned'
                    else:
                        goal=dict(self.saved)
                        goal[32]=max(self.low[32],min(self.high[32],2047+self.episode['bearing_deg']*self.c['yaw_counts_per_deg']))
                        # A has left: reach the bounded sound goal, rather than
                        # giving up before the permitted neck range is explored.
                        self.phase='search';self.leg(output,goal,now,self.c['search_s'])
            if self.phase in ('returned','found','empty'):
                if self.completed is None:
                    self.completed=dict(id=self.episode['id'],elapsed_s=now-self.episode['heard_monotonic'],
                        excursion_fraction=self.excursion_fraction)
                self.episode=None
        target_x=packet['target']['x'] if fresh_face else 0.
        status=dict(owner=self.owner,updated_monotonic=now,phase=self.phase if self.episode else 'idle',
            attention_id=self.episode['id'] if self.episode else (self.completed or {}).get('id'),
            result=self.phase if not self.episode else None,completed=self.completed,settled=settled,
            single_face=fresh_face and packet.get('faces',1)==1,
            main_bearing_deg=angle((position[32]-2047)/self.c['yaw_counts_per_deg']+target_x*self.c['camera_half_fov_deg']))
        status['visitor']=packet.get('visitor') if fresh_face else None
        status['visual_fresh']=bool(fresh_packet)
        status['face_count']=packet.get('faces',int(bool(fresh_face))) if fresh_packet else None
        status['brake_requested']=bool(self.episode is None and live and request.get('action') in ('glance','seek')
            and request.get('id') not in self.seen and not settled)
        return output,status
