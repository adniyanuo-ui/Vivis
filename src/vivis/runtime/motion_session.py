"""Supervisor-bound motion initialization evidence, never a persistent bypass.

Only managed P1 children may reuse it. Standalone examples still initialize.
Fresh feedback, device identity/layout and runtime safety remain mandatory.
"""
import hashlib
import json
import os
from pathlib import Path
import secrets
import time


class MotionSessionError(RuntimeError):pass


def configuration(c=None):
    c=dict(c or {'standby_check_interval_s':60.})
    if set(c)!={'standby_check_interval_s'} or type(c['standby_check_interval_s']) not in (int,float) or not 1<=c['standby_check_interval_s']<=3600:
        raise ValueError('Invalid motion lifecycle inspection interval')
    return c


def robot_digest(robot):
    digest=hashlib.sha256()
    keys=['hardware','homing','telemetry','calibration','tracking','channels']
    if 'parallel_neck' in robot.definition:keys.append('parallel_neck')
    for path in (robot.path,*(robot.resolve(k) for k in keys)):
        digest.update(str(path.resolve()).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def port_identity(port):
    path=Path(port)
    if not str(port).startswith('/dev/'):return {'port':str(port)}
    try:
        s=path.stat()
        return dict(port=str(path.resolve()),device=s.st_rdev,inode=s.st_ino)
    except OSError as exc:raise MotionSessionError('Serial device unavailable; motion initialization invalid') from exc


class MotionSession:
    def __init__(self,path,token,robot,*,parent=True):
        self.path=Path(path);self.token=token;self.robot=robot;self.parent=parent

    @classmethod
    def create(cls,path,robot,policy=None):
        obj=cls(path,secrets.token_hex(24),robot,parent=False)
        obj.write(dict(schema='vivis-motion-session-1.0',token=obj.token,supervisor_pid=os.getpid(),
            robot_path=str(robot.path),
            robot_digest=robot_digest(robot),port_identity=None,
            ready=False,policy=configuration(policy),axes={},fault=None))
        return obj

    def environment(self,action):
        if action not in ('startup','resume','standby'):raise ValueError('Invalid managed preparation action')
        return dict(VIVIS_MOTION_SESSION=str(self.path),VIVIS_MOTION_TOKEN=self.token,VIVIS_MOTION_ACTION=action)

    @classmethod
    def current(cls,robot):
        path=os.environ.get('VIVIS_MOTION_SESSION')
        if not path:return None
        if os.environ.get('VIVIS_MANAGED_SESSION')!='1':raise MotionSessionError('Motion session requires managed owner')
        obj=cls(path,os.environ.get('VIVIS_MOTION_TOKEN'),robot)
        obj.read(check_port=False,check_config=False)
        return obj

    def read(self,*,require_ready=False,check_port=True,check_config=True):
        try:d=json.loads(self.path.read_text())
        except (OSError,ValueError) as exc:raise MotionSessionError('Motion session evidence unavailable') from exc
        owner=os.getppid() if self.parent else os.getpid()
        if (d.get('schema')!='vivis-motion-session-1.0' or not self.token or d.get('token')!=self.token
                or d.get('supervisor_pid')!=owner or check_config and d.get('robot_digest')!=robot_digest(self.robot)):
            raise MotionSessionError('Motion owner/configuration changed; full restart initialization required')
        if check_port and d['port_identity']!=port_identity(self.robot.hardware['runtime']['port']):
            raise MotionSessionError('USB serial device changed; initialization invalid')
        if require_ready and not d.get('ready'):raise MotionSessionError('Motion session not ready: '+str(d.get('fault')))
        return d

    def write(self,data):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_name(self.path.name+'.'+str(os.getpid())+'.tmp')
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(self.path)

    def initialized(self,report):
        if report.get('self_check',{}).get('state')!='passed' or not report.get('homing',{}).get('completed'):
            raise MotionSessionError('Both startup self-check and neutral homing must succeed')
        d=self.read(check_port=False)
        snapshots=report.get('hardware_restore',{}).get('snapshots_after') or report['startup_snapshots']
        active=set(report.get('active_ids',[s['id'] for s in snapshots]))
        d.update(ready=True,fault=None,initialized_monotonic=time.monotonic(),
            port_identity=port_identity(self.robot.hardware['runtime']['port']),
            axes={str(s['id']):dict(config=s['config_0_39_hex'],acceleration=bytes.fromhex(s['state_40_70_hex'])[1],
                active=s['id'] in active,
                torque_limit=int.from_bytes(bytes.fromhex(s['state_40_70_hex'])[8:10],'little')) for s in snapshots})
        self.write(d)

    def invalidate(self,reason):
        d=self.read(check_port=False,check_config=False);d.update(ready=False,fault=str(reason));self.write(d)

    def validate_snapshot(self,snapshot):
        d=self.read(require_ready=True);axis=d['axes'].get(str(snapshot['id']))
        ram=bytes.fromhex(snapshot['state_40_70_hex'])
        if (axis is None or snapshot['config_0_39_hex']!=axis['config']
                or ram[1]!=axis['acceleration'] or int.from_bytes(ram[8:10],'little')!=axis['torque_limit']
                or axis['active'] and int.from_bytes(ram[6:8],'little')==0):
            raise MotionSessionError(f'ID{snapshot["id"]}: device configuration changed since startup')


def preparation_action(robot):
    session=MotionSession.current(robot)
    action=os.environ.get('VIVIS_MOTION_ACTION','startup') if session else 'startup'
    if action not in ('startup','resume','standby'):raise MotionSessionError('Unknown preparation action')
    return session,action


def maintain_standby(session,action,prepare,play,guard,*,clock=time.monotonic,wait=time.sleep):
    """Single motion owner: inspect between bounded idle segments, hold on faults.

    prepare(action) is synchronous and owns the serial bus only while play is
    stopped. A failed cold startup escapes; warm standby faults stay latched.
    No stale program or actuator command is replayed during recovery.
    """
    interval=session.read(check_port=False,check_config=False)['policy']['standby_check_interval_s']
    fault=False;first=True;next_check=clock();segment=0
    while True:
        guard()
        try:
            if first:
                prepare(action)
                first=False
            else:
                prepare('inspect')
                if fault:
                    prepare('startup')
                    print('STANDBY RECOVERED 自检与回中均成功，恢复待机',flush=True)
            fault=False;next_check=clock()+interval
            segment+=1
            playback=play(max(.001,next_check-clock()),segment)
            if playback is not None and not playback['completed']:
                raise RuntimeError(playback.get('error') or 'Standby playback did not complete')
        except RuntimeError as exc:
            from vivis.runtime.cancellation import operator_cancelled
            if operator_cancelled(exc):raise
            if first and action=='startup':raise  # A failed boot is not ready.
            first=False;fault=True;session.invalidate(exc)
            print('STANDBY FAULT '+str(exc).replace('\n',' ')[:1000],flush=True)
            next_check=clock()+interval
        while clock()<next_check:
            guard();wait(min(.05,next_check-clock()))


def for_profile(profile):
    path=os.environ.get('VIVIS_MOTION_SESSION')
    if not path:return None
    from vivis.runtime.robot import Robot
    try:robot=Robot(json.loads(Path(path).read_text())['robot_path'])
    except (OSError,ValueError,KeyError) as exc:raise MotionSessionError('Managed robot configuration unavailable') from exc
    session=MotionSession.current(robot);session.read(require_ready=True)
    if (profile['port']!=robot.hardware['runtime']['port'] or profile['baud']!=robot.hardware['runtime']['baud']
            or profile.get('calibration_sha256')!=robot.calibration_sha256):
        raise MotionSessionError('Playback profile differs from initialized robot')
    return session
