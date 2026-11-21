"""One P1 camera owner; mode requests never own or restart capture hardware."""
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import uuid

from vivis.paths import ROOT
from vivis.vision.policy import validate_config


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    fd=os.open(temporary,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(data,stream,ensure_ascii=False)
    temporary.replace(path)


def validate_mode(config,base,landmarker):
    if not isinstance(config,dict):raise ValueError('Vision configuration must be an object')
    validate_config(config)
    for key in ('camera','detector','transport'):
        if config[key]!=base[key]:raise ValueError('Persistent vision requires unchanged '+key+'; restart manager after configuration changes')
    known=config.get('familiar_faces',{})
    if known.get('enabled') and known!=base.get('familiar_faces',{}):
        raise ValueError('Persistent recognition configuration changed; restart manager')
    for key in ('social_smile','face_expression'):
        setting=config.get(key,{})
        if setting.get('enabled'):
            if any(setting.get(k)!=landmarker.get(k) for k in ('model','sha256')):
                raise ValueError('Persistent vision requires the preloaded landmark model')
            if key=='face_expression' and setting.get('face_selection')!='largest':
                raise ValueError('Expression follow requires largest-face selection')
            if key=='social_smile':
                from vivis.interaction.smile import SmileGate,selected_face_allowed
                selected_face_allowed(setting,0)
                SmileGate(setting['gate'],max_gap_s=config['transport']['max_age_s'])


class ModeControl:
    """Service-side atomic mode admission, consumed only at frame boundaries."""
    def __init__(self,path,base):
        self.path=Path(path);self.manifest=json.loads(self.path.read_text())
        if self.manifest['manager_pid']!=os.getppid():raise RuntimeError('Persistent camera supervisor changed')
        self.base=copy.deepcopy(base);self.landmarker=self.manifest['landmarker']
        self.request_path=self.path.with_name('request.json');self.status_path=self.path.with_name('mode_status.json')
        self.stamp=None;self.request_id=None;self.applied=0

    def poll(self):
        if os.getppid()!=self.manifest['manager_pid']:raise RuntimeError('Persistent camera supervisor exited')
        try:
            stat=self.request_path.stat();stamp=(stat.st_dev,stat.st_ino,stat.st_mtime_ns)
        except FileNotFoundError:return None
        if stamp==self.stamp:return None
        self.stamp=stamp;request=json.loads(self.request_path.read_text())
        if request.get('token')!=self.manifest['token']:raise RuntimeError('Unverified persistent vision request')
        try:
            raw=bytes.fromhex(request['config_bytes_hex']);config=json.loads(raw)
            validate_mode(config,self.base,self.landmarker)
            # The consumer hash describes its exact input, not a reconstructed JSON encoding.
            digest=hashlib.sha256(raw).hexdigest()
            if digest!=request['config_sha256']:
                raise ValueError('Persistent vision request/configuration mismatch')
        except (ValueError,KeyError,TypeError) as exc:
            atomic_json(self.status_path,dict(request_id=request.get('request_id'),accepted=False,error=str(exc)))
            return None
        self.request_id=request['request_id'];self.applied+=1
        return config,digest,self.request_id

    def applied_frame(self,digest):
        atomic_json(self.status_path,dict(request_id=self.request_id,accepted=True,config_sha256=digest))


class PersistentClient:
    def __init__(self,path,token):
        self.path=Path(path);self.manifest=json.loads(self.path.read_text())
        if (os.environ.get('VIVIS_MANAGED_SESSION')!='1' or token!=self.manifest['token']
                or os.getppid()!=self.manifest['manager_pid']):
            raise RuntimeError('Shared camera requires its current managed supervisor')
        self.request_id=uuid.uuid4().hex;self.status_path=self.path.with_name('mode_status.json')
        self.acknowledged=False

    def request(self,config_path):
        raw=Path(config_path).read_bytes();config=json.loads(raw)
        validate_mode(config,self.manifest['base'],self.manifest['landmarker'])
        atomic_json(self.path.with_name('request.json'),dict(token=self.manifest['token'],request_id=self.request_id,
            config_bytes_hex=raw.hex(),config_sha256=hashlib.sha256(raw).hexdigest()))

    def check(self):
        if os.getppid()!=self.manifest['manager_pid']:raise RuntimeError('Shared camera supervisor exited')
        try:os.kill(self.manifest['service_pid'],0)
        except ProcessLookupError:raise RuntimeError('常驻视觉进程已退出；查看 '+str(self.path.parent/'vision.log')) from None
        if self.acknowledged:return
        try:status=json.loads(self.status_path.read_text())
        except FileNotFoundError:return
        if status.get('request_id')==self.request_id and not status['accepted']:
            raise RuntimeError('Shared vision mode rejected: '+status['error'])
        if status.get('request_id')==self.request_id:self.acknowledged=True

    def matches(self,state):
        return isinstance(state,dict) and state.get('vision_request_id')==self.request_id


class PersistentVision:
    """Supervisor-owned process; no serial, audio or actuator access."""
    def __init__(self,robot,output):
        from vivis.interaction.knowledge import Knowledge
        self.folder=Path(output)/'persistent_vision';self.folder.mkdir()
        base=robot.tracking();knowledge=Knowledge(robot.path)
        base['social_smile']={'enabled':False};base['face_expression']={'enabled':False}
        if knowledge.recognition:base['familiar_faces']=knowledge.recognition
        perception=json.loads((ROOT/'configs/perception/face_landmarker.json').read_text())
        self.path=self.folder/'owner.json';self.token=secrets.token_hex(24)
        self.manifest=dict(manager_pid=os.getpid(),token=self.token,base=base,
            landmarker={k:perception[k] for k in ('model','sha256')})
        atomic_json(self.path,self.manifest)
        config_path=self.folder/'initial_config.json';atomic_json(config_path,base)
        self.log=(self.folder/'vision.log').open('w');self.process=None
        try:
            self.process=subprocess.Popen([str(ROOT/perception['python']),'-m','vivis.vision.service',
                '--config',str(config_path),'--output',str(self.folder/'vision'),'--managed-owner',str(self.path)],
                cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONNOUSERSITE':'1'},
                stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            self.manifest['service_pid']=self.process.pid;atomic_json(self.path,self.manifest)
        except BaseException:
            self.close();raise

    def environment(self):return dict(VIVIS_VISION_OWNER=str(self.path),VIVIS_VISION_TOKEN=self.token)

    def park(self):
        raw=json.dumps(self.manifest['base']).encode()
        atomic_json(self.path.with_name('request.json'),dict(token=self.token,request_id=uuid.uuid4().hex,
            config_bytes_hex=raw.hex(),config_sha256=hashlib.sha256(raw).hexdigest()))

    def poll(self):return self.process.poll()

    def close(self):
        try:
            if self.process is not None:
                try:
                    if self.process.poll() is None:self.process.terminate()
                    self.process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=2)
                # Reap recognizer helper descendants even if the camera parent exited.
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        finally:self.log.close()
