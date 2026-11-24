"""Resident camera transitions with real mode admission, fake capture/models only."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.vision.persistent import ModeControl,PersistentClient,atomic_json,validate_mode
from vivis.vision.policy import TargetReceiver


def request(path,config,request_id='mode',token='test'):
    raw=json.dumps(config).encode()
    atomic_json(Path(path).with_name('request.json'),dict(token=token,request_id=request_id,
        config_bytes_hex=raw.hex(),config_sha256=hashlib.sha256(raw).hexdigest()))


class PersistentTests(unittest.TestCase):
    def fixture(self,root):
        base=Robot().tracking();base.update(social_smile={'enabled':False},face_expression={'enabled':False})
        landmark=dict(model='same-model.task',sha256='verified')
        owner=Path(root)/'owner.json'
        atomic_json(owner,dict(manager_pid=os.getpid(),service_pid=os.getpid(),token='test',base=base,landmarker=landmark))
        return base,landmark,owner

    def test_canonical_xiaoshuai_and_mojia_modes_share_camera_and_models(self):
        from vivis.interaction.knowledge import Knowledge
        from vivis.interaction.standby import plan
        from vivis.interaction.follow import plan as follow
        landmark=json.loads((ROOT/'configs/perception/face_landmarker.json').read_text())
        for name in ('xiaoshuai','mojia'):
            robot=Robot(ROOT/f'configs/robots/{name}/robot.json');base=robot.tracking();knowledge=Knowledge(robot.path)
            if knowledge.recognition:base['familiar_faces']=knowledge.recognition
            for mode in ('standby','interaction','expression-follow','song'):
                with self.subTest(robot=name,mode=mode):
                    if mode in ('standby','interaction'):
                        settings=json.loads((ROOT/f'configs/interaction/{mode}.json').read_text())
                        if mode=='standby':settings['presence_only']=True
                        config=plan(robot,settings,1.)[4].tracking
                    elif mode=='expression-follow':
                        settings=json.loads((ROOT/'configs/interaction/expression_follow.json').read_text())
                        config=follow(robot,settings,1.)['tracking']
                    else:config=robot.tracking()
                    validate_mode(config,base,landmark)

    def test_incompatible_camera_or_model_rejected_without_changing_active_mode(self):
        with tempfile.TemporaryDirectory() as root,patch('os.getppid',return_value=os.getpid()):
            base,landmark,owner=self.fixture(root);control=ModeControl(owner,base)
            for key,value in [('camera',{**base['camera'],'rotation_deg':90}),
                              ('detector',{**base['detector'],'hz':30}),
                              ('face_expression',dict(enabled=True,model='other',sha256='bad',face_selection='largest'))]:
                with self.subTest(key=key):
                    invalid=copy.deepcopy(base);invalid[key]=value;request(owner,invalid)
                    self.assertIsNone(control.poll());self.assertFalse(json.loads(control.status_path.read_text())['accepted'])
                    self.assertIsNone(control.request_id)
            request(owner,base,'valid');self.assertEqual(control.poll()[2],'valid')
            self.assertIsNone(control.poll());self.assertEqual(control.applied,1)

    def test_atomic_mode_change_detected_even_with_same_file_timestamp(self):
        with tempfile.TemporaryDirectory() as root,patch('os.getppid',return_value=os.getpid()):
            base,landmark,owner=self.fixture(root);control=ModeControl(owner,base)
            for identifier in ('first','second','third'):
                request(owner,base,identifier);os.utime(control.request_path,ns=(1000000000,1000000000))
                self.assertEqual(control.poll()[2],identifier)
            self.assertEqual(control.applied,3)

    def test_client_needs_owner_and_exact_frame_ack_and_cannot_stop_service(self):
        from vivis.runtime.developer import VisionReadiness,vision_service
        with tempfile.TemporaryDirectory() as folder,patch('os.getppid',return_value=os.getpid()):
            base,landmark,owner=self.fixture(folder);file=Path(folder)/'config.json';atomic_json(file,base)
            env=dict(VIVIS_MANAGED_SESSION='1',VIVIS_VISION_OWNER=str(owner),VIVIS_VISION_TOKEN='test')
            with patch.dict(os.environ,env),patch('vivis.runtime.developer.subprocess.Popen') as launch,\
                 patch.object(VisionReadiness,'wait_ready'),redirect_stdout(io.StringIO()):
                with vision_service(file,Path(folder),True) as handle:
                    self.assertIsNotNone(handle.managed)
                    self.assertEqual(os.environ['VIVIS_VISION_REQUEST'],handle.managed.request_id)
                    self.assertFalse(handle.managed.matches({'vision_request_id':'old'}))
                    request_id=handle.managed.request_id
                    atomic_json(handle.managed.status_path,dict(request_id=request_id,accepted=False,error='wrong model'))
                    with self.assertRaisesRegex(RuntimeError,'wrong model'):handle.check()
                launch.assert_not_called();self.assertNotIn('VIVIS_VISION_REQUEST',os.environ)
            with patch.dict(os.environ,{'VIVIS_MANAGED_SESSION':'1'}),self.assertRaises(RuntimeError):PersistentClient(owner,'wrong')

    def test_readiness_waits_for_new_request_frame_not_old_matching_config(self):
        from vivis.runtime.developer import VisionReadiness
        with tempfile.TemporaryDirectory() as folder,patch('os.getppid',return_value=os.getpid()):
            base,_,owner=self.fixture(folder)
            with patch.dict(os.environ,{'VIVIS_MANAGED_SESSION':'1'}):client=PersistentClient(owner,'test')
            state=dict(config_sha256='digest',target_protocol_version='vivis-person-target-1.0',
                target_features=['face_xy','eye_line_roll'],camera_transform={k:base['camera'][k] for k in ('rotation_deg','mirror_x')},
                camera_device=base['camera']['device'],camera_eye=base['control'].get('camera_eye'),
                service_age_s=0.,frame_age_ms=1.,status='idle')
            handle=VisionReadiness(None,base,'digest',Path(folder),managed=client)
            with patch('vivis.runtime.developer.preview_state',side_effect=[dict(state,vision_request_id='old'),
                dict(state,vision_request_id=client.request_id)]) as peek,patch('vivis.runtime.developer.time.sleep') as wait:
                handle.wait_ready()
            self.assertEqual(peek.call_count,2);wait.assert_called_once();self.assertTrue(handle.ready)

    def test_udp_drops_previous_mode_packets_even_when_fresh(self):
        config=Robot().tracking()
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1',0));config['transport']['port']=probe.getsockname()[1]
        with patch.dict(os.environ,{'VIVIS_VISION_OWNER':'managed','VIVIS_VISION_REQUEST':'new'}):receiver=TargetReceiver(config)
        try:
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sender:
                now=time.monotonic()
                packet=dict(schema_version='vivis-person-target-1.0',sequence=1,timestamp_monotonic=now,
                    target=None,camera_transform=receiver.expected_transform,vision_request_id='old')
                sender.sendto(json.dumps(packet).encode(),('127.0.0.1',config['transport']['port']))
                self.assertIsNone(receiver.poll(now));self.assertEqual(receiver.rejected,1)
                packet.update(vision_request_id='new',sequence=2)
                sender.sendto(json.dumps(packet).encode(),('127.0.0.1',config['transport']['port']))
                self.assertEqual(receiver.poll(now)['sequence'],2)
        finally:receiver.close()

    def test_real_service_keeps_one_capture_and_model_across_four_mode_changes(self):
        from vivis.vision import service
        with tempfile.TemporaryDirectory() as folder,patch('os.getppid',return_value=os.getpid()):
            base,landmark,owner=self.fixture(folder);root=Path(folder)
            # The owner enables identity once. A song mode can omit its local
            # block, but must not suspend identification for the live dialogue.
            base['familiar_faces']={'enabled':True}
            presence=root/'configs/perception/speech_presence.json';presence.parent.mkdir(parents=True)
            presence.write_bytes((ROOT/'configs/perception/speech_presence.json').read_bytes())
            interaction=copy.deepcopy(base)
            interaction['social_smile']=dict(enabled=True,face_selection='largest',**landmark,
                gate=json.loads((ROOT/'configs/interaction/interaction.json').read_text())['social_response']['gate'])
            expression=copy.deepcopy(base);expression['face_expression']=dict(enabled=True,face_selection='largest',**landmark)
            file=root/'config.json';atomic_json(file,base)
            song=copy.deepcopy(base);song.pop('familiar_faces')
            calls=[0];states=[];sequence_modes={2:interaction,4:expression,6:song,8:interaction}
            class Stop:
                def set(self):pass
                def is_set(self):return calls[0]>=10
                def wait(self,_):pass
            camera=Mock(actual={})
            def get():
                calls[0]+=1
                if calls[0] in sequence_modes:request(owner,sequence_modes[calls[0]],str(calls[0]))
                return calls[0],time.monotonic(),None
            camera.get.side_effect=get
            detector=Mock()
            detector.detect.side_effect=lambda sample:(dict(schema_version='vivis-person-target-1.0',sequence=sample[0],
                timestamp_monotonic=sample[1],target=dict(x=0.,y=0.,bbox=[0,0,100,100])),b'jpeg',dict(processing_ms=1.,faces=1))
            landmarker=Mock()
            landmarker.observe.side_effect=lambda sample,target:dict(schema_version='vivis-face-expression-1.0',
                sequence=sample[0],timestamp_monotonic=sample[1],valid=True,weights=dict(mouthSmileLeft=.6,mouthSmileRight=.6))
            people=Mock(error=None);people.observation.return_value=dict(available=True,observed_monotonic=None,persons=0)
            familiar=Mock(available=True,error=None,photo_error=None,photo_pool=None,brightness_samples=None,brightness_error=None)
            familiar.c={'backend':'seeta6'}
            familiar.step.return_value=dict(state='recognized',person_id='alpha',similarity=.9,observed_monotonic=time.monotonic())
            preview=Mock();preview.update.side_effect=lambda state,jpeg=None:states.append(state)
            with patch.object(service,'ROOT',root),patch.object(service,'LatestCamera',return_value=camera) as capture,\
                 patch.object(service,'FaceVision',return_value=detector) as detection,\
                 patch('vivis.vision.landmarks.FaceLandmarkerDetector',return_value=landmarker) as model,\
                 patch('vivis.vision.person.PersonWorker',return_value=people),patch.object(service,'Preview',return_value=preview),\
                 patch('vivis.vision.familiar.FamiliarFaces',return_value=familiar) as recognizer,\
                 patch.object(service.socket,'socket'),patch.object(service.threading,'Event',Stop),redirect_stdout(io.StringIO()):
                service.main(['--config',str(file),'--managed-owner',str(owner),'--output',str(root/'vision')])
            capture.assert_called_once();detection.assert_called_once();model.assert_called_once()
            recognizer.assert_called_once();self.assertEqual(familiar.step.call_count,10)
            self.assertTrue(all(s['familiar_identity']['person_id']=='alpha' for s in states))
            camera.close.assert_called_once();landmarker.close.assert_called_once()
            observed=[c.args[0][0] for c in landmarker.observe.call_args_list]
            self.assertEqual(observed,[3,4,5,6,9,10])  # No inference in either standby segment.
            self.assertEqual(len(states),10)
            self.assertEqual(len({s['vision_pid'] for s in states}),1)
            self.assertFalse(states[6]['landmark_inference_enabled']);self.assertTrue(states[8]['landmark_inference_enabled'])
            summary=json.loads((root/'vision/summary.json').read_text())
            self.assertEqual(summary['mode_switches'],4);self.assertEqual(summary['camera_open_count'],1)
            self.assertEqual(summary['landmark_inference_frames'],6)


if __name__=='__main__':unittest.main()
