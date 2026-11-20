"""Camera startup/cleanup regressions; no physical camera or actuator access."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch


from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.runtime.developer import VisionReadiness,vision_service
from vivis.vision.detection import LatestCamera


class CameraSettingsTests(unittest.TestCase):
    def capture(self):
        import cv2
        values={cv2.CAP_PROP_FOURCC:cv2.VideoWriter_fourcc(*'MJPG'),cv2.CAP_PROP_FRAME_WIDTH:640,
                cv2.CAP_PROP_FRAME_HEIGHT:480,cv2.CAP_PROP_FPS:30,cv2.CAP_PROP_BUFFERSIZE:1,
                cv2.CAP_PROP_AUTOFOCUS:1}
        capture=Mock();capture.get.side_effect=lambda key:values.get(key,0)
        capture.isOpened.return_value=True;capture.set.return_value=True
        return cv2,capture,values

    def test_matching_video_settings_and_focus_never_issue_redundant_set(self):
        cv,capture,values=self.capture();progress=[]
        with patch.object(cv,'VideoCapture',return_value=capture),patch('vivis.vision.detection.threading.Thread'):
            camera=LatestCamera(dict(device=0,width=640,height=480,fps=30,autofocus=True),
                                progress=lambda stage,**data:progress.append((stage,data)))
        capture.set.assert_not_called()
        self.assertIsNone(camera.actual['controls']['autofocus']['accepted'])
        self.assertTrue(camera.actual['controls']['autofocus']['write_skipped'])
        self.assertTrue(any(stage=='camera_setting_unchanged' for stage,_ in progress))

    def test_resize_sets_both_dimensions_even_when_one_matches(self):
        cv,capture,values=self.capture();values[cv.CAP_PROP_FRAME_WIDTH]=320
        with patch.object(cv,'VideoCapture',return_value=capture),patch('vivis.vision.detection.threading.Thread'):
            LatestCamera(dict(device=0,width=640,height=480,fps=30))
        self.assertEqual(capture.set.call_args_list,[((cv.CAP_PROP_FRAME_WIDTH,640),),((cv.CAP_PROP_FRAME_HEIGHT,480),)])

    def test_changed_focus_still_applies_requested_setting(self):
        cv,capture,values=self.capture();values[cv.CAP_PROP_AUTOFOCUS]=0
        with patch.object(cv,'VideoCapture',return_value=capture),patch('vivis.vision.detection.threading.Thread'):
            camera=LatestCamera(dict(device=0,width=640,height=480,fps=30,autofocus=True))
        capture.set.assert_called_once_with(cv.CAP_PROP_AUTOFOCUS,1.)
        self.assertTrue(camera.actual['controls']['autofocus']['accepted'])


class VisionCleanupTests(unittest.TestCase):
    def test_stuck_cleanup_preserves_primary_exception_and_reports_unreleased_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=root/'tracking.json';config.write_text(json.dumps(Robot().tracking()))
            child=Mock();child.pid=12345;child.poll.return_value=None
            child.wait.side_effect=[subprocess.TimeoutExpired('camera',4),subprocess.TimeoutExpired('camera',2)]
            original=RuntimeError('original camera timeout')
            with patch('vivis.runtime.developer.preview_state',return_value=None),\
                 patch('vivis.runtime.developer.subprocess.Popen',return_value=child),\
                 patch.object(VisionReadiness,'wait_ready'),redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError) as result:
                    with vision_service(config,root,True):raise original
            self.assertIs(result.exception,original)
            if hasattr(original,'add_note'):self.assertTrue(any('12345' in note for note in original.__notes__))
            child.kill.assert_called_once()

    def test_stuck_cleanup_on_normal_exit_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=root/'tracking.json';config.write_text(json.dumps(Robot().tracking()))
            child=Mock();child.pid=12345;child.poll.return_value=None
            child.wait.side_effect=[subprocess.TimeoutExpired('camera',4),subprocess.TimeoutExpired('camera',2)]
            with patch('vivis.runtime.developer.preview_state',return_value=None),\
                 patch('vivis.runtime.developer.subprocess.Popen',return_value=child),\
                 patch.object(VisionReadiness,'wait_ready'),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'不能确认设备已释放'):
                    with vision_service(config,root,True):pass

    def test_failure_names_actual_startup_stage_and_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=Robot().tracking();child=Mock();child.poll.return_value=1
            status=dict(config_sha256='verified',stage='failed',error='Camera did not provide a fresh frame')
            (root/'vision_startup.json').write_text(json.dumps(status))
            handle=VisionReadiness(child,config,'verified',root)
            with self.assertRaisesRegex(RuntimeError,'Camera did not provide a fresh frame'):handle.check()
            status['config_sha256']='unrelated';(root/'vision_startup.json').write_text(json.dumps(status))
            with self.assertRaises(RuntimeError) as result:handle.check()
            self.assertNotIn('Camera did not provide',str(result.exception))


class FirstFrameBudgetTests(unittest.TestCase):
    def run_service(self,has_frame):
        from vivis.vision import service
        from vivis.vision.policy import VERSION
        clock=[0.]
        class Stop:
            def set(self):pass
            def is_set(self):return has_frame and clock[0]>=10.15
            def wait(self,delay):clock[0]+=max(delay,.02)
        camera=Mock();camera.actual={};calls=[0]
        def get():
            calls[0]+=1
            return None if not has_frame or calls[0]==1 else (1,10.02,None)
        camera.get.side_effect=get
        detector=Mock()
        detector.detect.side_effect=lambda sample:(dict(schema_version=VERSION,sequence=1,
            timestamp_monotonic=sample[1],target=None),b'jpeg',dict(processing_ms=1.,faces=0))
        def initialize(*args):clock[0]=10.;return detector
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);file=root/'config.json';file.write_text(json.dumps(Robot().tracking()))
            presence=root/'configs/perception/speech_presence.json';presence.parent.mkdir(parents=True)
            presence.write_text((ROOT/'configs/perception/speech_presence.json').read_text())
            with patch.object(service,'ROOT',root),patch.object(service,'LatestCamera',return_value=camera),\
                 patch.object(service,'FaceVision',side_effect=initialize),patch.object(service,'Preview'),\
                 patch.object(service.socket,'socket'),patch.object(service.threading,'Event',Stop),\
                 patch.object(service.time,'monotonic',side_effect=lambda:clock[0]),redirect_stdout(io.StringIO()):
                if has_frame:service.main(['--config',str(file),'--output',str(root/'vision'),'--seconds','20'])
                else:
                    with self.assertRaisesRegex(RuntimeError,'five seconds'):
                        service.main(['--config',str(file),'--output',str(root/'vision'),'--seconds','20'])
            summary=json.loads((root/'vision/summary.json').read_text())
            state=json.loads((root/'vision_startup.json').read_text())
            return summary,state,clock[0]

    def test_slow_model_initialization_does_not_consume_first_frame_budget(self):
        summary,state,elapsed=self.run_service(True)
        self.assertTrue(summary['completed']);self.assertEqual(summary['frames'],1)
        self.assertEqual(state['stage'],'first_frame_detected')

    def test_missing_frames_still_fail_after_actual_five_second_budget(self):
        summary,state,elapsed=self.run_service(False)
        self.assertFalse(summary['completed']);self.assertEqual(summary['frames'],0)
        self.assertTrue(15.<elapsed<15.1);self.assertEqual(state['stage'],'failed')


class DiagnosticIsolationTests(unittest.TestCase):
    def test_production_probe_uses_runtime_capture_and_reports_incomplete_release(self):
        import numpy as np
        spec=importlib.util.spec_from_file_location('camera_production_diagnostic',ROOT/'tools/check_camera_lifecycle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for alive in (False,True):
            clock=[10.];events=[];camera=Mock();camera.actual=dict(width=640,height=480)
            camera.get.return_value=(1,10.,np.zeros((480,640,3),dtype=np.uint8))
            camera.thread.is_alive.return_value=alive
            def event(stage,**data):events.append((stage,data))
            with patch('vivis.vision.detection.LatestCamera',return_value=camera) as capture,\
                 patch.object(module.time,'monotonic',side_effect=lambda:clock[0]),\
                 patch.object(module.time,'sleep',side_effect=lambda _:clock.__setitem__(0,clock[0]+.05)):
                result=module.production_worker({'device':'test-camera'},.1,event,10.)
            capture.assert_called_once_with({'device':'test-camera'},progress=event)
            camera.close.assert_called_once()
            self.assertEqual(result,4 if alive else 0)
            self.assertEqual(events[-1][0],'release_incomplete' if alive else 'release_complete')
            self.assertEqual(next(d for s,d in events if s=='capture_complete')['frames'],1)

    def test_timed_out_owned_worker_is_reaped_without_camera_or_new_trials(self):
        spec=importlib.util.spec_from_file_location('camera_diagnostic',ROOT/'tools/check_camera_lifecycle.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with (root/'lock').open('a') as lock:
                result=module.trial([sys.executable,'-c','import time; time.sleep(30)'],root/'worker.log',.1,lock.fileno())
            self.assertTrue(result['timed_out']);self.assertFalse(result['worker_unreaped'])
            self.assertIsNotNone(result['exit_code']);self.assertIsNone(result['capture'])


if __name__=='__main__':unittest.main()
