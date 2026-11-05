"""Camera lifecycle, local UDP and preview tests without real video devices."""
import copy,json,math,socket,tempfile,time,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from urllib.request import urlopen
import numpy as np
from vivis.paths import ROOT
from vivis.runtime.robot import Robot
from vivis.vision.policy import VERSION,TargetReceiver,validate_packet,GazePolicy
from vivis.vision.detection import select_largest,LatestCamera

def configuration():
    c=Robot().tracking()
    c['control'].update(camera_mount='fixed',mount_confirmed=True,calibration_confirmed=True)
    return c


def packet(stamp, x=.5, y=0., sequence=1):
    return {'schema_version': VERSION, 'sequence': sequence, 'timestamp_monotonic': stamp,
            'target': None if x is None else {'x': x, 'y': y},
            'camera_transform': {'rotation_deg': 0, 'mirror_x': False}}


def face(x, y, width, height, score=.95):
    return [x, y, width, height, x + width * .3, y + height * .35,
            x + width * .7, y + height * .35, x + width * .5, y + height * .5,
            x + width * .3, y + height * .8, x + width * .7, y + height * .8, score]


class SelectionTests(unittest.TestCase):
    def test_small_inference_preserves_original_preview_and_landmark_coordinates(self):
        import cv2
        from vivis.vision.detection import FaceVision
        c=Robot(ROOT/'configs/robots/xiaoshuai/robot.json').tracking();c['detector']['max_width']=320
        for width,height in ((640,480),(641,481)):
            image=np.zeros((height,width,3),dtype=np.uint8);small_height=round(height*320/width)
            detector=Mock();detector.detect.return_value=(1,np.array([face(10,20,100,80)],dtype=np.float32))
            factory=Mock();factory.create.return_value=detector
            with patch('cv2.FaceDetectorYN',factory):
                vision=FaceVision(c,ROOT);packet,jpeg,metrics=vision.detect((1,time.monotonic(),image))
            detector.setInputSize.assert_called_once_with((320,small_height))
            self.assertEqual(detector.detect.call_args.args[0].shape[:2],(small_height,320))
            self.assertEqual(cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR).shape[:2],(height,width))
            target=packet['target']
            np.testing.assert_allclose(target['bbox'],[10*width/320,20*height/small_height,100*width/320,80*height/small_height])
            np.testing.assert_allclose(target['point'],[60*width/320,48*height/small_height])
            self.assertAlmostEqual(target['x'],2*60/320-1);self.assertAlmostEqual(target['y'],2*48/small_height-1)
            self.assertFalse(image.any())

    def test_eye_line_roll_measures_tilt_without_changing_existing_face_target(self):
        for degrees in (-25.,0.,25.):
            row=face(100,100,100,100);angle=math.radians(degrees)
            dx,dy=20*math.cos(angle),20*math.sin(angle)
            row[4:8]=[150-dx,135-dy,150+dx,135+dy]
            selected,_=select_largest([row],640,480)
            self.assertAlmostEqual(selected['eye_line_roll_rad'],angle)
            self.assertEqual(selected['point'],[150.,135.])
            self.assertAlmostEqual(selected['x'],2*150/640-1)
            self.assertAlmostEqual(selected['y'],2*135/480-1)
            row[4:8]=row[6:8]+row[4:6]
            swapped,_=select_largest([row],640,480)
            self.assertAlmostEqual(swapped['eye_line_roll_rad'],angle)

    def test_invalid_eye_line_does_not_remove_an_otherwise_valid_face(self):
        for landmarks in ([0,0,0,0],[150,135,150.01,135],[99,130,170,135]):
            row=face(100,100,100,100);row[4:8]=landmarks
            selected,_=select_largest([row],640,480)
            self.assertIsNotNone(selected)
            self.assertIsNone(selected['eye_line_roll_rad'])

    def test_area_wins_over_confidence_and_distance_to_center(self):
        selected, faces = select_largest([face(300, 200, 40, 50, .99), face(10, 10, 150, 180, .85)], 640, 480)
        self.assertEqual(len(faces), 2)
        self.assertEqual(selected['bbox'], [10, 10, 150, 180])
        self.assertAlmostEqual(selected['area_ratio'], 150 * 180 / (640 * 480))
        self.assertAlmostEqual(selected['point'][1], 10 + 180 * .35)

    def test_clipped_area_and_landmark_fallback(self):
        outside = face(-190, 10, 200, 300)
        inside = face(200, 100, 100, 100)
        inside[4:8] = [-10, -10, -10, -10]
        selected, _ = select_largest([outside, inside], 640, 480)
        self.assertEqual(selected['bbox'], [200, 100, 100, 100])
        self.assertEqual(selected['point'], [250, 150])

    def test_bad_tiny_low_score_and_offscreen_faces_do_not_select(self):
        for faces in [None, [], [face(0, 0, 1, 1)], [face(0, 0, 100, 100, .5)],
                      [face(700, 10, 100, 100)], [[float('nan')] * 15]]:
            self.assertEqual(select_largest(faces, 640, 480), (None, []))


class CameraOpenTests(unittest.TestCase):
    def test_xiaoshuai_rotates_capture_before_detection_and_preview(self):
        import cv2
        config=Robot(ROOT/'configs/robots/xiaoshuai/robot.json').tracking()
        self.assertEqual(config['camera']['rotation_deg'],180)
        camera_config={**config['camera'],'device':0}
        frame=np.arange(12*16*3,dtype=np.uint8).reshape(12,16,3)
        capture=Mock();capture.isOpened.return_value=True;capture.get.return_value=30.
        with patch('cv2.VideoCapture',return_value=capture),patch('vivis.vision.detection.threading.Thread') as thread:
            thread.return_value.is_alive.return_value=False
            camera=LatestCamera(camera_config)
            def once():
                camera.stop.set();return True,frame.copy()
            capture.read.side_effect=once
            camera._capture()
            np.testing.assert_array_equal(camera.get()[2],frame[::-1,::-1])
            camera.close()
        capture.release.assert_called_once()

    def test_missing_camera_is_reported_before_capture(self):
        cv=Mock()
        with patch.dict('sys.modules',{'cv2':cv}),patch('vivis.vision.detection.os.path.exists',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Camera device is missing'):
                LatestCamera(dict(device='/dev/vivis-test-camera'))
        cv.VideoCapture.assert_not_called()

    def test_camera_permission_error_is_distinct_from_busy(self):
        cv=Mock()
        with patch.dict('sys.modules',{'cv2':cv}),patch('vivis.vision.detection.os.path.exists',return_value=True),patch('vivis.vision.detection.os.access',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Camera device access denied'):
                LatestCamera(dict(device='/dev/vivis-test-camera'))
        cv.VideoCapture.assert_not_called()

    def test_capture_failure_releases_device_and_explains_ownership_check(self):
        cv=Mock();cv.VideoCapture.return_value.isOpened.return_value=False
        with patch.dict('sys.modules',{'cv2':cv}),patch('vivis.vision.detection.os.path.exists',return_value=True),patch('vivis.vision.detection.os.access',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'sudo fuser -v'):
                LatestCamera(dict(device='/dev/vivis-test-camera'))
        cv.VideoCapture.return_value.release.assert_called_once()


class TransportTests(unittest.TestCase):
    def test_explicit_status_recording_retains_only_valid_sdk_payloads(self):
        from vivis.hardware.feetech.sdk import OfficialSDKBus,COMM_SUCCESS,COMM_RX_CORRUPT,COMM_RX_TIMEOUT
        from vivis.hardware.feetech.protocol import ServoStatusError,CorruptResponse,ReadTimeout
        bus=OfficialSDKBus.__new__(OfficialSDKBus);bus.sdk=Mock()
        bus.servo_status_notifications=[]
        bus.sdk.readTxRx.return_value=([255,7],COMM_SUCCESS,1)
        with self.assertRaises(ServoStatusError):bus._read_once(4,56,2)
        bus.servo_status_policy='record_only'
        self.assertEqual(bus._read_once(4,56,2),b'\xff\x07')
        self.assertEqual(bus.servo_status_notifications[0]['servo_status'],1)
        for result,exc in [(COMM_RX_CORRUPT,CorruptResponse),(COMM_RX_TIMEOUT,ReadTimeout)]:
            bus.sdk.readTxRx.return_value=([],result,1)
            with self.assertRaises(exc):bus._read_once(4,56,2)
        bus.sdk.readTxRx.return_value=([255],COMM_SUCCESS,1)
        with self.assertRaises(CorruptResponse):bus._read_once(4,56,2)
        self.assertEqual(len(bus.servo_status_notifications),1)

    def test_packet_rejects_old_future_nonfinite_or_non_numeric_targets(self):
        for sample in [packet(9.), packet(10.01), packet(float('nan')), packet(10., float('nan')),
                       packet(10., True), packet(10., 1.1), packet(10., .5, float('inf'))]:
            with self.assertRaises(ValueError):
                validate_packet(sample, 10., .3)
        validate_packet(packet(9.9, None), 10., .3)

    def test_real_udp_ignores_malformed_and_out_of_order_and_has_one_receiver(self):
        config = configuration()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 0))
            config['transport']['port'] = probe.getsockname()[1]
        receiver = TargetReceiver(config)
        try:
            with self.assertRaises(OSError):
                TargetReceiver(config)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                now = time.monotonic()
                address = ('127.0.0.1', config['transport']['port'])
                for data in [b'not-json', b'null', json.dumps(packet(now - .01, .8, sequence=2)).encode(),
                             json.dumps(packet(now - .02, -.5)).encode()]:
                    sender.sendto(data, address)
                latest = receiver.poll(now)
                self.assertEqual(latest['target']['x'], .8)
                self.assertEqual(receiver.rejected, 3)
                begin = time.monotonic()
                self.assertIs(receiver.poll(now), latest)
                self.assertLess(time.monotonic() - begin, .05)
        finally:
            receiver.close()

    def test_no_service_readiness_fails_without_opening_actuators(self):
        config = configuration()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 0))
            config['transport']['port'] = probe.getsockname()[1]
        receiver = TargetReceiver(config)
        try:
            with self.assertRaisesRegex(RuntimeError, 'No fresh camera packets'):
                receiver.wait_ready(seconds=.02)
        finally:
            receiver.close()

    def test_orientation_mismatch_cannot_drive_a_reversed_eye_command(self):
        config = configuration()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 0))
            config['transport']['port'] = probe.getsockname()[1]
        receiver = TargetReceiver(config)
        try:
            data = packet(time.monotonic())
            data['camera_transform']['mirror_x'] = True
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(json.dumps(data).encode(), ('127.0.0.1', config['transport']['port']))
                self.assertIsNone(receiver.poll())
                self.assertEqual(receiver.rejected, 1)
        finally:
            receiver.close()


class PreviewTests(unittest.TestCase):
    def test_local_http_state_is_available_without_any_actuator(self):
        from vivis.vision.preview import Preview
        preview = Preview('127.0.0.1', 0)
        port = preview.server.server_address[1]
        try:
            preview.update({'status': 'tracking', 'actuators_opened': False})
            with urlopen(f'http://127.0.0.1:{port}/state', timeout=1) as response:
                state = json.load(response)
            self.assertEqual(state['status'], 'tracking')
            self.assertIs(state['actuators_opened'], False)
            with urlopen(f'http://127.0.0.1:{port}/', timeout=1) as response:
                self.assertIn('不代表舵机实测位置', response.read().decode())
        finally:
            preview.close()
