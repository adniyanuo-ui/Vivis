"""Persistent credentials and service policy without cloud, devices or sudo."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vivis.paths import ROOT
from vivis.speech.credentials import load_key,save_key,validate


def tool(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class PersistentCredentialTests(unittest.TestCase):
    def test_save_and_reload_mode_and_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config/key';save_key('test-placeholder-one',path)
            self.assertEqual(path.stat().st_mode&0o777,0o600)
            self.assertEqual(load_key(environ={},path=path),'test-placeholder-one')
            save_key('test-placeholder-two',path)
            self.assertEqual(load_key(environ={},path=path),'test-placeholder-two')
            self.assertEqual(list(path.parent.iterdir()),[path])

    def test_environment_and_systemd_credential_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);save_key('credential-placeholder',folder/'dashscope_api_key')
            (folder/'dashscope_api_key').chmod(0o440)
            self.assertEqual(load_key(environ={'CREDENTIALS_DIRECTORY':tmp}),'credential-placeholder')
            self.assertEqual(load_key(environ={'DASHSCOPE_API_KEY':'environment-placeholder','CREDENTIALS_DIRECTORY':tmp}),'environment-placeholder')
            self.assertEqual(load_key(environ={},path=folder/'missing'),'')

    def test_systemd_readonly_group_read_does_not_relax_regular_key_file_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'dashscope_api_key';save_key('credential-placeholder',path)
            path.chmod(0o440)
            self.assertEqual(load_key(environ={'CREDENTIALS_DIRECTORY':tmp}),'credential-placeholder')
            with self.assertRaises(ValueError):load_key(environ={},path=path)
            for mode in (0o444,0o640,0o660):
                path.chmod(mode)
                with self.assertRaises(ValueError):load_key(environ={'CREDENTIALS_DIRECTORY':tmp})

    def test_reject_insecure_files_symlink_and_oversized_content_without_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'key';save_key('SECRET_CANARY',path);path.chmod(0o644)
            with self.assertRaises(ValueError) as caught:load_key(environ={},path=path)
            self.assertNotIn('SECRET_CANARY',str(caught.exception))
            link=Path(tmp)/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):load_key(environ={},path=link)
            path.chmod(0o600);path.write_text('x'*5000)
            with self.assertRaises(ValueError):load_key(environ={},path=path)
        for value in ('',' bad','bad\nvalue','中文','bad\x00'):
            with self.assertRaises(ValueError):validate(value)

    def test_service_retains_standby_and_safe_shutdown_without_recording_or_retry_loop(self):
        module=tool('install_autostart')
        text=module.render(ROOT,Path('/home/unituser'),'unituser',ROOT/'configs/robots/xiaoshuai/robot.json')
        self.assertNotIn('@ROOT@',text);self.assertNotIn('--ask-key',text);self.assertNotIn('--record-audio',text)
        for value in ('Restart=no','KillMode=mixed','TimeoutStopSec=30','LoadCredential=dashscope_api_key:',
                      'Conflicts=dms.service ies.service','UMask=0077','--robot'):
            self.assertIn(value,text)
        self.assertNotIn('DASHSCOPE_API_KEY=',text)
        with self.assertRaises(ValueError):module.render(Path('/tmp/a b'),Path('/home/x'),'x',Path('/tmp/r'))

    def test_boot_device_readiness_uses_metadata_only(self):
        module=tool('wait_p1_devices');robot=unittest.mock.Mock()
        robot.motion_profile.return_value=dict(port='/dev/unit-servo',audio_device='Unit Speaker')
        robot.tracking.return_value=dict(camera=dict(device='/dev/unit-camera'))
        fake=dict(capture=dict(device='Unit Mic:'))
        with patch.object(module.Path,'read_text',return_value=json.dumps(fake)), \
             patch.object(module.Path,'exists',return_value=True),patch.object(module.os.path,'exists',return_value=True), \
             patch.object(module.os,'access',return_value=True):
            self.assertEqual(module.pending(robot,'Unit Mic / Unit Speaker'),[])
            self.assertEqual(module.pending(robot,'Unit Mic'),['音箱未枚举'])
