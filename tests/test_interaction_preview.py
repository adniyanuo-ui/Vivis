import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from vivis.vision.interaction_log import interaction_snapshot
from vivis.vision.preview import Preview, PAGE


class InteractionPreviewTests(unittest.TestCase):
    def fixture(self, root, name='session'):
        folder = root / 'outputs/p1' / name
        folder.mkdir(parents=True)
        runtime = root / 'outputs/runtime'
        runtime.mkdir(exist_ok=True)
        (runtime / 'program_status.json').write_text(json.dumps(dict(
            output=str(folder), updated_monotonic=time.monotonic(), phase='interaction')))
        return folder / 'events.jsonl'

    def test_projection_partial_write_and_sensitive_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.fixture(root)
            events = [dict(kind='cloud_transcript', detail=dict(role='user', text='保存了吗？', identity_label='小王-other', secret='hidden')),
                      dict(kind='cloud_transcript', detail=dict(role='assistant', text='<script>alert(1)</script>')),
                      dict(kind='semantic_config', config=dict(api_key='hidden')),
                      dict(kind='preparation', last_message='SELF-CHECK PASS')]
            path.write_bytes(b''.join((json.dumps(dict(monotonic=time.monotonic(), **e))+'\n').encode() for e in events)
                             + b'{"partial":')
            result = interaction_snapshot(root)
            self.assertTrue(result['active'])
            self.assertEqual(len(result['entries']), 3)
            self.assertEqual(result['entries'][0]['role'], '小王-other')
            self.assertEqual(result['entries'][1]['role'], '小帅')
            self.assertNotIn('hidden', json.dumps(result))
            self.assertIn('content.textContent=entry.text', PAGE)
            self.assertNotIn('innerHTML', PAGE)

    def test_missing_outside_session_and_symlink_are_not_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertFalse(interaction_snapshot(root)['available'])
            path = self.fixture(root)
            outside = root / 'secret.jsonl'
            outside.write_text('{}\n')
            path.symlink_to(outside)
            self.assertFalse(interaction_snapshot(root)['available'])
            status = root / 'outputs/runtime/program_status.json'
            status.write_text(json.dumps(dict(output=str(root), updated_monotonic=time.monotonic())))
            self.assertFalse(interaction_snapshot(root)['available'])

    def test_bounded_history_and_current_session_switch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.fixture(root)
            path.write_text(''.join(json.dumps(dict(kind='cloud_transcript',monotonic=time.monotonic(),
                detail=dict(role='user',text=str(i)+'x'*4000)))+'\n' for i in range(300)))
            result = interaction_snapshot(root)
            self.assertEqual(len(result['entries']), 100)
            self.assertTrue(result['entries'][-1]['text'].startswith('299'))
            self.fixture(root, 'new_session').write_text('')
            result = interaction_snapshot(root)
            self.assertEqual(result['session'], 'new_session')
            self.assertEqual(result['entries'], [])

    def test_read_only_http_and_no_cache(self):
        with patch('vivis.vision.preview.interaction_snapshot', return_value=dict(entries=[], status='待机')):
            preview = Preview('127.0.0.1', 0)
            try:
                base = 'http://127.0.0.1:'+str(preview.server.server_address[1])
                with urlopen(base+'/interaction-log', timeout=2) as response:
                    self.assertEqual(json.load(response)['status'], '待机')
                    self.assertEqual(response.headers['Cache-Control'], 'no-store')
                with urlopen(base+'/', timeout=2) as response:
                    self.assertIn('交互日志', response.read().decode())
            finally:
                preview.close()

    def test_current_business_identity_is_separate_from_camera_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);events=self.fixture(root);events.write_text('')
            voice=events.parent/'voice';voice.mkdir()
            path=voice/'identity.json'
            path.write_text(json.dumps(dict(observed_monotonic=time.monotonic(),label='甲工',state='reviewing',identity_label='甲工-other')))
            self.assertEqual(interaction_snapshot(root)['identity_status'],'身份（逻辑-实时）：甲工-other')
            path.write_text(json.dumps(dict(observed_monotonic=0,label='旧身份',state='reviewing')))
            self.assertNotIn('identity_status',interaction_snapshot(root))
