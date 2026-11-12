"""Persistent capture, speaker epochs and cloud fault isolation without hardware."""
import asyncio
import io
import json
from pathlib import Path
import queue
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from vivis.paths import ROOT
from vivis.interaction.voice_bridge import VoiceBridge
from vivis.speech.managed import ManagedVoice,voice_configuration,session_with_program_context


def speech_config():
    speech=json.loads((ROOT/'configs/perception/speech.json').read_text())
    speech.update(managed_voice=json.loads((ROOT/'configs/interaction/p1.json').read_text())['realtime_voice'],
                  programs=str(ROOT/'configs/interaction/programs.json'))
    speech['managed_voice']['control_backend']='local'  # Exercise the retained explicit legacy path.
    speech['managed_voice']['lip_sync']['enabled']=False  # Keep model work out of offline tests.
    return speech


class BridgeTests(unittest.TestCase):
    def test_release_needs_current_epoch_ack_and_failure_does_not_auto_retry(self):
        stream=io.StringIO();bridge=VoiceBridge(stream)
        self.assertTrue(bridge.released)
        bridge.enable(True);self.assertFalse(bridge.released)
        bridge.accept(dict(kind='voice_state',epoch=1,status='ready',speaker_owned=True))
        bridge.enable(False)
        self.assertFalse(bridge.accept(dict(kind='voice_state',epoch=1,status='off',speaker_owned=False)))
        self.assertFalse(bridge.released)
        bridge.accept(dict(kind='voice_state',epoch=2,status='off',speaker_owned=False));self.assertTrue(bridge.released)
        bridge.enable(True);bridge.accept(dict(kind='voice_state',epoch=3,status='unavailable',speaker_owned=False))
        before=stream.getvalue();bridge.enable(True);self.assertEqual(stream.getvalue(),before)
        bridge.enable(False);bridge.enable(True);self.assertEqual(bridge.epoch,4)

    def test_voice_configuration_uses_robot_identity_and_catalog_but_exposes_no_tools(self):
        config,speaker,url,session=voice_configuration(speech_config(),ROOT/'configs/robots/xiaoshuai/robot.json')
        self.assertEqual(config['frontend']['aec3']['profile'],'voice-preserve')
        self.assertEqual(config['output_gain'],1.)
        self.assertIs(session['enable_search'],False)
        self.assertEqual(session['tools'],[])
        self.assertIn('白骨夫人',session['instructions'])
        self.assertIn('小帅',session['instructions'])
        self.assertIn('只介绍能力类别',session['instructions'])
        self.assertIn('唱歌、演剧本、学表情',session['instructions'])
        self.assertNotIn('你正在做纯语音测试',session['instructions'])
        self.assertIn('宇飞剧本',session['instructions'])
        self.assertNotIn('"title": "魏总剧本表演"',session['instructions'])
        self.assertIn('"artist": "李健"',session['instructions'])
        self.assertIn('想要什么给你',session['instructions'])
        self.assertIn('You Are Beautiful',session['instructions'])
        self.assertIn('不是举例',session['instructions'])
        self.assertIn('"category": "song"',session['instructions'])
        self.assertIn('"category": "script"',session['instructions'])
        self.assertIn('不要虚构其他流行歌曲',session['instructions'])

    def test_controller_result_reaches_next_voice_epoch_without_inventing_playback(self):
        stream=io.StringIO();bridge=VoiceBridge(stream)
        bridge.last_program=dict(id='wz2',title='宇飞剧本',outcome='completed')
        bridge.enable(True)
        command=json.loads(stream.getvalue())
        self.assertEqual(command['last_program'],bridge.last_program)
        base=dict(instructions='基础设定')
        current=session_with_program_context(base,command['last_program'])
        self.assertEqual(base['instructions'],'基础设定')
        self.assertIn('宇飞剧本',current['instructions'])
        self.assertIn('completed',current['instructions'])
        self.assertIn('不是播放请求',current['instructions'])
        for context in ({**bridge.last_program,'outcome':'requested'}, {'title':'宇飞剧本'}, {**bridge.last_program,'title':'a'*101}):
            with self.assertRaises(ValueError):session_with_program_context(base,context)


class FakeAudio:
    def __init__(self,*_,**__):
        self.output_stream=None;self.output_opened_at=None;self.last_capture=None;self.last_output=None
        self.fault=None;self.uplink=queue.Queue();self.meters=queue.Queue();self.stats={}
        self.opens=0;self.leases=0;self.releases=0;self.closed=False
    def open(self,with_output):
        if with_output:raise AssertionError('Standby must not own speaker')
        self.opens+=1;return dict(microphone='fake',speaker='fake')
    def enable_output(self,player):
        if self.output_stream is not None:raise AssertionError('Overlapping speaker owner')
        self.player=player;self.output_stream=object();self.output_opened_at=time.monotonic();self.leases+=1
    def disable_output(self):
        if self.output_stream is not None:self.releases+=1
        self.output_stream=None
    def close(self):self.closed=True;self.disable_output()


class ManagedVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_capture_survives_cloud_failure_and_mode_changes_release_speaker(self):
        events=[];feed_count=[];context=[];closing=[];fail_cloud=[False];connected_sessions=[]
        class Decoder:
            def __init__(self,*_):pass
            def feed(self,*args,**kwargs):feed_count.append(time.monotonic())
            def close(self):pass
        class WS:
            async def __aenter__(self):context.append('opened');return self
            async def __aexit__(self,*_):
                closing.append(time.monotonic());await asyncio.sleep(.8)
            async def send(self,*_):pass
            def __aiter__(self):
                async def data():
                    while not fail_cloud[0]:await asyncio.sleep(.01)
                    raise RuntimeError('simulated network failure')
                    yield ''
                return data()
        async def ready(_ws,session,_log):connected_sessions.append(session)
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{'DASHSCOPE_API_KEY':'unit-test-placeholder'}), \
             patch('vivis.speech.managed.AudioIO',FakeAudio),patch('vivis.speech.managed.Decoder',Decoder), \
             patch('vivis.speech.managed.connect_session',side_effect=ready), \
             patch('vivis.speech.managed.websocket_connection',side_effect=lambda *_:WS()):
            service=ManagedVoice(speech_config(),ROOT/'configs/robots/xiaoshuai/robot.json',Path(tmp),events.append)
            task=asyncio.create_task(service.run())
            async def capture():
                while not service.shutdown.is_set():
                    stamp=time.monotonic();service.audio.last_capture=stamp
                    if service.audio.output_stream is not None:service.audio.last_output=stamp
                    service.audio.uplink.put((stamp,b'\0'*640,b'\0'*640,b'\0'*640))
                    await asyncio.sleep(.02)
            producer=asyncio.create_task(capture())
            async def until(predicate):
                deadline=time.monotonic()+2
                while not predicate():
                    if task.done():task.result()
                    if time.monotonic()>deadline:raise AssertionError('Managed voice timed out')
                    await asyncio.sleep(.005)
            try:
                await until(lambda:any(e['kind']=='ready' for e in events))
                self.assertEqual(context,[]);self.assertEqual(service.audio.leases,0)
                service.controls.put(dict(kind='voice_mode',enabled=True,epoch=1))
                await until(lambda:any(e.get('status')=='ready' for e in events))
                self.assertEqual(service.audio.opens,1);self.assertEqual(service.audio.leases,1)
                service.controls.put(dict(kind='voice_mode',enabled=False,epoch=2))
                await until(lambda:any(e.get('epoch')==2 and e.get('status')=='off' for e in events))
                self.assertIsNone(service.audio.output_stream)
                service.controls.put(dict(kind='voice_mode',enabled=True,epoch=3,
                    last_program=dict(id='wz2',title='宇飞剧本',outcome='completed')))
                await until(lambda:any(e.get('epoch')==3 and e.get('status')=='ready' for e in events))
                self.assertEqual(service.audio.opens,1);self.assertEqual(service.audio.leases,2)
                self.assertNotIn('控制器已确认的最近一次节目记录',connected_sessions[0]['instructions'])
                self.assertIn('宇飞剧本',connected_sessions[1]['instructions'])
                self.assertIn('控制器已确认的最近一次节目记录',connected_sessions[1]['instructions'])
                self.assertIn('completed',connected_sessions[1]['instructions'])
                # A network failure must leave local frames and control alive.
                fail_cloud[0]=True
                await until(lambda:any(e.get('epoch')==3 and e.get('status')=='unavailable' for e in events))
                before=len(feed_count);await asyncio.sleep(.06)
                self.assertGreater(len(feed_count),before);self.assertFalse(task.done())
                self.assertTrue(any(e['kind']=='cloud_error' for e in events))
                self.assertTrue(any(e['kind']=='cloud_error' and 'simulated network failure' in e['error'] for e in events))
                self.assertFalse(any(e['kind']=='cloud_error' and 'queue full' in e['error'].lower() for e in events))
            finally:
                service.shutdown.set();await producer;await asyncio.wait_for(task,3.)
            self.assertTrue(service.audio.closed);self.assertEqual(service.summary['microphone_opens'],1)
            self.assertFalse(any(e['kind']=='asr_error' for e in events))


if __name__=='__main__':unittest.main()
