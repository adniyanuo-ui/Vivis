"""Prepared PCM/animation preserves audio clocks and never calls native A2F."""
from collections import deque
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import AsyncMock,Mock,patch

import numpy as np

from vivis.speech.fixed_replies import REPLIES,FORMAT,FixedReplyCache,digest,signature,validate_frames
from vivis.speech.prepare_fixed_replies import write_blob
from vivis.speech.mouth import SpeechMouth
from vivis.speech.realtime import VoiceSession
from vivis.speech.realtime_audio import VoiceBuffer
from vivis.speech.message_voice import MessageSpeaker


SESSION=dict(model='qwen-audio-3.1-realtime-plus',voice='longanyang_v3.1',speech_rate=1.,enable_speech_emotion=True)
ORDER=['channel'+str(i) for i in range(52)]


def bundle(folder):
    import wave
    samples=(np.sin(np.arange(4800)*.04)*5000).astype('<i2')
    wav=io.BytesIO()
    with wave.open(wav,'wb') as w:
        w.setparams((1,2,24000,0,'NONE','not compressed'));w.writeframes(samples.tobytes())
    npz=io.BytesIO();np.savez_compressed(npz,sample_indices=np.arange(0,3201,266,dtype='<i8'),
        weights=np.tile(np.linspace(0,1,13,dtype='<f4')[:,None],(1,52)))
    rows={}
    for name,text in REPLIES.items():
        audio,audio_sha=write_blob(folder,name,'.wav',wav.getvalue())
        animation,animation_sha=write_blob(folder,name,'.npz',npz.getvalue())
        rows[name]=dict(text=text,audio=audio,audio_sha256=audio_sha,animation=animation,animation_sha256=animation_sha)
    manifest=dict(schema_version=FORMAT,generation=signature(SESSION),order=ORDER,replies=rows)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return samples,manifest


def mouth():
    m=SpeechMouth.__new__(SpeechMouth);m.lock=threading.Lock();m.error=None;m.warning=None;m.order=ORDER
    m.responses={};m.current=None;m.heads=deque(maxlen=128);m.frame_limit=2000
    m.filters={};m.received={};m.enqueue=Mock();m.gated=0;m.prebuffer_timeouts=0
    return m


class FixedReplyTests(unittest.TestCase):
    def test_complete_bundle_has_same_pcm_and_rejects_foreign_voice_text_and_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);samples,_=bundle(folder);cache=FixedReplyCache(folder)
            self.assertEqual(cache.verify(SESSION,ORDER)['replies'],len(REPLIES))
            audio=cache.get(REPLIES['standby'],SESSION,ORDER)
            np.testing.assert_array_equal(np.rint(audio.samples*32768).astype('<i2'),samples)
            self.assertIs(audio,cache.get(REPLIES['standby'],SESSION,ORDER))
            self.assertIsNone(cache.get(REPLIES['standby'],dict(SESSION,voice='another'),ORDER))
            self.assertIsNone(cache.get(REPLIES['standby'],dict(SESSION,speech_rate=1.2),ORDER))
            self.assertIsNone(cache.get(REPLIES['standby'],SESSION,ORDER[::-1]))
            self.assertIsNone(cache.get('何工您好',SESSION,ORDER))

    def test_animation_uses16k_timestamps_for24k_pcm_and_rejects_partial_coverage(self):
        weights=np.zeros((13,52),dtype='f4');indices=np.arange(0,3201,266,dtype='i8')
        validate_frames(indices,weights,4800)
        with self.assertRaises(ValueError):validate_frames(indices//2,weights,4800)
        with self.assertRaises(ValueError):validate_frames(indices,weights+float('nan'),4800)

    def test_corruption_missing_files_changed_text_and_escape_paths_are_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);_,manifest=bundle(folder)
            (folder/manifest['replies']['standby']['audio']).write_bytes(b'bad')
            with self.assertRaises(ValueError):FixedReplyCache(folder).get(REPLIES['standby'],SESSION)
            manifest['replies']['stop']['audio']='../outside.wav'
            manifest['replies']['song_preparation']['text']='changed'
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):FixedReplyCache(folder).get(REPLIES['stop'],SESSION)
            self.assertIsNone(FixedReplyCache(folder).get(REPLIES['song_preparation'],SESSION))
            self.assertIsNone(FixedReplyCache(folder/'absent').get(REPLIES['stop'],SESSION))

    def test_cached_clip_needs_no_native_input_and_tracks_dac_pause_cancel_and_next_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);bundle(folder);audio=FixedReplyCache(folder).get(REPLIES['standby'],SESSION)
            m=mouth();m.install('cached',audio.mouth_order,audio.mouth_frames)
            p=VoiceBuffer();p.mouth=m;p.add('cached',b'\x10\x00'*4800);p.finish('cached')
            m.enqueue.assert_not_called()
            self.assertTrue(np.any(p.render(960,100.)))
            self.assertIsNone(m.weights(99.99));self.assertIsNotNone(m.weights(100.01))
            p.pause('cached');p.render(3840,100.02);held=p.count;p.render(960,100.1)
            self.assertEqual(p.count,held)
            p.cancel('cached');self.assertIsNone(m.weights(100.1));self.assertFalse(p.add('cached',b'\x10\x00'*20))
            m.coefficients=np.ones(63);m.add('next-live',b'\x10\x00'*240)
            self.assertEqual(m.enqueue.call_args_list[0].args[0]['kind'],'start')
            self.assertEqual(m.current,'next-live')


class CachedSpeechTests(unittest.IsolatedAsyncioTestCase):
    async def test_message_speaker_installs_cached_frames_before_first_pcm_and_never_generates(self):
        import asyncio
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);bundle(folder);audio=FixedReplyCache(folder).get(REPLIES['stop'],SESSION)
            m=mouth();p=VoiceBuffer();p.mouth=m;state=VoiceSession(p,SimpleNamespace(write=lambda *a,**k:None))
            tts=SimpleNamespace(generate=AsyncMock(return_value=audio));events=[]
            speaker=MessageSpeaker(None,p,state,lambda _:True,events.append,lambda *a,**k:None,tts=tts)
            with patch('vivis.speech.message_voice.pcm24',side_effect=AssertionError('cached PCM must not be re-encoded')):
                task=asyncio.create_task(speaker.say(REPLIES['stop'],{}))
                for _ in range(100):
                    if p.count==len(audio.samples)*2:break
                    await asyncio.sleep(.005)
                self.assertGreater(p.count,0);self.assertTrue(m.responses)
                m.enqueue.assert_not_called();p.render(p.count,time.monotonic())
                self.assertTrue(await asyncio.wait_for(task,1));self.assertEqual(events,[REPLIES['stop']])

    async def test_controller_farewell_uses_cache_without_reader_or_gpu(self):
        import asyncio
        import test_cloud_control as fixtures
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);assets=folder/'cache';assets.mkdir();bundle(assets)
            events=[];s=fixtures.CloudBackendTests().service(tmp,events);s.fixed_replies=FixedReplyCache(assets)
            s.active_voice=SESSION['voice'];s.mouth=mouth();s.player.mouth=s.mouth;s.enabled=True
            state=VoiceSession(s.player,s.log);s.cloud_state=state
            ticket=dict(id='cached-rest',generation=s.input_generation,connection=s.message_connection,target='standby',text=REPLIES['standby'])
            s.farewell=ticket;s.controller_reply_id='cached-rest';s.context.update(mode='standby',target='standby',ready=False,speech_ready=True)
            try:
                with patch('vivis.speech.trusted_reading.RealtimeReader',side_effect=AssertionError('cache must not synthesize')):
                    s.schedule_farewell(state)
                    for _ in range(100):
                        if s.player.count==4800*2:break
                        await asyncio.sleep(.005)
                    self.assertGreater(s.player.count,0);s.mouth.enqueue.assert_not_called()
                    s.player.render(s.player.count,time.monotonic())
                    await asyncio.wait_for(s.farewell_task,1)
                self.assertTrue(any(e.get('reason')=='played' for e in events))
                self.assertFalse(s.can_speak())
            finally:
                s.cancel_farewell('cleanup')
                if s.farewell_task:await asyncio.gather(s.farewell_task,return_exceptions=True)
                s.log.close()


if __name__=='__main__':unittest.main()
