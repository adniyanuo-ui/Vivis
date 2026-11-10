"""Local short-control timing, source rejection and independent event ordering."""
import unittest
from types import SimpleNamespace

import numpy as np

from vivis.speech.control_keywords import KeywordControl
from vivis.interaction.dialogue import EventGate


class Engine:
    def __init__(self,near=None,far=None):self.near=near or {};self.far=far or {};self.count=0
    def create_stream(self):
        index=self.count;self.count+=1
        return SimpleNamespace(words=self.near if index%2==0 else self.far,frame=0,ready=False,result='',
            accept_waveform=lambda *_:None)
    def is_ready(self,s):
        if not s.ready:s.frame+=1;s.ready=True;return True
        return False
    def decode_stream(self,s):s.result=s.words.get(s.frame,'')
    def get_result(self,s):return s.result
    def reset_stream(self,s):s.result=''


class ControlTests(unittest.TestCase):
    def config(self):return dict(confirmation_s=.25,reference_guard_s=.75,refractory_s=1.5)
    def replay(self,near,far=None,text='',wake_words=None):
        events=[];engine=Engine(near,far)
        # The fake accepts exactly one decode operation per input frame.
        control=KeywordControl(self.config(),events.append,engine,wake_words=wake_words)
        for i in range(60):
            control.near.ready=control.far.ready=False
            control.feed(np.zeros(320,np.float32),np.zeros(320,np.float32),10+i*.02,text)
        return events

    def test_stop_has_bounded_confirmation_and_does_not_wait_for_sentence_final(self):
        events=self.replay({5:'停止'},text='停止')
        actual=[e for e in events if e['kind']=='asr_control']
        self.assertEqual([e['text'] for e in actual],['停止'])
        self.assertLess(actual[0]['captured_end_monotonic']-actual[0]['keyword_detected_monotonic'],.3)

    def test_known_far_end_keyword_is_not_a_user_command(self):
        events=self.replay({12:'停止'},{8:'停止'},text='停止')
        self.assertFalse(any(e['kind']=='asr_control' for e in events))
        self.assertEqual(events[0]['reason'],'playback_reference')

    def test_repeated_candidates_do_not_postpone_first_stop_confirmation(self):
        actual=[e for e in self.replay({5:'停止',10:'停止',15:'停止'},text='停止') if e['kind']=='asr_control']
        self.assertEqual(len(actual),1);self.assertLess(actual[0]['captured_end_monotonic'],10.4)

    def test_recent_negative_and_question_block_candidates(self):
        for text in ['不要停止','不用退出','别结束','停止是什么意思','如果退出会怎样','怎么退出','为什么结束','他说停止']:
            with self.subTest(text=text):
                self.assertFalse(any(e['kind']=='asr_control' for e in self.replay({5:'停止'},text=text)))

    def test_onsite_solving_sentence_and_silence_never_become_stop_commands(self):
        for word,text in [('结束','解解决不了么'),('结束','解决不了问题怎么办'),('停止',''),('停止','听'),('停止','准备播放歌曲')]:
            with self.subTest(word=word,text=text):
                events=self.replay({5:word},text=text)
                self.assertFalse(any(e['kind']=='asr_control' for e in events))
                self.assertEqual(events[0]['reason'],'unconfirmed_streaming_intent')

    def test_confirmed_standby_and_exit_have_matching_semantics(self):
        for word,text in [('退出','请停止播放'),('待机','自己玩去吧')]:
            self.assertEqual(len([e for e in self.replay({5:word},text=text) if e['kind']=='asr_control']),1)
        self.assertFalse(any(e['kind']=='asr_control' for e in self.replay({5:'停止'},text='待机')))

    def test_confirmation_respects_configured_robot_wake_name(self):
        events=self.replay({5:'退出'},text='小白请退出吧',wake_words=['小白'])
        self.assertEqual(len([e for e in events if e['kind']=='asr_control']),1)

    def test_speech_and_keyword_sequence_spaces_are_independent_and_stale_events_reject(self):
        gate=EventGate(2.)
        a=dict(kind='asr_control',control_sequence=100,text='停止',captured_end_monotonic=10.)
        b=dict(kind='asr_final',utterance_id=1,text='待机',captured_end_monotonic=10.)
        self.assertTrue(gate.accept(a,10.1));self.assertTrue(gate.accept(b,10.1))
        self.assertFalse(gate.accept(a,10.2));self.assertFalse(gate.accept({**a,'control_sequence':101},13.))


if __name__=='__main__':unittest.main()
