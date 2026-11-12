"""Never feed synthesized mutation claims to playback before receipt validation."""
import base64
import unittest
from vivis.speech.message_receipts import MessageReceipts,ReceiptRejected,RECEIPTS


class MessageReceiptTests(unittest.TestCase):
    def test_courier_smalltalk_does_not_demand_a_database_write_receipt(self):
        gate=MessageReceipts();gate.bind('chat',1,'user')
        gate.filter(dict(type='conversation.item.input_audio_transcription.completed',item_id='user',transcript='快递员今天很忙。'))
        event=dict(type='response.audio_transcript.done',response_id='chat',transcript='是啊，送件也挺辛苦。')
        self.assertEqual(gate.filter(event),[event])

    def setUp(self):
        self.g=MessageReceipts();self.g.set('failed',1);self.g.bind('r',1)
        self.audio=dict(type='response.audio.delta',response_id='r',delta=base64.b64encode(b'\0'*480).decode())

    def test_failed_write_cannot_play_a_false_success(self):
        self.assertEqual(self.g.filter(self.audio),[])
        with self.assertRaises(ReceiptRejected):
            self.g.filter(dict(type='response.audio_transcript.done',response_id='r',transcript='好的，已经办完了！'))

    def test_verified_failure_plays_only_after_the_exact_receipt(self):
        self.assertEqual(self.g.filter(self.audio),[])
        events=self.g.filter(dict(type='response.audio_transcript.done',response_id='r',transcript=RECEIPTS['failed']))
        self.assertEqual(events[0],self.audio)

    def test_server_done_is_held_until_transcript_and_released_last(self):
        self.g.filter(self.audio)
        done=dict(type='response.done',response=dict(id='r',status='completed'))
        self.assertEqual(self.g.filter(done),[])
        events=self.g.filter(dict(type='response.audio_transcript.done',response_id='r',transcript=RECEIPTS['failed']))
        self.assertEqual([e['type'] for e in events],['response.audio.delta','response.audio_transcript.done','response.done'])

    def test_transcript_before_audio_and_ordinary_chat_streaming(self):
        self.g.filter(dict(type='response.audio_transcript.done',response_id='r',transcript=RECEIPTS['failed']))
        self.assertEqual(self.g.filter(self.audio),[self.audio])
        ordinary=dict(self.audio,response_id='ordinary')
        self.assertEqual(self.g.filter(ordinary),[ordinary])

    def test_tool_chain_preserves_the_new_receipt_and_retry_is_bounded(self):
        self.g.set('created',1)
        self.g.filter(dict(type='response.done',response=dict(id='r',status='completed')))
        self.assertEqual(self.g.pending['text'],RECEIPTS['created'])
        self.g.bind('next',1);self.assertTrue(self.g.retry('next',1))
        self.g.bind('last',1);self.assertFalse(self.g.retry('last',1))

    def test_old_user_turn_cannot_retry_into_new_users_reply(self):
        self.assertFalse(self.g.retry('r',2))

    def test_empty_receipt_is_rejected_instead_of_stalling_confirmation(self):
        with self.assertRaises(ReceiptRejected):
            self.g.filter(dict(type='response.done',response=dict(id='r',status='completed')))

    def test_message_promise_without_any_tool_is_blocked_even_if_source_text_arrives_late(self):
        gate=MessageReceipts();gate.bind('r',1,'user')
        self.assertEqual(gate.filter(self.audio),[])
        self.assertEqual(gate.filter(dict(type='response.audio_transcript.done',response_id='r',transcript='放心，这事包在我身上！')),[])
        source=dict(type='conversation.item.input_audio_transcription.completed',item_id='user',transcript='帮我办两件事，让小王买一个uber。')
        released=gate.filter(source)
        self.assertEqual(released,[source]);self.assertTrue(gate.rejected)
        self.assertTrue(gate.retry('r',1));self.assertIn('先调用',gate.instruction())

    def test_ordinary_audio_waits_only_for_source_text_then_streams(self):
        gate=MessageReceipts();gate.bind('r',1,'user');gate.filter(self.audio)
        source=dict(type='conversation.item.input_audio_transcription.completed',item_id='user',transcript='今天天气真好。')
        self.assertEqual(gate.filter(source),[source,self.audio])
        self.assertEqual(gate.filter(self.audio),[self.audio])

    def test_unknown_object_clarification_is_not_success_and_applies_to_next_short_answer(self):
        gate=MessageReceipts();gate.source_turns['first']=1;gate.set('meaning',1)
        gate.filter(dict(type='conversation.item.input_audio_transcription.completed',item_id='first',transcript='让小王买一个uber'))
        self.assertTrue(gate.awaiting_details)
        gate.source_turns['next']=2
        gate.filter(dict(type='conversation.item.input_audio_transcription.completed',item_id='next',transcript='一个U盘'))
        gate.pending=None;gate.bind('r',2,'next')
        self.assertEqual(gate.responses['r']['kind'],'needs_tool')
        self.assertIn('还没有保存',RECEIPTS['meaning'])

    def test_chat_classification_cannot_bypass_commit_claim_check(self):
        gate=MessageReceipts();gate.allow_chat('old','user',1);gate.bind('r',1)
        gate.filter(self.audio)
        with self.assertRaises(ReceiptRejected):
            gate.filter(dict(type='response.audio_transcript.done',response_id='r',transcript='我记下了，我会告诉小王。'))

    def test_original_network_timestamp_survives_receipt_wait(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from vivis.speech.realtime import VoiceSession
        audio=dict(self.audio,_network_received_monotonic=123.)
        self.g.filter(audio)
        released=self.g.filter(dict(type='response.audio_transcript.done',response_id='r',transcript=RECEIPTS['failed']))
        player=SimpleNamespace(add=Mock(),count=0);state=VoiceSession(player,Mock())
        state.event(released[0])
        self.assertEqual(state.responses['r']['first_audio_received'],123.)

    def test_database_query_can_report_existing_status_without_claiming_a_new_write(self):
        gate=MessageReceipts();gate.allow_database_reply('lookup',1);gate.bind('r',1)
        gate.filter(self.audio)
        events=gate.filter(dict(type='response.audio_transcript.done',response_id='r',transcript='您之前的留言已经记录，收件人确认收到了，目前没有新留言。'))
        self.assertEqual(events[0],self.audio)

    def test_reading_a_list_cannot_authorize_a_new_message_promise(self):
        gate=MessageReceipts();gate.allow_database_reply('lookup',1);gate.bind('r',1)
        gate.filter(self.audio)
        with self.assertRaises(ReceiptRejected):
            gate.filter(dict(type='response.audio_transcript.done',response_id='r',transcript='我记下了，我会告诉小王。'))

    def test_long_read_only_list_has_a_separate_bound_from_short_write_receipts(self):
        gate=MessageReceipts();gate.allow_database_reply('lookup',1);gate.bind('r',1)
        audio=dict(self.audio,delta=base64.b64encode(b'\0'*48000*16).decode())
        self.assertEqual(gate.filter(audio),[])
        released=gate.filter(dict(type='response.audio_transcript.done',response_id='r',transcript='已有留言如下，收件人已经确认收到。'))
        self.assertEqual(released[0],audio)
        with self.assertRaises(ReceiptRejected):self.g.filter(audio)

    def test_internal_review_requires_a_tool_instead_of_spoken_clarification(self):
        gate=MessageReceipts();gate.require_tool(1);gate.bind('r',1)
        with self.assertRaises(ReceiptRejected):
            gate.filter(dict(type='response.audio_transcript.done',response_id='r',transcript='这是另一件事，请您重新说一遍。'))

    def test_actual_database_relay_keeps_its_streaming_path(self):
        gate=MessageReceipts();gate.allow_database_reply('prepare',1,relay=True);gate.bind('r',1)
        self.assertEqual(gate.filter(self.audio),[self.audio])

    def test_delegate_variants_are_candidates_not_automatic_execution(self):
        from vivis.speech.message_receipts import MESSAGE_REQUEST
        for text in ('你告诉他就好','帮我捎句话','跟小王说一声','让小王帮我买一个东西'):
            with self.subTest(text=text):self.assertTrue(MESSAGE_REQUEST.search(text))


if __name__=='__main__':unittest.main()
