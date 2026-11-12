"""Public courier-code delivery is separate from verified-person private memory."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from vivis.paths import ROOT
from vivis.interaction.memory import PersonMemory
from vivis.interaction.courier import CourierHandoffs
from vivis.speech.message_receipts import MessageReceipts,ReceiptRejected


def config(path):return json.loads((ROOT/path).read_text())


class CourierStoreTests(unittest.TestCase):
    def test_phone_suffix_matches_local_contact_without_face_or_full_phone(self):
        self.store.phone_last4={'alice':'8244'}
        for text in ('我要取尾号8244的快递。','我来取手机后四位八二四四的快递。'):
            self.assertEqual(self.store.lookup('',text)['code'],'4321')
        self.assertNotIn('code',self.store.lookup('','取尾号9999的快递'))
        self.assertNotIn('code',self.store.lookup('','我的取件码8244'))
        self.assertNotIn('code',self.store.lookup('','尾号8244的私人留言'))

    def test_ambiguous_suffix_never_releases_a_code(self):
        mid=self.memory.append('bob','owner','user','退件码5678')
        self.store.register('bob','乙工',['乙工'],'5678',[mid],source_reason='test')
        self.store.phone_last4={'alice':'8244','bob':'8244'}
        self.assertEqual(self.store.lookup('','取尾号8244的快递')['status'],'ambiguous_owner')

    def test_visit_query_uses_playback_evidence_not_assistant_claims(self):
        self.assertEqual(self.store.visit_query('alice')['status'],'no_spoken_record')
        self.store.mark_announced(self.row['id'],self.row['version'],'test')
        self.assertEqual(self.store.visit_query('alice')['status'],'spoken_record')
        self.assertNotIn('等待快递员确认',self.store.owner_query('alice')['spoken_receipt'])
        self.assertNotIn('收到',self.store.lookup('甲工','我来取甲工的快递')['spoken_receipt'])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.memory=PersonMemory(config('configs/interaction/memory.json'),self.tmp.name)
        self.store=CourierHandoffs(self.memory)
        self.mid=self.memory.append('alice','owner','user','我有件快递要退，取件码4321，快递员来了就告诉他。')
        self.row=self.store.register('alice','甲工',['甲工','测试甲'],'4321',[self.mid],source_reason='synthetic owner instruction')

    def test_stranger_gets_only_the_requested_explicitly_released_code(self):
        result=self.store.lookup('甲工','我来取甲工的快递，告诉我取件码。')
        self.assertEqual(result['code'],'4321')
        self.assertEqual(set(result),{'status','handoff_id','version','owner_label','code','spoken_receipt'})
        self.assertNotIn('source_ids',result);self.assertNotIn('aliases',result)

    def test_owner_must_be_stated_and_private_queries_never_return_code(self):
        for name,source in (('','我来取快递。'),('甲工','我来取快递。'),('甲工','取出甲工的私人留言。')):
            with self.subTest(source=source):self.assertNotIn('code',self.store.lookup(name,source))
        self.assertEqual(self.store.lookup('乙工','取乙工的快递。')['status'],'not_found')

    def test_replacement_never_falls_back_to_old_code_and_cancel_hides_it(self):
        mid=self.memory.append('alice','owner','user','退件取件码改为6789，快递员来时告诉他。')
        row=self.store.register('alice','甲工',['甲工'],'6789',[self.mid,mid],source_reason='explicit update')
        self.assertEqual(row['id'],self.row['id']);self.assertGreater(row['version'],self.row['version'])
        self.assertEqual(self.store.lookup('甲工','取甲工的退件。')['code'],'6789')
        self.store.cancel(row['id'],'alice')
        self.assertEqual(self.store.lookup('甲工','取甲工的退件。')['status'],'not_found')

    def test_fresh_process_store_keeps_code_and_ack_only_means_message_receipt(self):
        mid=self.memory.append('anonymous:'+'a'*32,'visit','user','取件码收到了。')
        result=self.store.acknowledge(self.row['id'],self.row['version'],mid)
        self.assertFalse(result['physical_collection_verified'])
        fresh=CourierHandoffs(PersonMemory(self.memory.c,self.tmp.name))
        self.assertEqual(fresh.get(self.row['id'])['status'],'completed')
        # Explicitly asking again may repeat the CURRENT authorized code; there
        # is no automatic reminder and no completed-code history fallback.
        self.assertEqual(fresh.lookup('甲工','请再说甲工的取件码。')['code'],'4321')

    def test_other_persons_source_and_obsolete_ack_cannot_change_record(self):
        wrong=self.memory.append('bob','other','user','取件码9999。')
        with self.assertRaises(ValueError):self.store.register('alice','甲工',['甲工'],'9999',[wrong],source_reason='wrong source')
        mid=self.memory.append('alice','owner','user','取件码改为6789。')
        self.store.register('alice','甲工',['甲工'],'6789',[mid],source_reason='new code')
        ack=self.memory.append('anonymous:'+'b'*32,'visit','user','收到。')
        with self.assertRaises(ValueError):self.store.acknowledge(self.row['id'],self.row['version'],ack)
        self.assertEqual(self.store.get(self.row['id'])['code'],'6789')

    def test_another_owner_cannot_cancel_and_latest_source_is_authoritative(self):
        with self.assertRaises(ValueError):self.store.cancel(self.row['id'],'bob')
        newer=self.memory.append('alice','owner','user','取件码是6789。')
        with self.assertRaises(ValueError):self.store.register('alice','甲工',['甲工'],'4321',[self.mid,newer],source_reason='stale code')
        self.assertEqual(self.store.get(self.row['id'])['code'],'4321')

    def test_owner_query_reads_separate_ledger_without_marking_delivery(self):
        result=self.store.owner_query('alice')
        self.assertEqual(result['status'],'pending');self.assertIn('4、3、2、1',result['spoken_receipt'])
        self.assertEqual(self.store.get(self.row['id'])['status'],'pending')
        self.assertEqual(self.store.owner_query('bob')['status'],'not_found')
        self.store.cancel(self.row['id'],'alice')
        result=self.store.owner_query('alice')
        self.assertEqual(result['status'],'cancelled');self.assertNotIn('4、3、2、1',result['spoken_receipt'])


class CourierRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_phone_query_cannot_be_stolen_by_private_message_lane(self):
        from unittest.mock import Mock
        s=self.service;s.courier.phone_last4={'alice':'8244'}
        s.pending_user_text='我要取尾号8244的快递。';s.speak_message=AsyncMock(return_value=True)
        s.speak_message_notice=AsyncMock(return_value=True);state=SimpleNamespace(interrupt=Mock(),generating=None)
        s.final_message(dict(type='conversation.item.input_audio_transcription.completed',item_id=s.turn_item_id),state)
        await s.message_lane.task
        self.assertIn('4、3、2、1',s.speak_message.call_args.args[0]);s.speak_message_notice.assert_not_called()
        self.assertEqual(s.courier.get(self.row['id'])['status'],'announced')
        s.final_message(dict(type='conversation.item.input_audio_transcription.completed',item_id=s.turn_item_id),state)
        s.speak_message.assert_awaited_once()

    async def test_public_lookup_clarifies_then_accepts_suffix_without_face(self):
        from unittest.mock import Mock
        s=self.service;s.courier.phone_last4={'alice':'8244'};s.speak_message=AsyncMock(return_value=True)
        state=SimpleNamespace(interrupt=Mock(),generating=None)
        self.assertTrue(s.final_courier_read('我是快递员，我来取快递。',state));await s.message_lane.task
        self.assertIn('后四位',s.speak_message.call_args.args[0])
        s.input_generation+=1;s.turn_item_id='suffix';s.pending_user_text='尾号8244'
        self.assertTrue(s.final_courier_read(s.pending_user_text,state));await s.message_lane.task
        self.assertIn('4、3、2、1',s.speak_message.call_args.args[0])

    async def test_interrupted_lookup_does_not_mark_delivery_or_modify_code(self):
        from unittest.mock import Mock
        s=self.service;s.speak_message=AsyncMock(return_value=False)
        s.final_courier_read(s.pending_user_text,SimpleNamespace(interrupt=Mock(),generating=None));await s.message_lane.task
        self.assertEqual(s.courier.get(self.row['id'])['status'],'pending')
        self.assertEqual(s.courier.get(self.row['id'])['code'],'4321')

    async def test_pickup_clarification_cannot_follow_another_speaker(self):
        from unittest.mock import Mock
        s=self.service;s.speak_message=AsyncMock(return_value=True)
        state=SimpleNamespace(interrupt=Mock(),generating=None)
        s.final_courier_read('我是快递员，我来取快递。',state);await s.message_lane.task
        s.input_generation+=1;s.turn_person='bob'
        self.assertFalse(s.final_courier_read('甲工',state))
        self.assertEqual(s.courier.get(self.row['id'])['status'],'pending')

    async def test_owner_visit_question_has_read_receipt_not_register_failure(self):
        from unittest.mock import Mock
        s=self.service;s.identified_person=lambda:'alice';s.turn_person='alice'
        s.pending_user_text='小帅，快递员快递员刚才来来过没。';s.speak_message=AsyncMock(return_value=True)
        state=SimpleNamespace(interrupt=Mock(),generating=None)
        s.final_message(dict(type='conversation.item.input_audio_transcription.completed',item_id=s.turn_item_id),state)
        await s.message_lane.task
        text=s.speak_message.call_args.args[0]
        self.assertIn('不能据此断定',text);self.assertNotIn('保存',text)
        self.assertEqual(s.courier.get(self.row['id'])['status'],'pending')

    async def test_model_lookup_waiting_for_final_is_superseded_once_by_public_read(self):
        from unittest.mock import Mock
        s=self.service;s.courier.phone_last4={'alice':'8244'};s.pending_user_text=None
        s.speak_message=AsyncMock(return_value=True)
        event=dict(name='courier_pickup',response_id='r',call_id='early-public',arguments=json.dumps(dict(action='lookup',owner_name='')))
        await s.tool(self.ws,event)
        self.assertIn('early-public',s.courier_waiters)
        s.pending_user_text='我要取尾号8244的快递。'
        s.final_message(dict(type='conversation.item.input_audio_transcription.completed',item_id=s.turn_item_id),SimpleNamespace(interrupt=Mock(),generating=None))
        await s.resume_courier_calls(self.ws);await s.message_lane.task
        s.speak_message.assert_awaited_once();self.assertIn('4、3、2、1',s.speak_message.call_args.args[0])
        self.assertEqual(s.courier.get(self.row['id'])['status'],'announced')

    def test_registration_never_enters_public_read_route(self):
        from vivis.interaction.courier import courier_read_intent
        for text in ('我有个快递，你见到快递员来了，告诉他取件码是3366。','把取件码改成4466','取消给快递员的转告'):
            self.assertIsNone(courier_read_intent(text))

    def setUp(self):
        from test_cloud_control import CloudBackendTests
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.service=CloudBackendTests().service(self.tmp.name,[]);self.addCleanup(self.service.log.close)
        s=self.service;s.memory=PersonMemory(config('configs/interaction/memory.json'),self.tmp.name);s.courier=CourierHandoffs(s.memory)
        s.knowledge.people={'alice':dict(display_name='测试甲',preferred_address='甲工')}
        mid=s.memory.append('alice','owner','user','快递员来时告诉他，我的退件取件码4321。')
        self.row=s.courier.register('alice','甲工',['甲工','测试甲'],'4321',[mid],source_reason='owner')
        s.context.update(revision=1,mode='interaction',target='interaction',ready=True);s.enabled=True;s.presence.wake()
        s.identified_person=lambda:None;s.turn_person='anonymous:'+'a'*32;s.turn_item_id='courier-turn';s.turn_ended_at=time.monotonic()
        s.pending_user_text='我来取甲工的快递，告诉我取件码。';s.response_revisions['r']=1
        self.sent=[]
        async def send(text):self.sent.append(json.loads(text))
        self.ws=SimpleNamespace(send=send)

    async def call(self,args,call_id='courier'):
        await self.service.tool(self.ws,dict(name='courier_pickup',response_id='r',call_id=call_id,arguments=json.dumps(args)))
        return json.loads(self.sent[-1]['item']['output'])

    async def test_early_tool_waits_for_same_final_source_and_commits_once(self):
        s=self.service;s.identified_person=lambda:'alice';s.turn_person='alice'
        s.begin_user_turn('final-code');s.turn_ended_at=time.monotonic();s.pending_user_text=None
        s.speak_message=AsyncMock(return_value=True)
        event=dict(name='courier_pickup',response_id='r',call_id='early',arguments=json.dumps(dict(action='register',code='4466')))
        await s.tool(self.ws,event)
        self.assertFalse(self.sent);self.assertEqual(s.courier.get(self.row['id'])['code'],'4321')
        s.remember(dict(type='conversation.item.input_audio_transcription.completed',item_id='final-code',transcript='退件取件码4466。'))
        await s.resume_courier_calls(self.ws);await s.message_lane.task
        self.assertTrue(json.loads(self.sent[-1]['item']['output'])['accepted'])
        row=s.courier.get(self.row['id']);self.assertEqual(row['code'],'4466')
        await s.tool(self.ws,event);await s.resume_courier_calls(self.ws)
        self.assertEqual(s.courier.get(self.row['id'])['version'],row['version'])
        s.speak_message.assert_awaited_once();self.assertFalse(s.courier_waiters)

    async def test_deferred_courier_call_never_crosses_a_new_turn(self):
        s=self.service;s.pending_user_text=None
        event=dict(name='courier_pickup',response_id='r',call_id='early',arguments=json.dumps(dict(action='lookup',owner_name='甲工')))
        await s.tool(self.ws,event);self.assertFalse(self.sent)
        s.input_generation+=1;s.pending_user_text='另一句话。'
        await s.resume_courier_calls(self.ws)
        self.assertEqual(json.loads(self.sent[-1]['item']['output'])['status'],'stale_user_turn')
        self.assertIsNone(s.message_lane);self.assertFalse(s.courier_waiters)

    async def test_deferred_courier_source_timeout_does_not_write_or_claim_success(self):
        s=self.service;s.pending_user_text=None;s.speak_message=AsyncMock(return_value=True)
        event=dict(name='courier_pickup',response_id='r',call_id='early',arguments=json.dumps(dict(action='register',code='4466')))
        await s.tool(self.ws,event);s.courier_waiters['early']['deadline']=0
        await s.resume_courier_calls(self.ws);await s.message_lane.task
        self.assertFalse(json.loads(self.sent[-1]['item']['output'])['accepted'])
        self.assertEqual(s.courier.get(self.row['id'])['code'],'4321')
        self.assertNotIn('已保存',s.speak_message.call_args.args[0])

    async def test_failed_receipt_has_one_fixed_notice_not_a_claim_or_retry_loop(self):
        s=self.service;s.unattributed_source_notice=True;s.speak_message_notice=AsyncMock(return_value=True)
        s.receipt_failure_notice();await s.message_lane.task
        text=s.speak_message_notice.call_args.args[0]
        self.assertIn('请求尚未完成',text);self.assertNotIn('已保存',text)
        self.assertNotIn('取件码尚未确认保存',text)
        self.assertIsNone(s.message_receipts.pending);self.assertFalse(s.continue_response)
        self.assertEqual(s.courier.get(self.row['id'])['code'],'4321')

    async def test_unknown_face_can_receive_code_but_cannot_read_private_memory(self):
        result=await self.call(dict(action='lookup',owner_name='甲工'))
        self.assertTrue(result['accepted']);self.assertFalse(result['face_identity_required'])
        self.assertEqual(result['result']['code'],'4321')
        await self.service.tool(self.ws,dict(name='search_person_memory',response_id='r',call_id='private',arguments='{"keywords":[]}'))
        self.assertFalse(json.loads(self.sent[-1]['item']['output'])['accepted'])

    async def test_pickup_then_short_owner_answer_works_without_face_identity(self):
        s=self.service;s.pending_user_text='我来取快递。'
        result=await self.call(dict(action='lookup'))
        self.assertEqual(result['result']['status'],'need_owner')
        s.input_generation+=1;s.turn_item_id='owner-answer';s.pending_user_text='甲工';s.turn_message_id=None
        result=await self.call(dict(action='lookup',owner_name='甲工'),'owner-answer')
        self.assertEqual(result['result']['code'],'4321')

    async def test_stranger_cannot_register_or_cancel_someone_elses_code(self):
        for index,args in enumerate((dict(action='register',code='9999'),dict(action='cancel',handoff_id=self.row['id']))):
            result=await self.call(args,'denied'+str(index));self.assertFalse(result['accepted'])
        self.assertEqual(self.service.courier.get(self.row['id'])['code'],'4321')

    async def test_owner_status_is_not_public_lookup_and_never_marks_announced(self):
        s=self.service
        denied=await self.call(dict(action='owner_query'))
        self.assertFalse(denied['accepted'])
        s.identified_person=lambda:'alice';s.turn_person='alice';s.pending_user_text='我的退件留言保存了吗？'
        allowed=await self.call(dict(action='owner_query'),'owner')
        self.assertTrue(allowed['accepted']);self.assertTrue(allowed['face_identity_required'])
        self.assertEqual(allowed['result']['status'],'pending');self.assertIsNone(s.courier_offer)

    async def test_error_notice_cannot_mark_a_previous_code_offer_delivered(self):
        from unittest.mock import AsyncMock
        s=self.service
        s.courier_offer=s.courier.lookup('甲工','我来取甲工的快递。')
        s.speak_message_notice=AsyncMock(return_value=True)
        result=await self.call(dict(action='owner_query'))
        await s.message_lane.task
        self.assertFalse(result['accepted']);s.speak_message_notice.assert_awaited_once()
        self.assertEqual(s.courier.get(self.row['id'])['status'],'pending')

    async def test_late_chat_courier_call_cannot_replace_claimed_fixed_message_flow(self):
        s=self.service;s.message_claimed=s.input_generation
        result=await self.call(dict(action='lookup',owner_name='甲工'))
        self.assertFalse(result['accepted']);self.assertEqual(result['status'],'stale_user_turn')
        self.assertIsNone(s.courier_offer);self.assertIsNone(s.message_lane)

    async def test_lookup_for_another_owner_cannot_acknowledge_previous_offer(self):
        s=self.service;await self.call(dict(action='lookup',owner_name='甲工'))
        s.pending_user_text='我来取乙工的快递。'
        result=await self.call(dict(action='lookup',owner_name='乙工'),'other-owner')
        self.assertEqual(result['result']['status'],'not_found');self.assertIsNone(s.courier_offer)
        s.pending_user_text='收到了。'
        result=await self.call(dict(action='acknowledge',handoff_id=self.row['id'],version=self.row['version']),'old-ack')
        self.assertFalse(result['accepted']);self.assertEqual(s.courier.get(self.row['id'])['status'],'pending')

    async def test_anonymous_receipt_ack_updates_only_the_offered_handoff(self):
        s=self.service;result=await self.call(dict(action='lookup',owner_name='甲工'))
        s.turn_item_id='ack';s.turn_ended_at=time.monotonic();s.pending_user_text='取件码收到了。';s.turn_message_id=None
        s.memory_turns['ack']=dict(person=s.turn_person,session='visit',engaged=True,valid=True,text=s.pending_user_text,message_id=None)
        result=await self.call(dict(action='acknowledge',handoff_id=self.row['id'],version=self.row['version']),'ack')
        self.assertTrue(result['accepted']);self.assertFalse(result['result']['physical_collection_verified'])
        self.assertEqual(s.courier.get(self.row['id'])['status'],'completed')

    async def test_ack_without_model_repeating_internal_version_uses_server_offer(self):
        s=self.service;await self.call(dict(action='lookup',owner_name='甲工'))
        s.turn_item_id='ack';s.turn_ended_at=time.monotonic();s.pending_user_text='收到了。';s.turn_message_id=None
        s.memory_turns['ack']=dict(person=s.turn_person,session='visit',engaged=True,valid=True,text=s.pending_user_text,message_id=None)
        result=await self.call(dict(action='acknowledge'),'ack-without-version')
        self.assertTrue(result['accepted']);self.assertEqual(s.courier.get(self.row['id'])['status'],'completed')

    async def test_explicit_old_version_ack_is_rejected(self):
        s=self.service;await self.call(dict(action='lookup',owner_name='甲工'))
        result=await self.call(dict(action='acknowledge',version=self.row['version']+1),'wrong-version')
        self.assertFalse(result['accepted']);self.assertEqual(s.courier.get(self.row['id'])['status'],'pending')

    async def test_code_transcript_must_match_even_when_spoken_as_chinese_digits(self):
        gate=MessageReceipts();gate.set_courier_text('取件码是4、3、2、1。',0);gate.bind('r',0)
        gate.filter(dict(type='response.audio_transcript.done',response_id='r',transcript='取件码是四三二一。'))
        gate=MessageReceipts();gate.set_courier_text('取件码是4、3、2、1。',0);gate.bind('r',0)
        with self.assertRaises(ReceiptRejected):gate.filter(dict(type='response.audio_transcript.done',response_id='r',transcript='取件码是四三二二。'))
