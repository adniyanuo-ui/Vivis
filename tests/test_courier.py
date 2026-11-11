"""Public courier-code delivery is separate from verified-person private memory."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from vivis.paths import ROOT
from vivis.interaction.memory import PersonMemory
from vivis.interaction.courier import CourierHandoffs
from vivis.speech.message_receipts import MessageReceipts,ReceiptRejected


def config(path):return json.loads((ROOT/path).read_text())


class CourierStoreTests(unittest.TestCase):
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


class CourierRuntimeTests(unittest.IsolatedAsyncioTestCase):
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
