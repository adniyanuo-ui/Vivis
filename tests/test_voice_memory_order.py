"""Late cloud transcripts retain their original speaker, turn and session."""
import tempfile
import unittest

from test_cloud_control import config
from vivis.interaction.memory import PersonMemory
from vivis.interaction.robot_tasks import RobotTasks


class VoiceMemoryOrderTests(unittest.TestCase):
    def setUp(self):
        from test_cloud_control import CloudBackendTests
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.service=CloudBackendTests().service(temporary.name,[])
        self.addCleanup(self.service.log.close)
        self.service.memory=PersonMemory(config('configs/interaction/memory.json'),temporary.name)
        self.service.tasks=RobotTasks(self.service.memory,config('configs/interaction/tasks.json'))
        self.service.context.update(revision=1,mode='interaction',target='interaction',ready=True)
        self.service.enabled=True;self.service.presence.wake()
        self.person='alice';self.service.identified_person=lambda:self.person

    def transcript(self,item_id,text):
        self.service.remember(dict(type='conversation.item.input_audio_transcription.completed',
            item_id=item_id,transcript=text))

    def test_late_transcript_after_new_speaker_keeps_original_attribution_and_source(self):
        service=self.service;service.memory_session='first-visit';service.begin_user_turn('first')
        task=service.tasks.create(dict(action='create',kind='message',title='测试',content='准备资料',recipient_id='bob'),
            'alice','first-visit','first',None,{'alice':{},'bob':{}})
        self.person='bob';service.memory_session='second-visit';service.begin_user_turn('second')
        self.transcript('first','这是甲的留言。')
        rows=service.memory.search_all([])['matches']
        self.assertEqual((rows[0]['person'],rows[0]['session']),('alice','first-visit'))
        self.assertEqual(service.tasks.get(task['task_id'])['source_message_id'],rows[0]['id'])
        self.assertIsNone(service.turn_message_id);self.assertFalse(service.turn_saved)
        self.assertIsNone(service.pending_user_text)
        self.transcript('second','这是乙的话。')
        self.assertEqual(service.memory.search_all([])['matches'][0]['person'],'bob')

    def test_reordered_duplicate_transcripts_do_not_duplicate_or_replace_current_source(self):
        service=self.service;service.begin_user_turn('first');service.begin_user_turn('second')
        self.transcript('second','第二句话。');current=service.turn_message_id
        self.transcript('first','第一句话。');self.transcript('first','第一句话。')
        self.assertEqual(len(service.memory.search_all([])['matches']),2)
        self.assertEqual(service.turn_message_id,current)
        self.assertEqual(service.pending_user_text,'第二句话。')

    def test_explicit_wake_can_engage_its_original_turn_after_transcription(self):
        service=self.service;service.enabled=False;service.begin_user_turn('wake')
        self.transcript('wake','小帅，您好。')
        self.assertFalse(service.memory.search_all([])['matches'])
        service.turn_engaged=True;service.remember({})
        self.assertEqual(service.memory.search_all([])['matches'][0]['person'],'alice')
        self.assertTrue(service.turn_saved)

    def test_unknown_or_evicted_transcript_is_never_attached_to_current_person(self):
        service=self.service
        for index in range(130):service.begin_user_turn(str(index))
        self.assertEqual(len(service.memory_turns),128)
        self.transcript('0','过期的话。');self.transcript('unknown','身份不明的话。')
        self.assertFalse(service.memory.search_all([])['matches'])
        self.assertFalse(service.turn_saved)
        self.assertTrue(service.unattributed_source_notice)


if __name__=='__main__':unittest.main()
