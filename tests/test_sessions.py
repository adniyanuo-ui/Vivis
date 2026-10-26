"""Each repeat starts from zero with independent startup and cleanup."""
import copy,unittest
from unittest.mock import patch
from vivis.runtime.robot import Robot

class IndependentRoundTests(unittest.TestCase):
    def run_main(self,repeat=2,fail_prepare=0,fail_unload=0):
        import json,tempfile
        from pathlib import Path
        from contextlib import ExitStack
        from unittest.mock import Mock
        import numpy as np
        import vivis.runtime.show as entry
        from vivis.paths import ROOT
        data={'profile':Robot().profile('0001')}
        ids=data['profile']['diagnostic_command_ids'];events=[];receivers=[];prepared=[];unloaded=[]
        def prepare(*args):
            report=args[8];prepared.append(report);events.append('prepare')
            report['homing']={'completed':True}
            if len(prepared)==fail_prepare:raise RuntimeError('Preparation failure')
        def receiver(config):
            r=Mock();receivers.append(r)
            r.wait_ready.side_effect=lambda:events.append('ready')
            r.close.side_effect=lambda:events.append('receiver_close')
            return r
        def play(*args,**kwargs):
            events.append('play')
            np.testing.assert_array_equal(args[2][0],np.full(len(args[1]),2047.))
            return dict(completed=True,full_clip_requested=True,audio_seconds_at_stop=.05,
                        sent_frames=3,skipped_source_frames=0)
        def unload(profile,report):
            unloaded.append(report);events.append('unload')
            if len(unloaded)==fail_unload:raise RuntimeError('Cleanup failure')
            report['tongue_disabled_after']=True
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);output=root/'run'
            stack.enter_context(patch.object(entry,'ROOT',root))
            stack.enter_context(patch.object(entry,'load_targets',return_value=(ids,np.full((3,len(ids)),2047.),60)))
            stack.enter_context(patch.object(entry,'prepare_round',side_effect=prepare))
            stack.enter_context(patch.object(entry,'TargetReceiver',side_effect=receiver))
            adapter=stack.enter_context(patch.object(entry,'MojiaPlaybackGaze'))
            stack.enter_context(patch.object(entry,'play',side_effect=play))
            stack.enter_context(patch.object(entry,'unload_tongue',side_effect=unload))
            stack.enter_context(patch('sys.argv',['show','--robot',str(Robot().path),'--work','0001',
                '--output',str(output),'--repeat',str(repeat),'--authorized-tongue-observation']))
            if fail_prepare or fail_unload:
                with self.assertRaisesRegex(RuntimeError,'Preparation failure|Cleanup failure'):entry.main()
            else:entry.main()
            result=json.loads((output/'summary.json').read_text())
            reports=[json.loads(p.read_text()) for p in sorted(output.glob('round_*/preparation.json'))]
            return result,reports,events,receivers,adapter.call_count

    def test_each_round_prepares_owns_fresh_receiver_and_controller_and_cleans_up(self):
        result,reports,events,receivers,controllers=self.run_main()
        self.assertTrue(result['completed']);self.assertEqual(len(result['rounds']),2)
        self.assertEqual(events,['prepare','ready','play','receiver_close','unload']*2)
        self.assertEqual(len(receivers),2);self.assertIsNot(receivers[0],receivers[1])
        self.assertEqual(controllers,2)
        self.assertTrue(all(r['completed'] and r['homing']['completed'] and r['tongue_disabled_after'] for r in reports))

    def test_second_preparation_failure_cleans_up_and_never_starts_second_or_third_song(self):
        result,reports,events,receivers,controllers=self.run_main(repeat=3,fail_prepare=2)
        self.assertFalse(result['completed']);self.assertEqual(len(result['rounds']),1)
        self.assertEqual(events.count('prepare'),2);self.assertEqual(events.count('play'),1)
        self.assertEqual(events.count('unload'),2)
        self.assertTrue(result['tongue_disabled_after'])
        self.assertFalse(reports[1]['completed']);self.assertEqual(reports[1]['error'],'Preparation failure')

    def test_unverified_cleanup_prevents_next_round_and_is_not_retried(self):
        result,reports,events,receivers,controllers=self.run_main(fail_unload=1)
        self.assertFalse(result['completed']);self.assertEqual(result['rounds'],[])
        self.assertEqual(events.count('prepare'),1);self.assertEqual(events.count('unload'),1)
        self.assertFalse(reports[0]['completed']);self.assertEqual(reports[0]['cleanup_error'],'Cleanup failure')
