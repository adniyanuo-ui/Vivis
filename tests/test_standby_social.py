"""Standby interaction, single-owner gaze and non-actuating scene defaults."""
from contextlib import contextmanager,redirect_stdout,redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from vivis.paths import ROOT
from vivis.motion.mapping import read_json
from vivis.runtime.robot import Robot
from vivis.interaction.standby import plan,StandbyGaze,main
from vivis.interaction.animation import FrameMapper


def settings():return read_json(ROOT/'configs/interaction/standby.json')
def robot(name='xiaoshuai'):return Robot(ROOT/f'configs/robots/{name}/robot.json')
def packet(now,strength=1.,target=True):
    return dict(schema_version='vivis-person-target-1.0',timestamp_monotonic=now,sequence=int(now*60),
        target=dict(x=.6,y=.35,eye_line_roll_rad=.1) if target else None,
        social_smile=dict(schema_version='vivis-smile-event-1.0',timestamp_monotonic=now,valid=True,strength=strength))


class StandbySocialTests(unittest.TestCase):
    def test_sustained_smile_stays_up_until_relax_then_holds_and_releases(self):
        from vivis.interaction.smile import SmileGate
        gate=SmileGate(settings()['social_response']['gate'])
        for i in range(301):
            cue=gate.step(.9,.9,i/60)
            if i>=60:self.assertAlmostEqual(cue['strength'],1.)
        # A prolonged smile never self-expires on the old pulse timer.
        for i in range(301,331):cue=gate.step(0.,0.,i/60)
        self.assertAlmostEqual(cue['strength'],1.)
        release=[]
        for i in range(331,451):release.append(gate.step(0.,0.,i/60)['strength'])
        self.assertTrue(all(b<=a for a,b in zip(release,release[1:])))
        self.assertEqual(release[-1],0.)

    def test_sustained_smile_tolerates_brief_missing_landmarks_but_not_permanent_loss(self):
        from vivis.interaction.smile import SmileGate
        gate=SmileGate(settings()['social_response']['gate'])
        for i in range(120):gate.step(.9,.9,i/60)
        for i in range(120,128):
            self.assertEqual(gate.step(0.,0.,i/60,valid=False)['strength'],1.)
        for i in range(128,188):
            self.assertEqual(gate.step(.9,.9,i/60)['strength'],1.)
        for i in range(188,368):cue=gate.step(0.,0.,i/60,valid=False)
        self.assertEqual(cue['strength'],0.)

    def test_legacy_song_smile_remains_a_pulse(self):
        from vivis.interaction.smile import SmileGate
        legacy=read_json(ROOT/'configs/interaction/p0.json')['social_smile']['gate']
        gate=SmileGate(legacy);peak=0.
        for i in range(300):
            cue=gate.step(.9,.9,i/60);peak=max(peak,cue['strength'])
        self.assertGreater(peak,.9);self.assertEqual(cue['strength'],0.)

    def test_camera_processing_pause_cannot_restore_a_stale_held_smile(self):
        from vivis.interaction.smile import SmileGate
        gate=SmileGate(settings()['social_response']['gate'],max_gap_s=.3)
        for i in range(120):gate.step(.9,.9,i/60)
        self.assertEqual(gate.envelope,1.)
        # Camera calls stopped; a fresh neutral face must not resurrect the smile.
        self.assertEqual(gate.step(0.,0.,10.)['strength'],0.)
        for i in range(1,20):self.assertEqual(gate.step(0.,0.,10.+i/60)['strength'],0.)

    def test_multi_face_standby_smiles_use_the_tracked_largest_roi_legacy_is_unchanged(self):
        from vivis.interaction.smile import selected_face_allowed,SmileGate
        standby=settings()['social_response'];legacy=read_json(ROOT/'configs/interaction/p0.json')['social_smile']
        self.assertFalse(selected_face_allowed(standby,0))
        self.assertTrue(selected_face_allowed(standby,5));self.assertFalse(selected_face_allowed(legacy,5))
        _,_,_,_,source,_=plan(robot(),settings(),2.)
        self.assertEqual(source.tracking['social_smile']['face_selection'],'largest')
        for config,responds in ((standby,True),(legacy,False)):
            gate=SmileGate(config['gate']);peak=0.
            for i in range(120):
                cue=gate.step(.9,.9,i/60,valid=selected_face_allowed(config,5))
                peak=max(peak,cue['strength'])
            self.assertEqual(peak>.9,responds)
        with self.assertRaises(ValueError):selected_face_allowed({'face_selection':'any'},5)

    def test_no_device_offline_checks_and_singing_response_off_by_default(self):
        from vivis.interaction.singing import main as singing
        with patch('subprocess.Popen',side_effect=AssertionError('model/device opened')), \
             patch('vivis.interaction.standby.TargetReceiver',side_effect=AssertionError('UDP opened')):
            for name in ('xiaoshuai','mojia'):
                with redirect_stdout(io.StringIO()) as out:main(['--robot',str(robot(name).path),'--check'])
                report=json.loads(out.getvalue());self.assertTrue(report['tracking_enabled']);self.assertTrue(report['smile_enabled'])
                self.assertEqual(report['smile_channel_gain'],.5)
                self.assertFalse(report['hardware_opened']);self.assertFalse(report['camera_opened']);self.assertFalse(report['a2f_used'])
            for options,expected in (([],False),(['--smile','on'],True)):
                with redirect_stdout(io.StringIO()) as out:singing(['--check',*options])
                self.assertEqual(json.loads(out.getvalue())['social_smile_enabled'],expected)

    def test_standby_uses_calibration_smile_without_song_articulation_mask(self):
        for name in ('xiaoshuai','mojia'):
            p,ids,home,mapper,source,_=plan(robot(name),settings(),1.)
            self.assertEqual(set(ids),{a['id'] for a in home['axes']})
            if name=='xiaoshuai':self.assertFalse({9,24,29}&set(ids));self.assertEqual(len(ids),30)
            base=mapper.map_weights([0.]*52).copy()
            idle=mapper.map_weights([0.]*52,1.)
            song=FrameMapper(mapper.order,mapper.shapes,p,ids,.3,scope='face',social_channels=mapper.social)
            singing=song.map_weights([0.]*52,1.)
            index=ids.index(10)
            self.assertEqual(singing[index],base[index])
            self.assertGreater(abs(idle[index]-base[index]),10.)
            for m in p['motors']:
                self.assertTrue(m['safe_min']<=idle[ids.index(m['id'])]<=m['safe_max'])

    def test_fresh_smile_moves_face_stale_invalid_or_no_face_smoothly_releases(self):
        for bad in ('stale','nan','wrong_stamp','no_face','invalid','wrong_schema'):
            _,ids,_,mapper,source,_=plan(robot(),settings(),2.)
            receiver=SimpleNamespace(max_age=.3,event=None,poll=lambda now:receiver.event)
            source.receiver=receiver
            for i in range(60):
                now=i/60;receiver.event=packet(now);source.sample(now,now)
            self.assertGreater(source.smile,.99);before=source.smile
            for i in range(60,240):
                now=i/60;receiver.event=packet(now)
                if bad=='stale':receiver.event['timestamp_monotonic']-=1
                elif bad=='nan':receiver.event['social_smile']['strength']=float('nan')
                elif bad=='wrong_stamp':receiver.event['social_smile']['timestamp_monotonic']-=1
                elif bad=='no_face':receiver.event['target']=None
                elif bad=='invalid':receiver.event['social_smile']['valid']=False
                else:receiver.event['schema_version']='old'
                goals=source.sample(now,now)
                if i==60:self.assertAlmostEqual(source.smile,before)
            self.assertLess(source.smile,.0001);self.assertGreater(source.smile_ticks,0)
            self.assertGreater(source.social_delta_by_id[10],10.)
            self.assertAlmostEqual(goals[ids.index(10)],mapper.neutral[ids.index(10)],places=2)

    def test_tracking_and_quiet_glances_share_one_controller_and_preserve_bounds(self):
        profile,ids,_,mapper,source,neutral=plan(robot(),settings(),30.)
        receiver=SimpleNamespace(max_age=.3,event=None,poll=lambda now:receiver.event)
        source.receiver=receiver;source.next_look=.01
        gaze=StandbyGaze(source.tracking,profile,ids,receiver,source)
        gaze.prepare(neutral,mapper.neutral,[],0.);gaze.set_audio_time(0.,30.)
        last=mapper.neutral.copy();m={x['id']:x for x in profile['motors']}
        tracked=False;glanced=False
        for i in range(1,601):
            now=i/60;visible=1.<now<3.
            receiver.event=packet(now) if visible else packet(now,0.,target=False)
            # Ideal encoder simulation only; no statement about real skin motion.
            for sid,value in zip(ids,last):gaze.observe(sid,dict(position=float(value)),now)
            base=source.sample(now,now);goal=gaze.apply(base,last,1/60,now)
            if 2.<now<3.:
                tracked|=abs(goal[ids.index(32)]-2047)>5
                self.assertLess(gaze.glance_blend,.02)
            if i==300:source.next_look=6.
            if now>8.:glanced|=gaze.glance_blend>.9 and abs(goal[ids.index(8)]-2047)>1
            for sid,value in zip(ids,goal):self.assertTrue(m[sid]['safe_min']<=value<=m[sid]['safe_max'])
            last=goal
        self.assertTrue(tracked);self.assertTrue(glanced)

    def test_default_p1_performance_disables_response_but_keeps_explicit_opt_in(self):
        from vivis.interaction.dialogue import program_command
        c=read_json(ROOT/'configs/interaction/p1.json');speech=read_json(ROOT/c['speech'])
        for kind in ('song','recitation','script'):
            # All content backed by the registered runtime_a2f path inherits off.
            program=dict(id=kind,kind='runtime_a2f',work='0001')
            cmd=program_command(program,robot(),Path('/tmp/scene'),speech,c)
            self.assertEqual(cmd[cmd.index('--smile')+1],'off')
            program['realtime_response']=True
            cmd=program_command(program,robot(),Path('/tmp/scene'),speech,c)
            self.assertEqual(cmd[cmd.index('--smile')+1],'on')
        standby=program_command(None,robot(),Path('/tmp/idle'),speech,c)
        self.assertIn('vivis.interaction.standby',standby);self.assertIn('--overlap-warmup',standby)

    def test_idle_overlay_does_not_conceal_an_out_of_bounds_tracking_target(self):
        profile,ids,_,mapper,source,_=plan(robot(),settings(),1.)
        gaze=StandbyGaze(source.tracking,profile,ids,SimpleNamespace(),source)
        invalid=mapper.neutral.copy();invalid[ids.index(32)]=9999
        with patch('vivis.motion.ownership.MojiaPlaybackGaze.apply',return_value=invalid):
            with self.assertRaisesRegex(ValueError,'exceeds command limits'):
                gaze.apply(mapper.neutral,mapper.neutral,1/60,1.)

    def test_camera_servo_preparation_and_cancel_cleanup_use_one_receiver(self):
        from vivis.motion.ownership import MojiaPlaybackGaze
        for fail in (False,True):
            closed=[];receiver=Mock(max_age=.3);receiver.poll.return_value=None
            @contextmanager
            def vision(*args,**kwargs):
                try:yield SimpleNamespace(check=lambda:None,wait_ready=lambda guard:guard())
                finally:closed.append('camera')
            def play(*args,**kwargs):
                self.assertIs(kwargs['frame_source'].receiver,receiver)
                self.assertIs(kwargs['gaze_override'].receiver,receiver)
                self.assertIsInstance(kwargs['gaze_override'],MojiaPlaybackGaze)
                self.assertIsNone(args[4])
                if fail:raise RuntimeError('Operator stop during scheduled feedback')
                return dict(completed=True)
            with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()), \
                 patch.object(Path,'is_file',return_value=True),patch('vivis.interaction.standby.check_competitors'), \
                 patch('vivis.interaction.standby.vision_service',side_effect=vision), \
                 patch('vivis.interaction.standby.TargetReceiver',return_value=receiver), \
                 patch('vivis.interaction.standby.prepare_round') as prepare, \
                 patch('vivis.interaction.standby.play',side_effect=play):
                folder=Path(tmp)/'run'
                result=main(['--seconds','1','--overlap-warmup','--output',str(folder)])
                self.assertEqual(result,130 if fail else 0);prepare.assert_called_once();receiver.close.assert_called_once()
                self.assertEqual(closed,['camera'])
                summary=json.loads((folder/'summary.json').read_text());self.assertTrue(summary['tracking_enabled'])
                self.assertTrue(summary['smile_enabled']);self.assertFalse(summary['a2f_used'])

    def test_legacy_no_camera_mode_and_invalid_tuning_rejected_before_devices(self):
        original=(ROOT/'configs/interaction/standby.json').read_bytes()
        with patch('subprocess.Popen',side_effect=AssertionError('opened')),redirect_stdout(io.StringIO()) as out:
            main(['--tracking','off','--smile','off','--check'])
        report=json.loads(out.getvalue());self.assertFalse(report['tracking_enabled']);self.assertFalse(report['smile_enabled'])
        for value in ('nan','-1','.6'):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(['--smile-strength',value,'--check'])
        self.assertEqual((ROOT/'configs/interaction/standby.json').read_bytes(),original)

    def test_failed_camera_readiness_never_prepares_or_moves_actuators(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()), \
             patch.object(Path,'is_file',return_value=True),patch('vivis.interaction.standby.check_competitors'), \
             patch('vivis.interaction.standby.vision_service',side_effect=RuntimeError('camera readiness failed')), \
             patch('vivis.interaction.standby.prepare_round') as prepare,patch('vivis.interaction.standby.play') as play:
            folder=Path(tmp)/'run'
            with self.assertRaisesRegex(RuntimeError,'camera readiness failed'):
                main(['--seconds','1','--output',str(folder)])
            prepare.assert_not_called();play.assert_not_called()
            self.assertFalse(json.loads((folder/'summary.json').read_text())['completed'])
