"""Reboot mixer restoration must not change capture or unrelated devices."""
import subprocess
from types import SimpleNamespace
import unittest

from vivis.speech.reference_mixer import restore_reference_mixer


PROFILE=dict(schema_version='vivis-board-reference-mixer-1.0',enabled=True,device_prefix='DOV:',alsa_card='DOV',playback_value=999)


class MixerTests(unittest.TestCase):
    def runner(self,values=(0,0),enabled=True,ignore_write=False):
        self.values=list(values);self.enabled=enabled;self.commands=[]
        def run(args,**kw):
            self.commands.append(args)
            self.assertEqual(args[:3],['amixer','-c','DOV']);self.assertEqual(kw['timeout'],2)
            op,control=args[3:5]
            self.assertIn(control,('name=PCM Playback Volume','name=PCM Playback Switch'))
            if op=='cset' and not ignore_write:
                if control.endswith('Volume'):self.values=list(map(int,args[5].split(',')))
                else:self.enabled=args[5]=='on'
            value=('; min=0,max=999\n : values='+','.join(map(str,self.values))) if control.endswith('Volume') else ': values='+('on' if self.enabled else 'off')
            return SimpleNamespace(stdout=value+'\n')
        return run

    def test_reboot_zero_restores_and_independently_reads_back(self):
        report=restore_reference_mixer('DOV: USB Audio',PROFILE,runner=self.runner())
        self.assertTrue(report['restored']);self.assertEqual(report['after']['values'],[999,999])
        writes=[a for a in self.commands if a[3]=='cset']
        self.assertEqual(writes,[['amixer','-c','DOV','cset','name=PCM Playback Volume','999,999']])
        self.assertEqual(len(self.commands),5)

    def test_healthy_mixer_only_reads(self):
        self.assertFalse(restore_reference_mixer('DOV:',PROFILE,runner=self.runner((999,999)))['restored'])
        self.assertTrue(all(a[3]=='cget' for a in self.commands))

    def test_switch_off_restores_only_switch(self):
        r=restore_reference_mixer('DOV:',PROFILE,runner=self.runner((999,999),False))
        self.assertTrue(r['after']['enabled']);self.assertEqual(sum(a[3]=='cset' for a in self.commands),1)

    def test_failed_verification_is_not_claimed_successful(self):
        with self.assertRaisesRegex(RuntimeError,'verification failed'):
            restore_reference_mixer('DOV:',PROFILE,runner=self.runner(ignore_write=True))

    def test_non_board_device_and_disabled_profile_never_run(self):
        def fail(*a,**kw):self.fail('No mixer operation allowed')
        self.assertFalse(restore_reference_mixer('board',PROFILE,runner=fail)['checked'])
        self.assertFalse(restore_reference_mixer('DOV:',dict(PROFILE,enabled=False),runner=fail)['checked'])

    def test_missing_or_stalled_control_uses_output_recovery_error(self):
        def fail(*a,**kw):raise subprocess.TimeoutExpired(a[0],2)
        with self.assertRaisesRegex(RuntimeError,'mixer unavailable'):
            restore_reference_mixer('DOV:',PROFILE,runner=fail)

    def test_range_and_shell_targets_rejected(self):
        for change in (dict(alsa_card='DOV;bad'),dict(playback_value=1000)):
            with self.assertRaises(ValueError):restore_reference_mixer('DOV:',dict(PROFILE,**change),runner=self.runner())


if __name__=='__main__':unittest.main()
