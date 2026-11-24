"""Restore the tested board-reference output only, before opening its streams."""
import json
import re
import subprocess
from vivis.paths import ROOT


def restore_reference_mixer(device_name,profile=None,*,runner=None):
    if profile is None:profile=json.loads((ROOT/'configs/perception/board_reference_mixer.json').read_text())
    if (profile.get('schema_version')!='vivis-board-reference-mixer-1.0'
            or type(profile.get('enabled')) is not bool):raise ValueError('Invalid reference mixer profile')
    if not profile['enabled']:return dict(checked=False,reason='disabled')
    prefix=profile.get('device_prefix');card=profile.get('alsa_card');target=profile.get('playback_value')
    if (not isinstance(prefix,str) or not prefix or not isinstance(card,str)
            or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}',card)
            or type(target) is not int or not 0<=target<=65535):raise ValueError('Invalid reference mixer target')
    if not device_name.startswith(prefix):return dict(checked=False,reason='different_device')
    run=runner or subprocess.run
    def command(operation,control,value=None):
        args=['amixer','-c',card,operation,'name='+control]
        if value is not None:args.append(value)
        try:r=run(args,capture_output=True,text=True,check=True,timeout=2)
        except (OSError,subprocess.SubprocessError) as exc:
            raise RuntimeError('DOV reference mixer unavailable: '+type(exc).__name__) from None
        return r.stdout
    def read():
        volume=command('cget','PCM Playback Volume');switch=command('cget','PCM Playback Switch')
        limits=re.search(r'min=(-?\d+),max=(-?\d+)',volume)
        values=re.search(r'^\s*: values=(-?\d+),(-?\d+)\s*$',volume,re.M)
        enabled=re.search(r'^\s*: values=(on|off)\s*$',switch,re.M)
        if not limits or not values or not enabled:raise ValueError('Unexpected DOV reference controls')
        if not int(limits[1])<=target<=int(limits[2]):raise ValueError('Reference target outside device range')
        return dict(values=[int(values[1]),int(values[2])],enabled=enabled[1]=='on')
    before=read()
    if before['values']!=[target,target]:command('cset','PCM Playback Volume',f'{target},{target}')
    if not before['enabled']:command('cset','PCM Playback Switch','on')
    changed=before['values']!=[target,target] or not before['enabled']
    after=read() if changed else before
    if after!=dict(values=[target,target],enabled=True):raise RuntimeError('DOV reference mixer verification failed')
    return dict(checked=True,card=card,before=before,after=after,restored=changed)
