"""One source for live voice deadlines, separate from motion and persona timing."""
from functools import lru_cache
import json
import math
from vivis.paths import ROOT

@lru_cache(maxsize=1)
def policy():
    c=json.loads((ROOT/'configs/perception/voice_runtime.json').read_text())
    if c.pop('schema_version',None)!='vivis-voice-runtime-1.0':raise ValueError('Invalid voice runtime policy')
    required=set("uplink_max_age_s send_timeout_s callback_stall_s connect_timeout_s handshake_timeout_s close_timeout_s ping_interval_s ping_timeout_s controller_ack_s message_extract_s message_tts_operation_s playback_stall_s recovery_cooldown_s audio_close_s audio_open_s".split())
    if set(c)!=required:raise ValueError('Missing or unknown voice runtime deadline')
    for key,value in c.items():
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=120:
            raise ValueError('Invalid voice runtime deadline: '+key)
    return c

class VoiceTransportError(RuntimeError):
    """The current input turn is incomplete; reconnect, never replay its audio."""

class PlaybackStalled(RuntimeError):pass

class ProgressDeadline:
    def __init__(self,seconds,now):self.seconds=seconds;self.updated=now;self.value=None
    def observe(self,value,now):
        if value!=self.value:self.value=value;self.updated=now
        elif now-self.updated>=self.seconds:raise PlaybackStalled('Trusted speech made no playback progress')


async def bounded_send(ws,message):
    import asyncio
    return await asyncio.wait_for(ws.send(message),timeout=policy()['send_timeout_s'])
