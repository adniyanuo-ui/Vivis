"""Trusted result text -> existing Qwen TTS -> existing audio/AEC/A2F route."""
import asyncio
import math
import time
import uuid

import numpy as np
from scipy.signal import resample_poly

from vivis.interaction.message_lane import StaleMessageTurn


def pcm24(audio):
    """Encode a complete TTS clip off the receive/control event loop."""
    divisor=math.gcd(int(audio.sample_rate),24000)
    samples=resample_poly(audio.samples,24000//divisor,int(audio.sample_rate)//divisor)
    return np.rint(np.clip(samples,-1,1)*32767).astype('<i2').tobytes()


class MessageSpeaker:
    def __init__(self,engine,player,state,valid,emit,log):
        self.engine=engine;self.player=player;self.state=state;self.valid=valid;self.emit=emit;self.log=log

    async def say(self,text,request):
        rid='message_'+uuid.uuid4().hex;complete=False;announced=False
        response=dict(created=time.monotonic(),received_audio_s=0.,transcript=text,status='in_progress',
            source='trusted_message_tts',user_turn=None)
        self.state.responses[rid]=response
        try:
            for start in range(0,len(text),400):
                if not self.valid(request):raise StaleMessageTurn()
                audio=await asyncio.to_thread(self.engine.generate,text[start:start+400])
                if not self.valid(request):raise StaleMessageTurn()
                pcm=await asyncio.to_thread(pcm24,audio)
                for offset in range(0,len(pcm),4800):
                    part=pcm[offset:offset+4800]
                    while self.player.count+len(part)>self.player.maximum:
                        if not self.valid(request) or rid in self.state.blocked:raise StaleMessageTurn()
                        await asyncio.sleep(.02)
                    if not self.valid(request) or rid in self.state.blocked:raise StaleMessageTurn()
                    if not self.player.add(rid,part):raise StaleMessageTurn()
                    response['received_audio_s']+=len(part)/48000
                    # player.add also feeds A2F; even bounded packet work must
                    # yield between packets so mic/control/network can drain.
                    await asyncio.sleep(0)
            self.player.finish(rid);response['status']='completed'
            expected=round(response['received_audio_s']*48000)
            while True:
                self.state.poll_playback()
                if not self.valid(request) or rid in self.state.blocked:raise StaleMessageTurn()
                if not announced and rid in self.player.started_responses:
                    self.emit(text);announced=True
                if expected and self.player.consumed.get(rid,0)>=expected and response.get('empty_dac',float('inf'))<=time.monotonic():
                    complete=True;return True
                await asyncio.sleep(.02)
        finally:
            if not complete:self.state.interrupt(rid,source='message_speech_interrupted')
            self.log('message_tts_finished',response_id=rid,complete=complete)
